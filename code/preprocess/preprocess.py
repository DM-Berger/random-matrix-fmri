from __future__ import annotations

import json
import json as json_
import os
import re
import shutil
import sys
import traceback
from abc import ABC
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple
from warnings import warn

import ants
import numpy as np
import gc
from ants import ANTsImage, image_read, motion_correction, resample_image
from ants.registration import reorient_image
from nipype.interfaces.base.support import InterfaceResult
from nipype.interfaces.fsl import BET, SliceTimer
from numpy import ndarray
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
if os.environ.get("CC_CLUSTER") == "niagara":
    ROOT = Path("/scratch/j/jlevman/dberger/random-matrix-fmri")
    DATA = Path("/scratch/j/jlevman/dberger/random-matrix-fmri/data")

TEMPLATE = DATA / "tpl-MNI152NLin2009aAsym_res-1_T1w.nii.gz"
TEMPLATE_MASK = DATA / "tpl-MNI152NLin2009aAsym_res-1_desc-brain_mask.nii.gz"
if not TEMPLATE.exists():
    raise FileNotFoundError(
        f"No registration template found at {TEMPLATE}. "
        f"Run `python {DATA / 'download_template.py'} to download it."
    )

NII_SUFFIX = ".nii.gz"
MASK_SUFFIX = "_mask.nii.gz"
MINMASK_SUFFIX = "_mask-min.nii.gz"
EXTRACTED_SUFFIX = "_extracted.nii.gz"
SLICETIME_SUFFIX = "_slicetime-corrected.nii.gz"
MOTION_CORRECTED_SUFFIX = "_motion-corrected.nii.gz"
ANAT_REGISTERED_SUFFIX = "_anat-reg.nii.gz"
MNI_REGISTERED_SUFFIX = "_mni-reg.nii.gz"


def min_mask_path_from_mask_path(mask: Path) -> Path:
    return Path(str(mask).replace(MASK_SUFFIX, MINMASK_SUFFIX))


class Loadable(ABC):
    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source: Path = source

    def load(self) -> ANTsImage:
        return image_read(str(self.source))


class FmriScan(Loadable):
    def __init__(self, source: Path) -> None:
        super().__init__(source)
        self.source: Path
        self.t1w_source: Path
        self.sid: str
        self.ses: Optional[str]
        self.run: Optional[str]
        self.local_json: Optional[Path]
        self.global_json: Optional[Path]
        self.slicetimes: List[float]
        self.repetition_time: float
        self.slicetime_file: Path

        self.source = source
        self.sid, self.ses, self.run = self.parse_source()
        self.local_json, self.global_json = self.find_json_file()
        self.t1w_source = self.find_t1w_file()
        (
            self.slicetimes,
            self.slicetime_file,
            self.repetition_time,
        ) = self.write_out_slicetimes()

    def brain_extract(self, force: bool = False) -> BrainExtracted:
        """Computes a 4D mask for input fMRI.

        Notes
        -----
        Brain-Extracted Data:

        Non-Extracted:
            Parkinsons
            Learning
            Bilinguality
            Depression
            Osteo

        The ANTS computed mask is truly 4D, e.g. the mask at t=1 will not in general be identical to
        the mask at time t=2. We do unforunately need this to get a timeseries at each voxel, and so
        also must define a `min_mask` (or max_mask) for some purposes.

        ants.get_mask seems to be single-core, so it is extremely worth parallelizing brain
        extraction across subjects
        """
        outfile = self.mask_path
        if outfile.exists():
            if not force:
                return BrainExtracted(self)
            os.remove(outfile)

        infile = self.source
        if os.environ.get("CC_CLUSTER") == "niagara":
            DATA = Path(
                "/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri/data/updated"
            )

        cmd = BET()
        cmd.inputs.in_file = str(self.source.resolve())
        cmd.inputs.out_file = str(self.extracted_path.resolve())
        cmd.inputs.output_type = "NIFTI_GZ"
        cmd.inputs.functional = True
        # cmd.inputs.frac = 0.9  # default with functional is 0.3, leaves too much skull
        if "Vigil" in str(self.source):
            cmd.inputs.frac = 0.3  # default with functional is 0.3, leaves too much skull
        if "Learning" in str(self.source):
            cmd.inputs.frac = 0.7  # default with functional is 0.3, leaves too much skull
        if "Osteo" in str(self.source):
            cmd.inputs.frac = 0.5  # default with functional is 0.3, leaves too much skull
        else:
            cmd.inputs.frac = 0.7  # default with functional is 0.3, leaves too much skull
        cmd.inputs.mask = True

        print(f"Computing mask for {self.source}")
        results = cmd.run()
        maskfile = Path(results.outputs.mask_file).resolve()  # type: ignore
        maskfile.rename(Path(str(maskfile).replace("extracted_", "")))
        print(f"Wrote brain mask to {self.mask_path}")
        print(f"Wrote extracted brain to {self.extracted_path}")

        # dunno, but ANTs causing huge memory usage here
        cmd = None
        results = None
        gc.collect()

        print(f"Cleaning up {self.extracted_path} with ANTS get_mask...")
        img = image_read(str(self.extracted_path))
        if "Vigil" in str(self.source):
            # too slow on this data for some reason
            avg = img.ndimage_to_list()[0].new_image_like(img.mean(axis=-1))
            mask3d = avg.get_mask()
            mask4d = np.stack([mask3d.numpy() for _ in range(img.shape[-1])], axis=-1)
            mask = img.astype("uint8").new_image_like(mask4d)
        else:
            mask = img.get_mask()
        img = img.mask_image(mask)
        ants.image_write(img, str(self.extracted_path))

        # print(f"Loading {self.source}")
        # img: ANTsImage = image_read(str(self.source))
        # # img = self.reorient_4d_to_RAS(img)
        # print(f"Computing mask for {self.source}")
        # mask = ants.get_mask(img)
        # # min_mask_frame = mask.ndimage_to_list()[0].new_image_like(mask.min(axis=-1))
        # min_mask_frame = mask.min(axis=-1)
        # min_mask_data = np.stack([min_mask_frame for _ in range(mask.shape[-1])], axis=-1)
        # min_mask = img.new_image_like(min_mask_data)
        # extracted = img * mask

        # ants.image_write(mask, str(self.mask_path))
        # print(f"Wrote brain mask for {self.source} to {self.mask_path}")
        # ants.image_write(min_mask, str(self.min_mask_path))
        # print(f"Wrote min brain mask for {self.source} to {self.min_mask_path}")
        # ants.image_write(extracted, str(self.extracted_path))
        return BrainExtracted(self)

    def anat_extract(self, force: bool = False) -> AnatExtracted:
        """Computes a 4D mask for input fMRI.

        Notes
        -----
        Using "robust" option causes a huge amount of phony stderr spam about missing or
        truncated files. Ignore it.
        """
        outfile = Path(
            str(self.t1w_source.resolve()).replace(NII_SUFFIX, EXTRACTED_SUFFIX)
        )
        maskfile = Path(str(self.t1w_source.resolve()).replace(NII_SUFFIX, MASK_SUFFIX))
        if outfile.exists():
            if not force:
                return AnatExtracted(self, mask=maskfile, extracted=outfile)
            if "ses-2" in str(outfile):
                # special case, no need to recompute even when forcing
                return AnatExtracted(self, mask=maskfile, extracted=outfile)
            os.remove(outfile)

        if os.environ.get("CC_CLUSTER") == "niagara":
            DATA = Path(
                "/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri/data/updated"
            )

        cmd = BET()
        cmd.inputs.in_file = str(self.t1w_source.resolve())
        cmd.inputs.out_file = str(outfile)
        cmd.inputs.output_type = "NIFTI_GZ"
        if "Depress" in str(self.t1w_source):
            # this dataset is very strange, needs lower frac to maintain more brain...
            # some subjects still lose a bit in some slices with the lower value below,
            # but with frac = 0.05 only really 2 subjects are very mangled in some slices
            cmd.inputs.frac = 0.05
        elif "Older" in str(self.t1w_source):
            # This dataset has a lot of neck left behind for some reason, so we use the
            # robust option. Experimentation also hows that the `frac` below works best
            cmd.inputs.robust = True
            cmd.inputs.frac = 0.7
        elif "Vigil" in str(self.t1w_source):
            # This also needs the robust treatment. Huge amounts of neck.
            # Also there are only 22 subjects for this dataset
            cmd.inputs.robust = True
            cmd.inputs.frac = 0.3
        else:
            cmd.inputs.frac = 0.3
        cmd.inputs.mask = True

        print(f"Computing mask for {self.t1w_source}")
        results: InterfaceResult = cmd.run()
        bet_maskfile = Path(results.outputs.mask_file).resolve()  # type: ignore
        sleep(1)
        if bet_maskfile.exists():
            shutil.move(bet_maskfile, maskfile)
        else:
            from pprint import pformat

            raise FileNotFoundError(
                f"\n===========================\n"
                f"\tCannot find maskfile: {bet_maskfile}. Details:\n"
                f"\tresults.inputs:\n{pformat(results.inputs, indent=2, depth=5)}\n"
                f"\tresults.outputs:\n{pformat(results.outputs, indent=2, depth=5)}\n"
                f"\toutfile: {outfile}\n"
                f"\tmaskfile: {maskfile}\n"
                f"===========================\n"
            )
        print(f"Wrote brain mask to {maskfile}")
        print(f"Wrote extracted brain to {outfile}")
        return AnatExtracted(self, mask=maskfile, extracted=outfile)

    def reorient_4d_to_RAS(self, img: ANTsImage) -> ANTsImage:
        arrs = [ants.reorient_image2(im).numpy() for im in ants.ndimage_to_list(img)]
        data = np.stack(arrs, axis=-1)
        reoriented = img.new_image_like(data)
        return reoriented

    @property
    def mask_path(self) -> Path:
        parent = self.source.parent
        stem = str(self.source.resolve().name).replace(NII_SUFFIX, "")
        outname = f"{stem}_mask.nii.gz"
        return Path(parent / outname)

    @property
    def min_mask_path(self) -> Path:
        return min_mask_path_from_mask_path(self.mask_path)

    @property
    def extracted_path(self) -> Path:
        parent = self.source.parent
        outname = str(self.source.resolve().name).replace(NII_SUFFIX, EXTRACTED_SUFFIX)
        return Path(parent / outname)

    def parse_source(self) -> Tuple[str, Optional[str], Optional[str]]:
        sid_ = re.search(r"sub-(?:[a-zA-Z]*)?([0-9]+)_.*", self.source.stem)
        ses = re.search(r"ses-([0-9]+)_.*", self.source.stem)
        run_ = re.search(r"run-([0-9]+)_.*", self.source.stem)

        if sid_ is None:
            raise RuntimeError(f"Didn't find an SID in filename {self.source}")
        sid = sid_[1]
        session = ses[1] if ses is not None else None
        run = run_[1] if run_ is not None else None
        return sid, session, run

    def find_t1w_file(self) -> Path:
        anat_dir = self.source.parent.parent / "anat"
        if "ses-2" in str(anat_dir):
            anat_dir = Path(str(anat_dir).replace("ses-2", "ses-1"))
        # if anat_dir.parent.name == "ses-2":
        #     anat_dir = anat_dir.parent.parent / "ses-1/anat"
        anat_img = sorted(anat_dir.glob("*T1w.nii.gz"))
        if len(anat_img) == 0:
            raise RuntimeError(f"Missing T1w file at {anat_dir}")
        if len(anat_img) > 1:
            raise RuntimeError(f"Too many T1w files at {anat_dir}")

        return anat_img[0]

    def find_json_file(self) -> Tuple[Optional[Path], Optional[Path]]:
        # recurse up to BIDS root for remaining json files
        root = self.source.parent
        while "download" not in root.name:
            root = root.parent

        local_json: Optional[Path] = Path(str(self.source).replace(NII_SUFFIX, ".json"))
        if not local_json.exists():
            local_json = None

        # just hardcode the remaining cases where data is not per scan:
        source = self.source.name

        # this data has local files but without slice timing in them
        if "Park" in str(root):
            global_json: Optional[Path] = root / "task-ANT_bold.json"
            return local_json, global_json

        # Rest_w_VigilanceAttention
        if "prefrontal_bold" in source:
            global_json = root / "task-rest_acq-prefrontal_bold.json"
        elif "rest_acq-fullbrain" in source:
            global_json = root / "task-rest_acq-fullbrain_bold.json"
        # Park_v_Control
        elif ("RC" in source) and ("task-ANT" in source):
            global_json = root / "task-ANT_bold.json"
        # Rest_w_Depression_v_Control
        elif "Depression" in root.parent.name:
            global_json = root / "task-rest_bold.json"
        # Rest_w_Healthy_v_OsteoPain
        elif "Osteo" in root.parent.name:
            global_json = root / "task-rest_bold.json"
        elif "Learning" in str(root):
            global_json = None
        elif "Biling" in str(root):
            global_json = None
        elif "Older" in str(root):
            global_json = None
        else:
            raise RuntimeError(f"Can't find .json info from source: {self.source}")
        return local_json, global_json

    def write_out_slicetimes(self) -> Tuple[Optional[List[str]], Optional[Path], float]:
        timings: Optional[List[str]] = None
        TR: Optional[float] = None
        for jsonfile in [self.local_json, self.global_json]:
            if jsonfile is None:
                continue

            with open(jsonfile, "r") as handle:
                info = json_.load(handle)
            found_timings = info.get("SliceTiming", None)
            if (found_timings is not None) and (timings is None):
                timings = found_timings

            found_TR = info.get("RepetitionTime", None)
            if (found_TR is not None) and (TR is None):
                TR = float(found_TR)

        if TR is None:
            raise KeyError(
                f"Could not find RepetitionTime field in any .json for {self.source}"
            )

        if timings is not None:
            outfile: Optional[Path] = Path(
                str(self.source).replace(NII_SUFFIX, ".slicetimes.txt")
            )
            if not outfile.exists():
                with open(outfile, "w") as handle:
                    handle.writelines(list(map(lambda t: f"{t}\n", timings)))
                print(f"Wrote slice timings to {outfile}")
        else:
            outfile = None

        return timings, outfile, TR


class AnatExtracted(Loadable):
    def __init__(self, raw: FmriScan, mask: Path, extracted: Path) -> None:
        super().__init__(extracted)
        self.raw = raw
        self.mask = mask
        self.source: Path = extracted


class BrainExtracted(Loadable):
    def __init__(self, raw: FmriScan) -> None:
        super().__init__(raw.extracted_path)
        self.raw = raw
        self.mask = self.raw.mask_path
        self.min_mask = min_mask_path_from_mask_path(self.mask)
        self.source: Path = raw.extracted_path

    def eigenvalues(self) -> ndarray:
        """Extract masked correlation matrix from self.source and compute eigenvalues"""
        raise NotImplementedError()

    def slicetime_correct(
        self, slice_direction: int = 3, force: bool = False
    ) -> SliceTimeCorrected:
        """
        Notes
        -----
        Only Rest_w_VigilanceAttention data has SliceEncodingDirection = "k"
        (i.e. k+, slices along third spatial dimension, first entry of file
        corresponds to smallest index along thid spatial dim)

        For BIDS slice timing documentation, see:

        https://poc.vl-e.nl/distribution/manual/fsl-3.2/slicetimer/index.html

        reproduced below:

        # INTRODUCTION

            slicetimer is a pre-processing tool designed to correct for sampling offsets inherent in
            slice-wise EPI acquisition sequences.

            Each voxel's timecourse is processed independently and intensities are shifted in time
            so that they reflect the interpolated value of the signal at a common reference
            timepoint for all voxels, providing an instantaneous `snapshot' of the data, rather than
            a staggered sample throughout each volume. Sinc interpolation with a Hanning windowing
            kernel is applied to each timecourse to calculate the interpolated values.

            It is necessary to know in what order the slices were acquired and set the appropriate
            option. The default correction is appropriate if slices were acquired from the bottom of
            the brain.

            If slices were acquired from the top of the brain to the bottom select the --down
            option.

            If the slices were acquired with interleaved order (0, 2, 4 ... 1, 3, 5 ...) then choose
            the --odd option.

            If slices were not acquired in regular order you will need to use a slice order file or
            a slice timings file. If a slice order file is to be used, create a text file with a
            single number on each line, where the first line states which slice was acquired first,
            the second line states which slice was acquired second, etc. The first slice is numbered
            1 not 0.

            If a slice timings file is to be used, put one value (ie for each slice) on each line of
            a text file. The units are in TRs, with 0.5 corresponding to no shift. Therefore a
            sensible range of values will be between 0 and 1.

        #  USAGE AND OPTIONS

            ```
            slicetimer -i  [-o ] [options]

            Compulsory arguments (You MUST set one or more of):

                    -i,--in filename of input timeseries

            Optional arguments (You may optionally specify one or more of):

                    -o,--out        filename of output timeseries
                    -h,--help       display this message
                    -v,--verbose    switch on diagnostic messages
                    --down          reverse slice indexing
                    -r,--repeat     Specify TR of data - default is 3s
                    -d,--direction  direction of slice acquisition (x=1,y=2,z=3) - default is z
                    --odd           use interleaved acquisition
                    --tcustom       filename of single-column custom interleave timing file
                    --ocustom       filename of single-column custom interleave order file (first
                                    slice is referred to as 1 not 0)
            ```
        """

        infile = self.source
        outfile = Path(str(infile).replace(NII_SUFFIX, SLICETIME_SUFFIX))
        if not self.raw.slicetime_file.exists():
            # just copy over extracted brain
            shutil.copy(infile, outfile)
            warn(
                f"No slicetimes for {self.source}. Copying brain-extracted file instead."
            )
            return SliceTimeCorrected(self, outfile)
        if outfile.exists():
            if not force:
                return SliceTimeCorrected(self, outfile)
            os.remove(outfile)

        cmd = SliceTimer()
        cmd.inputs.in_file = str(infile)
        cmd.inputs.custom_timings = str(self.raw.slicetime_file)
        cmd.inputs.time_repetition = self.raw.repetition_time
        cmd.inputs.out_file = str(outfile)
        cmd.inputs.output_type = "NIFTI_GZ"
        cmd.inputs.slice_direction = slice_direction
        print(f"Slice time correcting {infile}...")
        cmd.run()
        return SliceTimeCorrected(self, outfile)


class SliceTimeCorrected(Loadable):
    def __init__(self, extracted: BrainExtracted, corrected: Path) -> None:
        super().__init__(corrected)
        self.raw: FmriScan = extracted.raw
        self.extracted: BrainExtracted = extracted
        self.source: Path = corrected

    def eigenvalues(self) -> ndarray:
        """Extract masked correlation matrix from self.source and compute eigenvalues"""
        raise NotImplementedError()

    def motion_corrected(self, force: bool = False) -> MotionCorrected:
        outfile = Path(str(self.source).replace(NII_SUFFIX, MOTION_CORRECTED_SUFFIX))
        if outfile.exists():
            if not force:
                return MotionCorrected(self, outfile)
            os.remove(outfile)

        img = ants.image_read(str(self.source))
        mask = ants.image_read(str(self.extracted.min_mask))
        mask3d = ants.ndimage_to_list(mask)[0]
        results = motion_correction(
            img,
            fixed=None,  # uses mean image in this case
            type_of_transform="BOLDRigid",
            mask=mask3d,
        )
        corrected = results["motion_corrected"]
        print(f"Performing motion correction for {self.source}")
        ants.image_write(corrected, str(outfile))
        print(f"Saved motion-corrected image to {outfile}")
        return MotionCorrected(self, motion_corrected=outfile)


class MotionCorrected(Loadable):
    def __init__(self, corrected: SliceTimeCorrected, motion_corrected: Path) -> None:
        self.raw = corrected.raw
        self.extracted = corrected.extracted
        self.slicetime_corrected: SliceTimeCorrected = corrected
        self.source: Path = motion_corrected
        super().__init__(self.source)

    @staticmethod
    def reorient_template_to_img(template: ANTsImage, img: ANTsImage) -> ANTsImage:
        """The actual ANTs functions are completely broken for some reason, so we
        do it manually...

        Notes
        -----
                               data  x_n  y_n   z_n x_mm  y_mm  z_mm    t       TR orient
                     Park_v_Control   80   80   43  3.00  3.00  3.00  149     2.40    RPI
                     Park_v_Control   80   80   43  3.00  3.00  3.00  300     2.40    RPI
                     Park_v_Control   96  114   96  2.00  2.00  2.00  149     2.40    LPI
              Rest_v_LearningRecall   64   64   36  3.00  3.00  3.00  195     2.00    RPI
               Rest_w_Bilinguiality  100  100   72  1.80  1.80  1.80  823     0.88    RPI
               Rest_w_Bilinguiality  100   96   72  1.80  1.80  1.80  823     0.88    RPI
               Rest_w_Bilinguiality  100  100   72  1.80  1.80  1.80  823     0.93    RIA
        Rest_w_Depression_v_Control  112  112   25  1.96  1.96  5.00  100     2.50    RPI
         Rest_w_Healthy_v_OsteoPain   64   64   36  3.44  3.44  3.00  244     2.50    RPI
         Rest_w_Healthy_v_OsteoPain   64   64   36  3.44  3.44  3.00  292     2.50    RPI
         Rest_w_Healthy_v_OsteoPain   64   64   36  3.44  3.44  3.00  300     2.50    RPI
             Rest_w_Older_v_Younger   74   74   32  2.97  2.97  4.00  300     2.00    RPI
          Rest_w_VigilanceAttention   64   64   35  3.00  3.00  3.00  300  3000.00    RPI
          Rest_w_VigilanceAttention  128  128   70  1.50  1.50  1.50  300  3000.00    RPI
          Rest_w_VigilanceAttention  200   60   40  0.75  0.75  0.75  150  4000.00    RPI

        >>> df.value_counts()
        data                         x_n  y_n  z_n  x_mm  y_mm  z_mm  t    TR       orient
        Park_v_Control               80   80   43   3.00  3.00  3.00  149  2.40     RPI       552
                                     96   114  96   2.00  2.00  2.00  149  2.40     LPI       552
        Rest_v_LearningRecall        64   64   36   3.00  3.00  3.00  195  2.00     RPI       432
        Rest_w_Bilinguiality         100  100  72   1.80  1.80  1.80  823  0.88     RPI        90
        Rest_w_VigilanceAttention    128  128  70   1.50  1.50  1.50  300  3000.00  RPI        84
        Rest_w_Healthy_v_OsteoPain   64   64   36   3.44  3.44  3.00  300  2.50     RPI        74
        Rest_w_Depression_v_Control  112  112  25   1.96  1.96  5.00  100  2.50     RPI        72
        Rest_w_Older_v_Younger       74   74   32   2.97  2.97  4.00  300  2.00     RPI        62
        Rest_w_VigilanceAttention    200  60   40   0.75  0.75  0.75  150  4000.00  RPI        44
                                     64   64   35   3.00  3.00  3.00  300  3000.00  RPI         4
        Park_v_Control               80   80   43   3.00  3.00  3.00  300  2.40     RPI         1
        Rest_w_Bilinguiality         100  96   72   1.80  1.80  1.80  823  0.88     RPI         1
                                          100  72   1.80  1.80  1.80  823  0.93     RIA         1
        Rest_w_Healthy_v_OsteoPain   64   64   36   3.44  3.44  3.00  244  2.50     RPI         1
                                                                      292  2.50     RPI         1

        """

        temp_orient = template.get_orientation()
        img_orient = img.get_orientation()
        oriented = template.clone()
        if (temp_orient == "LPI") and (img_orient == "RPI"):
            return oriented.reflect_image(axis=1).reflect_image(axis=2)
        else:
            raise ValueError(
                "Impossible! All scans to register should be RPI orientation"
            )

        for i, (axis_temp, axis_img) in enumerate(zip(temp_orient, img_orient)):
            if axis_temp == axis_img:
                continue
            oriented = oriented.reflect_image(axis=i)
        # reoriented = template.new_image_like(oriented)
        reoriented = oriented
        return reoriented

    def t1w_register(self, force: bool = False) -> T1wRegistered:
        """
        Notes
        -----
        fMRI is too low-q to register straight to MNI template. we need to see if
        it is better to go through anat in one way or another. e.g. the classic

            fMRI: reg to -> ant: reg to -> MNI

        or:

            reg fMRI to anat: save reverse transform (anat_to_fmri)
            reg anat to MNI: save reverse transform (mni_to_anat)

        # State of Extraction

        Not extracted:
            Park_v_Control
            Bilingual
            Vigilance
            Osteo
            Learning
            Older_v_Younger

        Extracted:
            Depression

        """
        outfile = Path(str(self.source).replace(NII_SUFFIX, ANAT_REGISTERED_SUFFIX))
        if outfile.exists():
            if not force:
                return T1wRegistered(self, registered=outfile)
            os.remove(outfile)

        img = ants.image_read(str(self.source))
        mask = ants.image_read(str(self.extracted.min_mask))
        anat = ants.image_read(str(self.raw.t1w_source))

        img = ants.mask_image(img, mask)
        imgs: List[ANTsImage] = ants.ndimage_to_list(img)
        avg = imgs[0].new_image_like(img.mean(axis=-1))
        template = ants.image_read(str(TEMPLATE))
        template_mask = ants.image_read(str(TEMPLATE_MASK))
        template = ants.mask_image(template, template_mask)

    def mni_register(self, force: bool = False) -> MNI152Registered:
        outfile = Path(str(self.source).replace(NII_SUFFIX, MNI_REGISTERED_SUFFIX))
        if outfile.exists():
            if not force:
                return MNI152Registered(self, registered=outfile)
            os.remove(outfile)

        img = ants.image_read(str(self.source))
        mask = ants.image_read(str(self.extracted.min_mask))
        img = ants.mask_image(img, mask)
        imgs: List[ANTsImage] = ants.ndimage_to_list(img)
        avg = imgs[0].new_image_like(img.mean(axis=-1))
        template = ants.image_read(str(TEMPLATE))
        template_mask = ants.image_read(str(TEMPLATE_MASK))
        template = ants.mask_image(template, template_mask)
        # template = ants.reorient_image2(template)
        template = self.reorient_template_to_img(template, avg)
        template_mask = self.reorient_template_to_img(template_mask, avg)

        # template = resample_image(
        #     template, resample_params=imgs[0].shape, use_voxels=True, interp_type=4
        # )
        template = resample_image(
            template, resample_params=imgs[0].spacing, use_voxels=False, interp_type=4
        )
        template_mask = resample_image(
            template_mask,
            resample_params=imgs[0].spacing,
            use_voxels=False,
            interp_type=4,
        )

        print(f"Registering {self.source} average image to {TEMPLATE}")
        results = ants.registration(
            fixed=template,
            moving=avg,
            type_of_transform="SyNBold",
        )
        transforms = results["fwdtransforms"]
        print("Applying transform from average image to full 4D data")
        registered = ants.apply_transforms(
            fixed=template,
            moving=img,
            transformlist=transforms,
            imagetype=3,
        )
        ants.plot(registered)

        # registereds = []
        # for img in imgs:
        #     result = ants.apply_transforms(fixed=template, moving=img, transformlist=transform, )
        #     registered = result["warpedmovout"]
        #     registereds.append(registered)
        # registered = results["warpedmovout"]
        ants.image_write(registered, str(outfile))
        print(f"Saved MNI-registered image to {outfile}")
        return MNI152Registered(self, registered=outfile)


class T1wRegistered(Loadable):
    def __init__(self, motion_corrected: MotionCorrected, registered: Path) -> None:
        self.raw = motion_corrected.raw
        self.extracted = motion_corrected.extracted
        self.slicetime_corrected: SliceTimeCorrected = (
            motion_corrected.slicetime_corrected
        )
        self.motion_corrected: MotionCorrected = motion_corrected
        self.source: Path = registered
        super().__init__(self.source)


class MNI152Registered(Loadable):
    def __init__(self, motion_corrected: MotionCorrected, registered: Path) -> None:
        self.raw = motion_corrected.raw
        self.extracted = motion_corrected.extracted
        self.slicetime_corrected: SliceTimeCorrected = (
            motion_corrected.slicetime_corrected
        )
        self.motion_corrected: MotionCorrected = motion_corrected
        self.source: Path = registered
        super().__init__(self.source)


def get_fmri_paths(filt: Optional[str] = None) -> List[Path]:
    UPDATED = DATA / "updated"
    parents = sorted(filter(lambda p: p.is_dir(), UPDATED.glob("*")))
    paths = []
    for parent in parents:
        paths.extend(sorted(parent.rglob("*bold.nii.gz")))
    paths = sorted(filter(lambda p: "derivative" not in str(p), paths))
    paths = sorted(filter(lambda p: "prefrontal" not in str(p), paths))
    if filt is not None:
        paths = sorted(filter(lambda p: filt in str(p), paths))
    return paths


def brain_extract_parallel(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        fmri.brain_extract(force=True)
    except Exception:
        traceback.print_exc()


def anat_extract_parallel(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        fmri.anat_extract(force=True)
    except Exception:
        traceback.print_exc()


def make_slicetime_file(path: Path) -> None:
    try:
        fmri = FmriScan(path)
    except Exception:
        traceback.print_exc()


def inspect_extractions(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        stripped = fmri.brain_extract(force=False)
        orig_file = str(path).replace(NII_SUFFIX, "_plot.png")
        extr_file = str(stripped.source).replace(NII_SUFFIX, "_plot.png")
        if Path(extr_file).exists():
            return
        orig = image_read(str(path))
        extracted = image_read(str(stripped.source))
        orig.ndimage_to_list()[5].plot(filename=orig_file)
        extracted.ndimage_to_list()[5].plot(filename=extr_file)
    except Exception:
        traceback.print_exc()


def reinspect_extractions(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        stripped = fmri.brain_extract(force=False)
        orig_file = str(path).replace(NII_SUFFIX, "_plot.png")
        extr_file = str(stripped.source).replace(NII_SUFFIX, "_plot.png")
        orig = image_read(str(path))
        extracted = image_read(str(stripped.source))
        orig.ndimage_to_list()[5].plot(filename=orig_file)
        extracted.ndimage_to_list()[5].plot(filename=extr_file)
    except Exception:
        traceback.print_exc()


def inspect_anat_extractions(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        stripped = fmri.anat_extract(force=False)
        orig_file = str(fmri.t1w_source).replace(NII_SUFFIX, "_plot.png")
        extr_file = str(stripped.source).replace(NII_SUFFIX, "_plot.png")
        if Path(extr_file).exists():
            return
        orig = image_read(str(fmri.t1w_source))
        extracted = image_read(str(stripped.source))
        orig.plot(filename=orig_file)
        extracted.plot(filename=extr_file)
    except Exception:
        traceback.print_exc()


def reinspect_anat_extractions(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        stripped = fmri.anat_extract(force=False)
        orig_file = str(fmri.t1w_source).replace(NII_SUFFIX, "_plot.png")
        extr_file = str(stripped.source).replace(NII_SUFFIX, "_plot.png")
        orig = image_read(str(fmri.t1w_source))
        extracted = image_read(str(stripped.source))
        orig.plot(filename=orig_file)
        extracted.plot(filename=extr_file)
    except Exception:
        traceback.print_exc()


def slicetime_correct_parallel(path: Path) -> None:
    try:
        fmri = FmriScan(path)
        extracted = fmri.brain_extract(force=False)
        corrected = extracted.slicetime_correct(force=False)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    # on Niagara need module load gcc/8.3.0 openblas/0.3.7 fsl/6.0.4
    paths = get_fmri_paths(filt="Vigil")
    # paths = get_fmri_paths()
    # process_map(make_slicetime_file, paths, chunksize=1)
    # NOTE: for "Vigilance" data, can only have up to 10 workers
    process_map(brain_extract_parallel, paths, chunksize=1, max_workers=10)
    # process_map(inspect_extractions, paths, chunksize=1, max_workers=40)
    # process_map(anat_extract_parallel, paths, chunksize=1, max_workers=40)
    # process_map(inspect_extractions, paths, chunksize=1, max_workers=40)
    # process_map(inspect_anat_extractions, paths, chunksize=1, max_workers=40)
    # process_map(reinspect_anat_extractions, paths, chunksize=1, max_workers=40)
    # process_map(slicetime_correct_parallel, paths, chunksize=1, max_workers=40)
    process_map(reinspect_extractions, paths, chunksize=1, max_workers=40)

    sys.exit()
    for path in paths:
        fmri = FmriScan(path)
        extracted = fmri.brain_extract(force=True)
        print(f"Extracted: {extracted.source}")
        continue

        slice_corrected = extracted.slicetime_correct(force=False)
        print(f"Slicetimed: {slice_corrected.source}")

        motion_corrected = slice_corrected.motion_corrected(force=False)
        print(f"Motion-corr: {motion_corrected.source}")

        t1w_reg = motion_corrected.t1w_register(force=True)
        print(f"T1w-registered: {t1w_reg.source}")

        # registered = motion_corrected.mni_register(force=True)
        # registered = motion_corrected.mni_register(force=False)
        # print(f"Registered: {registered.source}")
