from __future__ import annotations

import json
import json as json_
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import ants
import numpy as np
from ants import ANTsImage, image_read, motion_correction, resample_image
from ants.registration import reorient_image
from nipype.interfaces.fsl import SliceTimer
from numpy import ndarray

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
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
MNI_REGISTERED_SUFFIX = "_mni-registered.nii.gz"


def min_mask_path_from_mask_path(mask: Path) -> Path:
    return Path(str(mask).replace(MASK_SUFFIX, MINMASK_SUFFIX))


class FmriScan:
    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source: Path
        self.sid: str
        self.ses: Optional[str]
        self.run: Optional[str]
        self.json_file: Path
        self.slicetimes: List[float]
        self.repetition_time: float
        self.slicetime_file: Path

        self.source = source
        self.sid, self.ses, self.run = self.parse_source()
        self.json_file = self.find_json_file()
        (
            self.slicetimes,
            self.slicetime_file,
            self.repetition_time,
        ) = self.write_out_slicetimes(self.json_file)

    def brain_extract(self, force: bool = False) -> BrainExtracted:
        """Computes a 4D mask for input fMRI.

        Notes
        -----

        The ANTS computed mask is truly 4D, e.g. the mask at t=1 will not in general be identical to
        the mask at time t=2. We do unforunately need this to get a timeseries at each voxel, and so
        also must define a `min_mask` (or max_mask) for some purposes.

        ants.get_mask seems to be single-core, so it is extremely worth parallelizing brain
        extraction across subjects
        """
        if self.mask_path.exists():
            if not force:
                return BrainExtracted(self)
            os.remove(self.mask_path)

        print(f"Loading {self.source}")
        img: ANTsImage = image_read(str(self.source))
        img = self.reorient_4d_to_RAS(img)
        print(f"Computing mask for {self.source}")
        mask = ants.get_mask(img)
        # min_mask_frame = mask.ndimage_to_list()[0].new_image_like(mask.min(axis=-1))
        min_mask_frame = mask.min(axis=-1)
        min_mask_data = np.stack([min_mask_frame for _ in range(mask.shape[-1])], axis=-1)
        min_mask = img.new_image_like(min_mask_data)
        extracted = img * mask

        ants.image_write(mask, str(self.mask_path))
        print(f"Wrote brain mask for {self.source} to {self.mask_path}")
        ants.image_write(min_mask, str(self.min_mask_path))
        print(f"Wrote min brain mask for {self.source} to {self.min_mask_path}")
        ants.image_write(extracted, str(self.extracted_path))
        print(f"Wrote min brain mask for {self.source} to {self.extracted_path}")
        return BrainExtracted(self)

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

    def find_json_file(self) -> Path:
        local = Path(str(self.source).replace(NII_SUFFIX, ".json"))
        if local.exists():
            return local

        # recurse up to BIDS root for remaining json files
        root = self.source.parent
        while "download" not in root.name:
            root = root.parent

        # just hardcode the remaining cases where data is not per scan:
        source = self.source.name
        # Rest_w_VigilanceAttention
        if "prefrontal_bold" in source:
            return root / "task-rest_acq-prefrontal_bold.json"
        if "rest_acq-fullbrain" in source:
            return root / "task-rest_acq-fullbrain_bold.json"
        # Park_v_Control
        if ("RC" in source) and ("task-ANT" in source):
            return root / "task-ANT_bold.json"
        # Rest_w_Depression_v_Control
        if "Depression" in root.parent.name:
            return root / "task-rest_bold.json"
        # Rest_w_Healthy_v_OsteoPain
        if "Osteo" in root.parent.name:
            return root / "task-rest_bold.json"

        raise RuntimeError(f"Can't find .json info from source: {self.source}")

    def write_out_slicetimes(self, jsonfile: Path) -> Tuple[List[float], Path, float]:
        with open(jsonfile, "r") as handle:
            info = json_.load(handle)
        timings = info.get("SliceTiming", None)
        if timings is None:
            raise KeyError(f"Could not find SliceTiming field in {jsonfile}")
        TR = info.get("RepetitionTime", None)
        if TR is None:
            raise KeyError(f"Could not find RepetitionTime field in {jsonfile}")
        outfile = Path(str(self.source).replace(NII_SUFFIX, SLICETIME_SUFFIX))
        if not outfile.exists():
            with open(outfile, "w") as handle:
                handle.writelines(list(map(lambda t: f"{t}\n", timings)))
            print(f"Wrote slice timings to {outfile}")
        return timings, outfile, TR


class BrainExtracted:
    def __init__(self, raw: FmriScan) -> None:
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
        Only Rest_w_VigilanceAttention data has SliceEncodingDirection = "k" (i.e. k+, slices along
        third spatial dimension, first entry of file corresponds to smallest index along thid spatial
        dim)

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


class SliceTimeCorrected:
    def __init__(self, extracted: BrainExtracted, corrected: Path) -> None:
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


class MotionCorrected:
    def __init__(self, corrected: SliceTimeCorrected, motion_corrected: Path) -> None:
        self.raw = corrected.raw
        self.extracted = corrected.extracted
        self.slicetime_corrected: SliceTimeCorrected = corrected
        self.source: Path = motion_corrected

    @staticmethod
    def reorient_template_to_img(template: ANTsImage, img: ANTsImage) -> ANTsImage:
        """The actual ANTs functions are completely broken for some reason, so we
        do it manually...
        """
        temp_orient = template.get_orientation()
        img_orient = img.get_orientation()
        oriented = template.clone()
        if (temp_orient == "LPI") and (img_orient == "RPI"):
            return oriented.reflect_image(axis=1).reflect_image(axis=2)
        else:
            raise ValueError("Need to test this")

        for i, (axis_temp, axis_img) in enumerate(zip(temp_orient, img_orient)):
            if axis_temp == axis_img:
                continue
            oriented = oriented.reflect_image(axis=i)
        # reoriented = template.new_image_like(oriented)
        reoriented = oriented
        return reoriented

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


class MNI152Registered:
    def __init__(self, motion_corrected: MotionCorrected, registered: Path) -> None:
        self.raw = motion_corrected.raw
        self.extracted = motion_corrected.extracted
        self.slicetime_corrected: SliceTimeCorrected = (
            motion_corrected.slicetime_corrected
        )
        self.motion_corrected: MotionCorrected = motion_corrected
        self.source: Path = registered


if __name__ == "__main__":
    path = (
        DATA
        / "updated/Rest_w_Depression_v_Control/ds002748-download/sub-01/func/sub-01_task-rest_bold.nii.gz"
    )
    # path = (
    #     DATA
    #     / "updated/Rest_w_Older_v_Younger/ds003871-download/sub-1004/func/sub-1004_task-rest_dir-AP_run-01_bold.nii.gz"
    # )

    print(TEMPLATE)
    fmri = FmriScan(path)
    extracted = fmri.brain_extract(force=False)
    print(f"Extracted: {extracted.source}")
    slice_corrected = extracted.slicetime_correct(force=False)
    print(f"Slicetimed: {slice_corrected.source}")
    motion_corrected = slice_corrected.motion_corrected(force=False)
    print(f"Motion-corr: {motion_corrected.source}")
    registered = motion_corrected.mni_register(force=True)
    # registered = motion_corrected.mni_register(force=False)
    print(f"Registered: {registered.source}")
