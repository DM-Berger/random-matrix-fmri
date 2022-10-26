import json
import re
import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ants import ANTsImage, image_read, motion_correction
from nipype.interfaces.fsl import SliceTimer
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal


def min_mask_from_mask(mask: Path) -> Path:
    return Path(str(mask).replace("_mask.nii.gz", "_mask_min.nii.gz"))

def direction_parse(string: str) -> int:
    if string in ["i", "x", "1"]:
        return 1
    elif string in ["j", "y", "2"]:
        return 2
    elif string in ["k", "z", "3"]:
        return 3
    else:
        raise ValueError(
            "Argument to --direction must be one of ['i', 'j', 'k', '1', '2', '3', 'x', 'y', 'z']."
        )

def slicetime_correct(infile: Path, timings: Path, TR: float, slice_direction: str = "z") -> Path:
    """
    Only Rest_w_VigilanceAttention data has SliceEncodingDirection = "k" (i.e. k+, slices along
    third spatial dimension, first entry of file corresponds to smallest index along thid spatial
    dim)
    """
    MCFLIRT_SUFFIX = "mcflirted.nii.gz"
    SLICETIME_SUFFIX = "stationary.nii.gz"
    outfile = Path(str(infile).replace(MCFLIRT_SUFFIX, SLICETIME_SUFFIX))
    cmd = SliceTimer()
    cmd.inputs.in_file = str(infile)
    cmd.inputs.custom_timings = str(timings)
    cmd.inputs.time_repetition = TR
    cmd.inputs.out_file = str(infile).replace(MCFLIRT_SUFFIX, SLICETIME_SUFFIX)
    cmd.inputs.output_type = "NIFTI_GZ"
    cmd.inputs.slice_direction = slice_direction
    if outfile.exists():
        return outfile
    cmd.run()
    return outfile  # different interface than above

class PreprocessedScan(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.source: Path
        self.sid: str

    @abstractmethod
    def eigenvalues(self) -> ndarray:
        """Extract masked correlation matrix from self.source and compute eigenvalues"""
        pass

    def parse_source(self) -> Tuple[str, Optional[str]]:
        sid_ = re.search(r"sub-(?:[a-zA-Z]*)?([0-9]+)_.*", self.source.stem)
        if sid_ is None:
            raise RuntimeError(f"Didn't find an SID in filename {self.source}")
        sid = sid_[1]
        ses = re.search(r"ses-([0-9]+)_.*", self.source.stem)
        session = ses[1] if ses is not None else None
        return sid, session


class SliceTimeAligned(PreprocessedScan):
    CMD = (
        "slicetimer -i {masked} "
        "-o {time_corrected} "
        "--repeat={TR} "
        "--direction={direction} "
        "--tcustom={slicetime_json}"
    )

    def __init__(self, source: Path, per_subject_slicetimes: bool) -> None:
        self.source = source
        self.per_subject: bool = per_subject_slicetimes

    def slicetime_file(self) -> Path:
        """
        Notes
        -----
        https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#timing-parameters

        # SliceTiming

            The time at which each slice was acquired within each volume (frame) of the acquisition.
            Slice timing is not slice order -- rather, it is a list of times containing the time (in
            seconds) of each slice acquisition in relation to the beginning of volume acquisition.
            The list goes through the slices along the slice axis in the slice encoding dimension
            (see below). Note that to ensure the proper interpretation of the "SliceTiming" field,
            it is important to check if the OPTIONAL SliceEncodingDirection exists. In particular,
            if "SliceEncodingDirection" is negative, the entries in "SliceTiming" are defined in
            reverse order with respect to the slice axis, such that the final entry in the
            "SliceTiming" list is the time of acquisition of slice 0. Without this parameter slice
            time correction will not be possible.

        # SliceEncodingDirection

            The axis of the NIfTI data along which slices were acquired, and the direction in which
            "SliceTiming" is defined with respect to. i, j, k identifiers correspond to the first,
            second and third axis of the data in the NIfTI file. A - sign indicates that the
            contents of "SliceTiming" are defined in reverse order - that is, the first entry
            corresponds to the slice with the largest index, and the final entry corresponds to
            slice index zero. When present, the axis defined by "SliceEncodingDirection" needs to be
            consistent with the slice_dim field in the NIfTI header. When absent, the entries in
            "SliceTiming" must be in the order of increasing slice index as defined by the NIfTI
            header.

            Must be one of: "i", "j", "k", "i-", "j-", "k-".
        """
        if self.per_subject:
            info_file = str(self.source).replace(".nii.gz", ".json")
            info = json.load(info_file)
            slicetimes: List[float] = info["SliceTiming"]
            # now write to temp or perm file for FSL to use, and then run `slicetimer`
            outfile = ...

    @staticmethod
    def cmd(
        input: Path,
        outfile: Path,
        slicetime_file: Path,
        TR: float,
        direction: Literal["x", "y", "z", 1, 2, 3, "i", "j", "k"],
        reverse: bool,
        interleaved: bool,
    ) -> str:
        command = SliceTimeAligned.CMD.format(
            masked=input,
            time_corrected=str(outfile),
            TR=str(TR),
            direction=str(direction),
            slicetime_file=str(slicetime_file),
        )
        if reverse:
            command += " --down"
        if interleaved:
            command += " --odd"
        return command


class BrainExtracted(PreprocessedScan):
    def __init__(self, mask: Path) -> None:
        self.mask = mask
        self.min_mask = min_mask_from_mask(self.mask)

    def slice_time(self, per_subject_slicetimes: bool) -> None:
        """Ultimately, uses FSL to execute:

            slicetimer -i self

        Notes
        -----
        For documentation, see:

        https://poc.vl-e.nl/distribution/manual/fsl-3.2/slicetimer/index.html

        reproduced below:

        # INTRODUCTION

        slicetimer is a pre-processing tool designed to correct for sampling offsets inherent in
        slice-wise EPI acquisition sequences.

        Each voxel's timecourse is processed independently and intensities are shifted in time so
        that they reflect the interpolated value of the signal at a common reference timepoint for
        all voxels, providing an instantaneous `snapshot' of the data, rather than a staggered
        sample throughout each volume. Sinc interpolation with a Hanning windowing kernel is applied
        to each timecourse to calculate the interpolated values.

        It is necessary to know in what order the slices were acquired and set the appropriate
        option. The default correction is appropriate if slices were acquired from the bottom of the
        brain.

        If slices were acquired from the top of the brain to the bottom select the --down option.

        If the slices were acquired with interleaved order (0, 2, 4 ... 1, 3, 5 ...) then choose the
        --odd option.

        If slices were not acquired in regular order you will need to use a slice order file or a
        slice timings file. If a slice order file is to be used, create a text file with a single
        number on each line, where the first line states which slice was acquired first, the second
        line states which slice was acquired second, etc. The first slice is numbered 1 not 0.

        If a slice timings file is to be used, put one value (ie for each slice) on each line of a
        text file. The units are in TRs, with 0.5 corresponding to no shift. Therefore a sensible
        range of values will be between 0 and 1.

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
        pass

    def slice_time_and_align(self) -> SliceTimeAligned:
        pass


class FullRegistered:
    def __init__(self) -> None:
        pass


class RawfMRI:
    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = source

    def brain_extract(self) -> None:
        """Computes a 4D mask for input fMRI.


        Notes
        -----

        The ANTS computed mask is truly 4D, e.g. the mask at t=1 will not in general be identical to
        the mask at time t=2. We do unforunately need this to get a timeseries at each voxel, and so
        also must define a `min_mask` (or max_mask) for some purposes.

        ants.get_mask seems to be single-core, so it is extremely worth parallelizing brain
        extraction across subjects
        """
        img: ANTsImage = image_read(self.source)
        mask: ANTsImage = ants.get_mask(img)
        min_mask_frame = mask.ndimage_to_list()[0].new_image_like(mask.min(axis=-1))
        min_masked = ants.list_to_ndimage([min_mask_frame for _ in range(img.shape[-1])])
        # masked = img * mask
        ants.image_write(mask, str(self.mask_path))
        print(f"Wrote brain mask for {self.source} to {self.mask_path}")
        ants.image_write(mask, str(self.min_mask_path))
        print(f"Wrote min brain mask for {self.source} to {self.min_mask_path}")

    @property
    def mask_path(self) -> Path:
        parent = self.source.parent
        stem = str(self.source.resolve().name).replace(".nii.gz", "")
        outname = f"{stem}_mask.nii.gz"
        return Path(parent / outname)

    @property
    def min_mask_path(self) -> Path:
        return min_mask_from_mask(self.mask_path)

    @property
    def extracted_path(self) -> Path:
        parent = self.source.parent
        stem = str(self.source.resolve().name).replace(".nii.gz", "")
        outname = f"{stem}_extracted.nii.gz"
        return Path(parent / outname)