from abc import ABC
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
from ants import ANTsImage, image_read
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal


def min_mask_from_mask(mask: Path) -> Path:
    return Path(str(mask).replace("_mask.nii.gz", "_mask_min.nii.gz"))


class BrainExtracted:
    def __init__(self, mask: Path) -> None:
        self.mask = mask
        self.min_mask = min_mask_from_mask(self.mask)


class SliceTimeAligned(PreprocessedfMRI):
    def __init__(self) -> None:
        pass


class FullRegistered(PreprocessedfMRI):
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