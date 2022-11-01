from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

import gc
import json as json_
import os
import re
import shutil
import traceback
from abc import ABC
from pathlib import Path
from time import sleep
from typing import Any, List, Optional, Tuple
from warnings import warn

import nibabel as nib
import numpy as np
from nibabel import Nifti2Image
from numpy import ndarray
from tqdm.contrib.concurrent import process_map

from rmt.constants import DATA_ROOT as DATA
from rmt.constants import EIGS_SUFFIX, NII_SUFFIX, UPDATED
from rmt.enumerables import SeriesKind

if os.environ.get("CC_CLUSTER") == "niagara":
    ROOT = Path("/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri").resolve()
    DATA = Path("/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri/data").resolve()
    UPDATED = Path(
        "/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri/data/updated"
    ).resolve()
    RMT_DIR = Path(
        "/gpfs/fs0/scratch/j/jlevman/dberger/random-matrix-fmri/data/updated/rmt"
    ).resolve()
    os.environ["MPLCONFIGDIR"] = str(
        Path("/gpfs/fs0/scratch/j/jlevman/dberger/.mplconfig")
    )


class Loadable(ABC):
    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source: Path = source

    def load(self) -> Nifti2Image:
        return nib.load(str(self.source))


class RMTComputatable(Loadable):
    def __init__(self, source: Path) -> None:
        super().__init__(source)

    def compute_timeseries(self, force: bool = False) -> None:
        for kind in SeriesKind:
            long_outfile = (
                str(self.source)
                .replace(NII_SUFFIX, kind.suffix())
                .replace("data/updated", "data/updated/rmt")
                .replace("func/", "")
            )
            outfile = Path(re.sub(r"ds.*download/", "", long_outfile))
            if outfile.exists():
                if not force:
                    return np.load(outfile)  # type: ignore
                os.remove(outfile)
            img = self.load()
            arr = img.get_fdata()
            arr2d = arr.reshape(-1, arr.shape[-1])
            mask = np.var(arr2d, axis=1) > 0
            arr2d = arr2d[mask]

            reduced: ndarray = kind.reduce(arr2d).ravel()
            if len(reduced) != arr2d.shape[1]:
                raise RuntimeError(
                    f"Shape mismatch for series type: {kind}. "
                    f"Expected reduced series to have length {arr2d.shape[1]}, "
                    f"but got length {len(reduced)} instead."
                )

            outdir = outfile.parent
            if not outdir.exists():
                outdir.mkdir(exist_ok=True, parents=True)
            np.save(outfile, reduced, allow_pickle=False)
            print(f"Saved {kind.name} time series from {self.source} to {outfile}")

    def compute_eigenvalues(self, force: bool = False) -> ndarray:
        from empyricalRMT.eigenvalues import Eigenvalues

        long_outfile = (
            str(self.source)
            .replace(NII_SUFFIX, EIGS_SUFFIX)
            .replace("data/updated", "data/updated/rmt")
            .replace("func/", "")
        )
        outfile = Path(re.sub(r"ds.*download/", "", long_outfile))
        if outfile.exists():
            if not force:
                return np.load(outfile)  # type: ignore
            os.remove(outfile)

        img = self.load()
        arr = img.get_fdata()
        arr2d = arr.reshape(-1, arr.shape[-1])
        mask = np.var(arr2d, axis=1) > 0
        arr2d = arr2d[mask]
        eigs = Eigenvalues.from_time_series(
            arr2d, covariance=False, trim_zeros=False, time_axis=1
        )
        values = eigs.values
        outdir = outfile.parent
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        np.save(outfile, values, allow_pickle=False)
        print(f"Saved computed eigenvalues from {self.source} to {outfile}")
        return values

    def load(self) -> Nifti2Image:
        return nib.load(str(self.source))


def get_paths() -> List[Path]:
    globs = [
        "*bold_extracted.nii.gz",
        "*slicetime-corrected.nii.gz",
        "*motion-corrected.nii.gz",
        "*mni-reg.nii.gz",
    ]
    paths: List[Path] = []
    for glob in globs:
        found = sorted(UPDATED.rglob(glob))
        paths.extend(found)
    paths = sorted(paths)
    return paths


def compute_eigs(path: Path) -> None:
    try:
        rmt = RMTComputatable(path)
        rmt.compute_eigenvalues()
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")


if __name__ == "__main__":
    paths = get_paths()
    process_map(
        compute_eigs, paths, desc="Computing eigenvalues", chunksize=1, max_workers=12
    )
