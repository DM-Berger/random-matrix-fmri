import gc
import multiprocessing as mp
import nibabel as nib
import numpy as np
import sys

from glob import glob
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import Any, Optional


def is_some(v: Optional[Any]) -> bool:
    return v is not None


def make_cheaty_nii(orig: nib.Nifti1Image, array: np.array) -> nib.Nifti1Image:
    """clone the header and extraneous info from `orig` and data in `array`
    into a new Nifti1Image object, for plotting
    """
    affine = orig.affine
    header = orig.header
    # return new_img_like(orig, array, copy_header=True)
    return nib.Nifti1Image(dataobj=array, affine=affine, header=header)


niis = [Path(p).resolve() for p in glob("all_fmri/*fullpre_stripped.nii.gz")]
echo1 = sorted(list(filter(is_some, [p if str(p).find("echo-1") > 0 else None for p in niis])))
echo2 = sorted(list(filter(is_some, [p if str(p).find("echo-2") > 0 else None for p in niis])))
echo3 = sorted(list(filter(is_some, [p if str(p).find("echo-3") > 0 else None for p in niis])))

masks = [Path(p).resolve() for p in glob("all_fmri/*fullpre_stripped_mask.nii.gz")]
mask1 = sorted(list(filter(is_some, [p if str(p).find("echo-1") > 0 else None for p in masks])))
mask2 = sorted(list(filter(is_some, [p if str(p).find("echo-2") > 0 else None for p in masks])))
mask3 = sorted(list(filter(is_some, [p if str(p).find("echo-3") > 0 else None for p in masks])))


def sum_images(e1: Path, e2: Path, e3: Path):
    orig = nib.load(str(e1))
    nii = orig.get_fdata()
    nii = nii + nib.load(str(e2)).get_fdata()
    nii = nii + nib.load(str(e3)).get_fdata()
    outfile = str(e1).replace("echo-1", "summed")
    nib.save(make_cheaty_nii(orig, nii), outfile)


def join_masks(m1: Path, m2: Path, m3: Path):
    orig = nib.load(str(m1))
    only_brain = np.copy(np.array(orig.get_fdata(), dtype=bool))
    only_brain &= np.array(nib.load(str(m2)).get_fdata(), dtype=bool)
    only_brain &= np.array(nib.load(str(m3)).get_fdata(), dtype=bool)
    outfile = str(m1).replace("echo-1", "summed")
    nib.save(make_cheaty_nii(orig, only_brain), outfile)


with Pool(mp.cpu_count()) as pool:
    pool.starmap_async(sum_images, iterable=list(zip(echo1, echo2, echo3)))
    pool.close()
    pool.join()

print("Done sums")

with Pool(mp.cpu_count()) as pool:
    pool.starmap_async(join_masks, iterable=list(zip(mask1, mask2, mask3)))
    pool.close()
    pool.join()

print("Done masks")
