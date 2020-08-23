import gc
import multiprocessing as mp
import nibabel as nib
import numpy as np

from glob import glob
from multiprocessing import Pool
from numba import jit
from numpy import ndarray
from pathlib import Path
from tqdm import tqdm

from typing import Any, Optional

NIIS_GLOB = str(Path(__file__).resolve().parent / "all_fmri" / "*fullpre_stripped.nii.gz")
MASKS_GLOB = str(Path(__file__).resolve().parent / "all_fmri" / "*fullpre_stripped_mask.nii.gz")


def something(v: Optional[Any]) -> bool:
    return v is not None


def make_cheaty_nii(orig: nib.Nifti1Image, array: np.array) -> nib.Nifti1Image:
    """clone the header and extraneous info from `orig` and data in `array`
    into a new Nifti1Image object, for plotting
    """
    affine = orig.affine
    header = orig.header
    # return new_img_like(orig, array, copy_header=True)
    return nib.Nifti1Image(dataobj=array, affine=affine, header=header)


@jit(nopython=True, cache=True)
def interweave(m1: ndarray, m2: ndarray, m3: ndarray) -> ndarray:
    x, y, z, T = m1.shape
    m = np.empty((x, y, z, 3 * T), dtype=np.float64)
    for t in range(T):
        m[:, :, :, 3 * t] = m1[:, :, :, t]
        m[:, :, :, 3 * t + 1] = m2[:, :, :, t]
        m[:, :, :, 3 * t + 2] = m3[:, :, :, t]
    return m


def interleave_echoes(e1, e2, e3):
    orig = nib.load(str(e1))
    mat1 = nib.load(str(e1)).get_fdata()
    mat2 = nib.load(str(e2)).get_fdata()
    mat3 = nib.load(str(e3)).get_fdata()
    interweaved = interweave(mat1, mat2, mat3)
    outfile = str(e1).replace("echo-1", "interweaved")
    nib.save(make_cheaty_nii(orig, interweaved), outfile)


def join_masks(m1, m2, m3):
    orig = nib.load(str(m1))
    only_brain = np.copy(np.array(orig.get_fdata(), dtype=bool))
    only_brain &= np.array(nib.load(str(m2)).get_fdata(), dtype=bool)
    only_brain &= np.array(nib.load(str(m3)).get_fdata(), dtype=bool)
    outfile = str(m1).replace("echo-1", "interweaved")
    nib.save(make_cheaty_nii(orig, only_brain), outfile)


niis = [Path(p).resolve() for p in glob(NIIS_GLOB)]
echo1 = sorted(list(filter(something, [p if str(p).find("echo-1") > 0 else None for p in niis])))
echo2 = sorted(list(filter(something, [p if str(p).find("echo-2") > 0 else None for p in niis])))
echo3 = sorted(list(filter(something, [p if str(p).find("echo-3") > 0 else None for p in niis])))

masks = [Path(p).resolve() for p in glob(MASKS_GLOB)]
masks1 = sorted(list(filter(something, [p if str(p).find("echo-1") > 0 else None for p in masks])))
masks2 = sorted(list(filter(something, [p if str(p).find("echo-2") > 0 else None for p in masks])))
masks3 = sorted(list(filter(something, [p if str(p).find("echo-3") > 0 else None for p in masks])))

with Pool(mp.cpu_count()) as pool:
    pool.starmap_async(interleave_echoes, iterable=list(zip(echo1, echo2, echo3)))
    pool.close()
    pool.join()
    print("Done interleaving")

with Pool(mp.cpu_count()) as pool:
    pool.starmap_async(join_masks, iterable=list(zip(masks1, masks2, masks3)))
    pool.close()
    pool.join()
    print("Done joining interleaved masks.")
