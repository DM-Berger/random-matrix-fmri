import nibabel as nib
import numpy as np

from glob import glob
from numba import jit, prange
from numpy import ndarray
from pathlib import Path
from tqdm import tqdm

from typing import Any, Optional

GLOB = str(Path(__file__).resolve().parent / "all_fmri" / "*bold.nii.gz")


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


@jit(nopython=True, cache=True, parallel=True)
def interweave(m1: ndarray, m2: ndarray, m3: ndarray) -> ndarray:
    x, y, z, T = m1.shape
    m = np.empty((x, y, z, 3 * T), dtype=np.float64)
    for t in prange(T):
        m[:, :, :, 3 * t] = m1[:, :, :, t]
        m[:, :, :, 3 * t + 1] = m2[:, :, :, t]
        m[:, :, :, 3 * t + 2] = m3[:, :, :, t]
    return m


niis = [Path(p).resolve() for p in glob(GLOB)]
echo1 = sorted(list(filter(something, [p if str(p).find("echo-1") > 0 else None for p in niis])))
echo2 = sorted(list(filter(something, [p if str(p).find("echo-2") > 0 else None for p in niis])))
echo3 = sorted(list(filter(something, [p if str(p).find("echo-3") > 0 else None for p in niis])))

e1: Path
e2: Path
e3: Path
for e1, e2, e3 in tqdm(zip(echo1, echo2, echo3), desc="Interweaving", total=len(echo1)):
    orig = nib.load(str(e1))
    mat1 = nib.load(str(e1)).get_fdata()
    mat2 = nib.load(str(e2)).get_fdata()
    mat3 = nib.load(str(e3)).get_fdata()
    interweaved = interweave(mat1, mat2, mat3)
    outfile = str(e1).replace("echo-1", "interweaved")
    nib.save(make_cheaty_nii(orig, interweaved), outfile)
    print(f"Saved {e1.name.replace('run-01', 'run-0X')} interweaved echo image to {outfile}")
