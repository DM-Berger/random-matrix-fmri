import argparse
import nibabel as nib
import numpy as np
import os

from empyricalRMT.eigenvalues import Eigenvalues
from numba import jit
from numpy import ndarray
from pathlib import Path
from skimage.transform import rescale

DATA_ROOT = Path(__file__).resolve().parent
FMRI_ROOT = DATA_ROOT / "all_fmri"
TARGET_SHAPE = (64, 64, 33, 260)


def make_cheaty_nii(orig: nib.Nifti1Image, array: np.array) -> nib.Nifti1Image:
    """clone the header and extraneous info from `orig` and data in `array`
    into a new Nifti1Image object, for plotting
    """
    affine = orig.affine
    header = orig.header
    # return new_img_like(orig, array, copy_header=True)
    return nib.Nifti1Image(dataobj=array, affine=affine, header=header)


def res(string: str) -> str:
    return str(Path(string).resolve())


@jit(nopython=True, cache=True, fastmath=True)
def remask(img: ndarray, mask: ndarray):
    x, y, z, T = img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                signal = img[i, j, k, :]
                if np.sum(signal) == 0 or np.sum(signal * signal) == 0 or np.std(signal) == 0:
                    mask[i, j, k, 0] = False


parser = argparse.ArgumentParser(description="Handle reshaping")
parser.add_argument("bold", metavar="<bold.nii.gz>", type=res, nargs=1, action="store")
parser.add_argument("outfile", metavar="out.npy", type=res, nargs=1, action="store")

args = parser.parse_args()

img = nib.load(args.bold[0]).get_fdata()

factors = [t / m for m, t in zip(img.shape, TARGET_SHAPE)]  # scaling factors per dimension
factors[-1] = 1.0
rescaled = rescale(img, factors, clip=False, preserve_range=True)

mask = np.ones(rescaled.shape[:-1] + (1,), dtype=bool)
remask(rescaled, mask)

N, t = (np.prod(rescaled.shape[:-1]), rescaled.shape[-1])
rescaled = rescaled.reshape([N, t])
mask = mask.reshape(-1)

brain = rescaled[mask, :]

eigs = Eigenvalues.from_time_series(brain, covariance=False, trim_zeros=False)
vals = eigs.vals

outfile = Path(args.outfile[0])
shape_outfile = outfile.parent / outfile.name.replace("eigs", "shapes")
parent = outfile.resolve().parent.resolve()
os.makedirs(parent, exist_ok=True)
np.save(outfile, vals, allow_pickle=False)
np.save(shape_outfile, np.array(brain.shape, dtype=int), allow_pickle=False)
print(f"Saved eigenvalues to {args.outfile[0]}")
