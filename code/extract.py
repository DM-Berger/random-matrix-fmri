import argparse
import nibabel as nib
import numpy as np
import os
import sys

from empyricalRMT.eigenvalues import Eigenvalues
from pathlib import Path


def mkdirp(path: Path) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(
            f"Error making directory {path}. Another program may have modified the file "
            "while this script was running.",
            file=sys.stderr,
        )
        print("Original error:", file=sys.stderr)
        raise e


def res(string: str) -> str:
    return str(Path(string).resolve())


parser = argparse.ArgumentParser(description="Handle eigenvalue extraction")
parser.add_argument("bold", metavar="<bold.nii.gz>", type=res, nargs=1, action="store")
parser.add_argument("mask", metavar="<mask.nii.gz>", type=res, nargs=1, action="store")
parser.add_argument("outfile", metavar="out.npy", type=res, nargs=1, action="store")

args = parser.parse_args()

img = nib.load(args.bold[0]).get_fdata()
mask = np.array(nib.load(args.mask[0]).get_fdata(), dtype=bool)

N, t = (np.prod(img.shape[:-1]), img.shape[-1])
img = img.reshape([N, t])
mask = mask.reshape(-1)

# remove dead or constant voxels
for i, signal in enumerate(img):
    if np.sum(signal) == 0 or np.sum(signal * signal) == 0 or np.std(signal, ddof=1) == 0:
        mask[i] = False

brain = img[mask, :]

eigs = Eigenvalues.from_time_series(brain, covariance=False, trim_zeros=False)
vals = eigs.vals

outfile = Path(args.outfile[0])
shape_outfile = outfile.parent / outfile.name.replace("eigs", "shapes")
parent = outfile.resolve().parent.resolve()
mkdirp(parent)
np.save(outfile, vals, allow_pickle=False)
np.save(shape_outfile, np.array(brain.shape, dtype=int), allow_pickle=False)
print(f"Saved eigenvalues to {args.outfile[0]}")
