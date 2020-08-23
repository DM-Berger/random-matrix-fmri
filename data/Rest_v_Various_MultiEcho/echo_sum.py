import nibabel as nib
import numpy as np

from glob import glob
from pathlib import Path
from pprint import pprint

from typing import Any, Optional


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


niis = [Path(p).resolve() for p in glob("all_fmri/*bold.nii.gz")]
echo1 = sorted(list(filter(something, [p if str(p).find("echo-1") > 0 else None for p in niis])))
echo2 = sorted(list(filter(something, [p if str(p).find("echo-2") > 0 else None for p in niis])))
echo3 = sorted(list(filter(something, [p if str(p).find("echo-3") > 0 else None for p in niis])))

e1: Path
e2: Path
e3: Path
for e1, e2, e3 in zip(echo1, echo2, echo3):
    orig = nib.load(str(e1))
    nii = orig.get_fdata()
    nii = nii + nib.load(str(e2)).get_fdata()
    nii = nii + nib.load(str(e3)).get_fdata()
    outfile = str(e1).replace("echo-1", "summed")
    nib.save(make_cheaty_nii(orig, nii), outfile)
    print(f"Saved {e1.name.replace('run-01', 'run-0X')} summed echo image to {outfile}")
