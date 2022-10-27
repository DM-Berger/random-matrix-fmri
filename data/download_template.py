import shutil
from pathlib import Path

from templateflow import api as tflow

DATA = Path(__file__).resolve().parent
ATLAS = "tpl-MNI152NLin2009aAsym_res-1_T1w.nii.gz"
MASK = "tpl-MNI152NLin2009aAsym_res-1_desc-brain_mask.nii.gz"
ATLAS_OUTFILE = DATA / ATLAS
MASK_OUTFILE = DATA / MASK

if __name__ == "__main__":
    outfile = tflow.get(
        "MNI152NLin2009aAsym", resolution=1, suffix="T1w", raise_empty=True
    )
    outdir = Path(outfile).parent
    atlas = outdir / ATLAS
    mask = outdir / MASK
    shutil.copy(atlas, ATLAS_OUTFILE)
    print(f"Copied downloaded atlas to {ATLAS_OUTFILE}")
    shutil.copy(mask, MASK_OUTFILE)
    print(f"Copied downloaded mask to {MASK_OUTFILE}")
