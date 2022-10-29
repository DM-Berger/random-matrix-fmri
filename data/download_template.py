import shutil
from pathlib import Path
from typing import List

from templateflow import api as tflow

DATA = Path(__file__).resolve().parent
TEMPLATE = "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
MASKED = "tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"
TEMPLATE_OUTFILE = DATA / TEMPLATE
MASK_OUTFILE = DATA / MASKED

if __name__ == "__main__":
    outfiles: List[Path] = tflow.get(
        "MNI152NLin2009cAsym", resolution=2, suffix="T1w", raise_empty=True
    )
    for outfile in outfiles:
        out = DATA / Path(outfile).name
        shutil.copy(Path(outfile), out)
        print(f"Copied {outfile} to {out}")

