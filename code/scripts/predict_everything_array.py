# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import os
from pathlib import Path

from rmt.features import FEATURE_OUTFILES
from rmt.predict import FeatureSlice, summarize_all_predictions

ARRAY_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID"))  # type: ignore

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9]
    feature, fname = list(FEATURE_OUTFILES.items())[ARRAY_ID]
    df = summarize_all_predictions(
        feature_cls=feature,
        degrees=DEGREES,
        feature_slices=[*FeatureSlice],
    )
    outfile = ROOT.parent / fname
    df.to_json(outfile)
    print(f"Saved predictions to {outfile}")
