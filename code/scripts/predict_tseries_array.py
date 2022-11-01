# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import os
from pathlib import Path

from rmt.tseries_predict import TSERIES_OUTFILES, summarize_all_tseries_predictions

ARRAY_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID"))  # type: ignore

if __name__ == "__main__":
    kind, fname = list(TSERIES_OUTFILES.items())[ARRAY_ID]
    df = summarize_all_tseries_predictions(
        kind=kind,
        debug=False,
    )
    outfile = ROOT.parent / fname
    df.to_json(outfile)
    print(f"Saved tseries predictions to {outfile}")
