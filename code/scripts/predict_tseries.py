# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.tseries_predict import TSERIES_OUTFILES, summarize_all_tseries_predictions

if __name__ == "__main__":
    for kind, fname in TSERIES_OUTFILES.items():
        outfile = ROOT.parent / fname
        if outfile.exists():
            continue
        df = summarize_all_tseries_predictions(
            kind=kind,
            debug=False,
        )
        df.to_json(outfile)
        print(f"Saved tseries predictions to {outfile}")
