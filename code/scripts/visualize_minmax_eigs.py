# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import EigsMinMax5, EigsMinMax10, EigsMinMax20
from rmt.predict import summarize_all_predictions

if __name__ == "__main__":
    FEATURE_IDXS: list[int | slice | None] = [None]
    fnames = {
        EigsMinMax5: "eigs-minmax-5_predictions.json",
        EigsMinMax10: "eigs-minmax-10_predictions.json",
        EigsMinMax20: "eigs-minmax-20_predictions.json",
    }
    for feature, fname in fnames.items():
        df = summarize_all_predictions(
            feature_cls=feature,
            feature_idxs=FEATURE_IDXS,
        )
        outfile = ROOT.parent / fname
        df.to_json(outfile)
        print(f"Saved minmax eigenvalue predictions to {outfile}")
