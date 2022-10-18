# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import EigsMiddle10, EigsMiddle20, EigsMiddle40
from rmt.predict import summarize_all_predictions

if __name__ == "__main__":
    FEATURE_IDXS: list[int | slice | None] = [None]
    fnames = {
        EigsMiddle10: "eigs-middle-10_predictions.json",
        EigsMiddle20: "eigs-middle-20_predictions.json",
        EigsMiddle40: "eigs-middle-40_predictions.json",
    }
    for feature, fname in fnames.items():
        df = summarize_all_predictions(
            feature_cls=feature,
            feature_slices=FEATURE_IDXS,
        )
        outfile = ROOT.parent / fname
        df.to_json(outfile)
        print(f"Saved middle eigenvalue predictions to {outfile}")
