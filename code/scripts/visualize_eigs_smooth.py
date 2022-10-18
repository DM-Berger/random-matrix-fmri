# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import (
    EigenvaluesPlusEigenvaluesSmoothed,
    EigenvaluesPlusSavGol,
    EigenvaluesSavGol,
    EigenvaluesSmoothed,
)
from rmt.predict import summarize_all_predictions

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9, 11]
    FEATURE_IDXS: list[int | slice | None] = [
        None,
        slice(-10, -1),
        slice(-5, -1),
        slice(-3, -1),
    ]
    fnames = {
        # EigenvaluesSmoothed: "eig_smoothed_predictions.json",
        EigenvaluesPlusSavGol: "eigenvalues+eig_savgol_predictions.json",
        EigenvaluesSavGol: "eig_savgol_predictions.json",
        # EigenvaluesPlusEigenvaluesSmoothed: "eigenvalues+eig_smoothed_predictions.json",
    }
    for feature, fname in fnames.items():
        df = summarize_all_predictions(
            feature_cls=feature,
            degrees=DEGREES,
            feature_slices=FEATURE_IDXS,
        )
        outfile = ROOT.parent / fname
        df.to_json(outfile)
        print(f"Saved smoothed eigenvalue predictions to {outfile}")
