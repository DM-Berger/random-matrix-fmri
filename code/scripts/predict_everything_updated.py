# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.updated_features import FEATURE_OUTFILES, Levelvars
from rmt.updated_predict import FeatureSlice, summarize_all_updated_predictions

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9]
    for feature, fname in FEATURE_OUTFILES.items():
        df = summarize_all_updated_predictions(
            feature_cls=feature,
            degrees=DEGREES,
            feature_slices=[*FeatureSlice],
            debug=False,
        )
        outfile = ROOT.parent / fname
        df.to_json(outfile)
        print(f"Saved predictions to {outfile}")
