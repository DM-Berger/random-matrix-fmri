# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import (
    AllFeatures,
    EigPlusLevelvar,
    EigPlusRigidity,
    RigidityPlusLevelvar,
)
from rmt.predict import summarize_all_predictions

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9]
    FEATURE_IDXS: list[int | slice | None] = [
        None,
        slice(-10, -1),
        slice(-5, -1),
        slice(-3, -1),
    ]
    fnames = {
        AllFeatures: "all_combined_predictions.json",
        EigPlusRigidity: "eig+rigidity_predictions.json",
        EigPlusLevelvar: "eig+levelvar_predictions.json",
        RigidityPlusLevelvar: "rigidity+levelvar_predictions.json",
    }
    for feature, fname in fnames.items():
        df = summarize_all_predictions(
            feature_cls=feature,
            degrees=DEGREES,
            feature_idxs=FEATURE_IDXS,
        )
        outfile = ROOT.parent / fname
        df.to_json(outfile)
        print(f"Saved predictions to {outfile}")
