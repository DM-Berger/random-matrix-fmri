# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import Rigidities
from rmt.predict import summarize_all_predictions
from rmt.visualize import plot_all_features

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9]
    FEATURE_IDXS: list[int | slice | None] = [
        None,
        slice(-10, -1),
        slice(-5, -1),
        slice(-3, -1),
    ]
    # plot_all_features(
    #     feature_cls=Rigidities,
    #     plot_separations=False,
    #     degrees=DEGREES,
    #     save=False,
    # )
    df = summarize_all_predictions(
        feature_cls=Rigidities,
        degrees=DEGREES,
        feature_idxs=FEATURE_IDXS,
    )
    outfile = ROOT.parent / "rigidity_predictions.json"
    df.to_json(outfile)
    print(f"Saved rigidity predictions to {outfile}")