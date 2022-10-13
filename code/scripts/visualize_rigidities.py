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
    # L_IDXS: List[int | None] = [None]
    L_IDXS: list[int | slice | None] = [-2]
    plot_all_features(
        feature_cls=Rigidities,
        plot_separations=False,
        degrees=DEGREES,
        save=False,
    )
    summarize_all_predictions(
        feature_cls=Rigidities,
        degrees=DEGREES,
        feature_idxs=L_IDXS,
    )
