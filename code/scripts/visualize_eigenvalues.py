# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.features import Eigenvalues
from rmt.predict import summarize_all_predictions
from rmt.visualize import plot_all_features

if __name__ == "__main__":
    FEATURE_IDXS: list[int | slice | None] = [
        None,
        slice(-80, -1),
        slice(-40, -1),
        slice(-20, -1),
        slice(-10, -1),
        slice(-5, -1),
    ]
    # plot_all_features(
    #     feature_cls=Eigenvalues,
    #     plot_separations=False,
    #     save=False,
    # )
    df = summarize_all_predictions(
        feature_cls=Eigenvalues,
        feature_slices=FEATURE_IDXS,
    )
    outfile = ROOT.parent / "eigenvalue_predictions.json"
    df.to_json(outfile)
    print(f"Saved eigenvalue predictions to {outfile}")
