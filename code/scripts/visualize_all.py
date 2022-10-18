# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

from rmt.dataset import Dataset
from rmt.visualize import plot_all_features_multi

if __name__ == "__main__":
    plot_all_features_multi(
        # sources=[*Dataset],
        sources=[
            Dataset.TaskAttention,
            Dataset.Parkinsons,
            Dataset.Osteo,
        ],
        save=True,
    )
