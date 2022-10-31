# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import os
from argparse import Namespace

from sklearn.model_selection import ParameterGrid

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset, levelvars, rigidities

TASK = int(os.environ.get("SLURM_ARRAY_TASK_ID"))  # type: ignore

if __name__ == "__main__":
    trims = [None, *TrimMethod]
    degrees = [3, 5, 7, 9]
    observables = [levelvars, rigidities]
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            {
                "trim": trims,
                "degree": degrees,
                "obs": observables,
            }
        )
    ]  # length is 32
    print(f"Job {TASK} of {len(grid)}...")
    trim = grid[TASK].trim
    degree = grid[TASK].degree
    obs = grid[TASK].obs

    for source in UpdatedDataset:
        for preproc in PreprocLevel:
            data = UpdatedProcessedDataset(source=source, preproc_level=preproc)
            obs(
                dataset=data, degree=degree, trim_method=trim, parallel=True
            )
