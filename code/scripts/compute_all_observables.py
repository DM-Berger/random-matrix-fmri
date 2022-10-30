# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import os
from argparse import Namespace

from empyricalRMT.smoother import SmoothMethod
from sklearn.model_selection import ParameterGrid

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset, levelvars, rigidities

TASK = int(os.environ.get("SLURM_ARRAY_TASK_ID"))  # type: ignore

if __name__ == "__main__":
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            {
                "source": [*UpdatedDataset],
                "preproc": [*PreprocLevel],
            }
        )
    ]  # length is 28
    source = grid[TASK].source
    preproc = grid[TASK].preproc
    for degree in [3, 5, 7, 9]:
        for trim_method in [None, *TrimMethod]:
            data = UpdatedProcessedDataset(source=source, preproc_level=preproc)
            rigs = rigidities(
                dataset=data,
                degree=degree,
                trim_method=trim_method,
                parallel=True,
            )
            level_vars = levelvars(
                dataset=data,
                degree=degree,
                trim_method=trim_method,
                parallel=True,
            )
