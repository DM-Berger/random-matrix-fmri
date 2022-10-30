# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from rmt.enumerables import PreprocLevel, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset, rigidities

if __name__ == "__main__":
    for source in UpdatedDataset:
        for preproc in PreprocLevel:
            for degree in [3, 5, 7, 9]:
                data = UpdatedProcessedDataset(source=source, preproc=preproc)
                rigs = rigidities(dataset=data, degree=degree, parallel=True)
