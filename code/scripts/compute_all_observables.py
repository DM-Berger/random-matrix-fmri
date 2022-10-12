# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from rmt.dataset import ProcessedDataset, levelvars, rigidities
from rmt.enumerables import Dataset

if __name__ == "__main__":
    for source in Dataset:
        for preproc in [True, False]:
            for degree in [3, 5, 7, 9]:
                data = ProcessedDataset(source=source, full_pre=preproc)
                rigs = rigidities(dataset=data, degree=degree, parallel=True)
                level_vars = levelvars(dataset=data, degree=degree, parallel=True)
