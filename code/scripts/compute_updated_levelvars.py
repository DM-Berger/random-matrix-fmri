# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset, levelvars

if __name__ == "__main__":
    for source in UpdatedDataset:
        for preproc in PreprocLevel:
            for degree in [3, 5, 7, 9]:
                for trim in [None, *TrimMethod]:
                    data = UpdatedProcessedDataset(source=source, preproc_level=preproc)
                    rigs = levelvars(dataset=data, degree=degree, trim_method=trim, parallel=True)
