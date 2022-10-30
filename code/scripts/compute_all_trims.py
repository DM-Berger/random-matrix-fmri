# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on


from rmt.updated_dataset import UpdatedProcessedDataset
from rmt.enumerables import UpdatedDataset, TrimMethod, PreprocLevel

if __name__ == "__main__":
    for source in [*UpdatedDataset]:
        for preproc in [*PreprocLevel]:
            for trim_method in [None, *TrimMethod]:
                data = UpdatedProcessedDataset(source=source, preproc_level=preproc)
                data.trimmed(trim_method=trim_method)
                print(
                    f"Computed for {source.name}, preproc={preproc}, trim={trim_method} "
                )
