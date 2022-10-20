# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on


from rmt.dataset import ProcessedDataset
from rmt.enumerables import Dataset, TrimMethod

if __name__ == "__main__":
    for source in [*Dataset]:
        for preproc in [True, False]:
            for trim_method in [None, *TrimMethod]:
                data = ProcessedDataset(source=source, full_pre=preproc)
                data.trimmed(trim_method=trim_method)
                print(
                    f"Computed for {source.name}, preproc={preproc}, trim={trim_method} "
                )
