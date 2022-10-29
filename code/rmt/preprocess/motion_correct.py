# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from tqdm.contrib.concurrent import process_map

from rmt.preprocess.preprocess import get_fmri_paths, motion_correct_parallel

if __name__ == "__main__":
    paths = get_fmri_paths()
    process_map(motion_correct_parallel, paths, max_workers=8)
