import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from rmt.constants import DATASETS, DATASETS_FULLPRE
from rmt.enumerables import Dataset


class ProcessedDataset:
    def __init__(self, source: Dataset, full_pre: bool) -> None:
        self.source = source
        self.full_pre = full_pre
        self.eig_paths: List[Path]

    def load_eigs(self) -> List[ndarray]:
        raise NotImplementedError()
