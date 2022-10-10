# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from rmt._types import Subgroups
from rmt.constants import DATA_ROOT, DATASETS, DATASETS_FULLPRE
from rmt.enumerables import Dataset


def to_path_frame(info: dict[str, List[Path]]) -> DataFrame:
    dfs = []
    for key, paths in info.items():
        df = DataFrame(columns=["path"], data=paths, index=paths)
        df["cls"] = str(key)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


PATH_DATA: dict[Dataset, DataFrame] = {
    # subs: "task", "rest" # noqa
    Dataset.Learning: to_path_frame(DATASETS["LEARNING"]),
    # subs: "allpain", "nopain", "duloxetine", "pain" # noqa
    Dataset.Osteo: to_path_frame(DATASETS["OSTEO"]),
    # subs: 'control', 'parkinsons', 'control_pre', 'park_pre' # noqa
    Dataset.Parkinsons: to_path_frame(DATASETS["PARKINSONS"]),
    # subs: "task", "rest" # noqa
    Dataset.ReflectionSummed: to_path_frame(DATASETS["REFLECT_SUMMED"]),
    # subs: "task", "rest" # noqa
    Dataset.ReflectionInterleaved: to_path_frame(DATASETS["REFLECT_INTERLEAVED"]),
    # subs: "high", "low" # noqa
    Dataset.VigilanceSes1: to_path_frame(DATASETS["PSYCH_VIGILANCE_SES-1"]),
    # subs: "high", "low" # noqa
    Dataset.VigilanceSes2: to_path_frame(DATASETS["PSYCH_VIGILANCE_SES-2"]),
    # subs: "high", "low" # noqa
    Dataset.TaskAttentionSes1: to_path_frame(DATASETS["PSYCH_TASK_ATTENTION_SES-1"]),
    # subs: "high", "low" # noqa
    Dataset.TaskAttentionSes2: to_path_frame(DATASETS["PSYCH_TASK_ATTENTION_SES-2"]),
    # subs: "high", "low" # noqa
    Dataset.WeeklyAttentionSes1: to_path_frame(DATASETS["PSYCH_WEEKLY_ATTENTION_SES-1"]),
    # subs: "high", "low" # noqa
    Dataset.WeeklyAttentionSes2: to_path_frame(DATASETS["PSYCH_WEEKLY_ATTENTION_SES-2"]),
}
PATH_DATA[Dataset.Vigilance] = pd.concat(
    [PATH_DATA[Dataset.VigilanceSes1], PATH_DATA[Dataset.VigilanceSes2]]
)
PATH_DATA[Dataset.TaskAttention] = pd.concat(
    [PATH_DATA[Dataset.TaskAttention], PATH_DATA[Dataset.TaskAttentionSes2]]
)
PATH_DATA[Dataset.WeeklyAttention] = pd.concat(
    [PATH_DATA[Dataset.WeeklyAttentionSes1], PATH_DATA[Dataset.WeeklyAttentionSes2]]
)

PATH_DATA_PRE: dict[Dataset, DataFrame] = {
    # subs: "task", "rest" # noqa
    Dataset.Learning: to_path_frame(DATASETS_FULLPRE["LEARNING"]),
    # subs: "allpain", "nopain", "duloxetine", "pain" # noqa
    Dataset.Osteo: to_path_frame(DATASETS_FULLPRE["OSTEO"]),
    # subs: 'control', 'parkinsons', 'control_pre', 'park_pre' # noqa
    Dataset.Parkinsons: to_path_frame(DATASETS_FULLPRE["PARKINSONS"]),
    # subs: "task", "rest" # noqa
    Dataset.ReflectionSummed: to_path_frame(DATASETS_FULLPRE["REFLECT_SUMMED"]),
    # subs: "task", "rest" # noqa
    Dataset.ReflectionInterleaved: to_path_frame(DATASETS_FULLPRE["REFLECT_INTERLEAVED"]),
    # subs: "high", "low" # noqa
    Dataset.VigilanceSes1: to_path_frame(DATASETS_FULLPRE["PSYCH_VIGILANCE_SES-1"]),
    # subs: "high", "low" # noqa
    Dataset.VigilanceSes2: to_path_frame(DATASETS_FULLPRE["PSYCH_VIGILANCE_SES-2"]),
    # subs: "high", "low" # noqa
    Dataset.TaskAttentionSes1: to_path_frame(
        DATASETS_FULLPRE["PSYCH_TASK_ATTENTION_SES-1"]
    ),
    # subs: "high", "low" # noqa
    Dataset.TaskAttentionSes2: to_path_frame(
        DATASETS_FULLPRE["PSYCH_TASK_ATTENTION_SES-2"]
    ),
    # subs: "high", "low" # noqa
    Dataset.WeeklyAttentionSes1: to_path_frame(
        DATASETS_FULLPRE["PSYCH_WEEKLY_ATTENTION_SES-1"]
    ),
    # subs: "high", "low" # noqa
    Dataset.WeeklyAttentionSes2: to_path_frame(
        DATASETS_FULLPRE["PSYCH_WEEKLY_ATTENTION_SES-2"]
    ),
}
PATH_DATA[Dataset.Vigilance] = pd.concat(
    [PATH_DATA[Dataset.VigilanceSes1], PATH_DATA[Dataset.VigilanceSes2]]
)
PATH_DATA[Dataset.TaskAttention] = pd.concat(
    [PATH_DATA[Dataset.TaskAttention], PATH_DATA[Dataset.TaskAttentionSes2]]
)
PATH_DATA[Dataset.WeeklyAttention] = pd.concat(
    [PATH_DATA[Dataset.WeeklyAttentionSes1], PATH_DATA[Dataset.WeeklyAttentionSes2]]
)


class ProcessedDataset:
    def __init__(self, source: Dataset, full_pre: bool) -> None:
        self.source = source
        self.full_pre = full_pre
        self.eig_paths: List[Path]

    def load_eigs(self) -> List[ndarray]:
        data = DATASETS_FULLPRE if self.full_pre else DATASETS

        if self.source is Dataset.Learning:
            data["LEARNING"]

        raise NotImplementedError()
