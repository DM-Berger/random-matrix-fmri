# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from empyricalRMT.eigenvalues import Eigenvalues, Unfolded
from empyricalRMT.observables.levelvariance import level_number_variance
from empyricalRMT.observables.rigidity import spectral_rigidity
from empyricalRMT.smoother import SmoothMethod
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt._types import Subgroups
from rmt.constants import DATA_ROOT, DATASETS, DATASETS_FULLPRE
from rmt.enumerables import Dataset

CACHE_DIR = ROOT.parent / "__OBSERVABLES_CACHE__"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MEMOIZER = Memory(location=str(CACHE_DIR))
L_VALUES = np.arange(1.0, 21.0, step=1.0)


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
    [PATH_DATA[Dataset.TaskAttentionSes1], PATH_DATA[Dataset.TaskAttentionSes2]]
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
PATH_DATA_PRE[Dataset.Vigilance] = pd.concat(
    [PATH_DATA_PRE[Dataset.VigilanceSes1], PATH_DATA_PRE[Dataset.VigilanceSes2]]
)
PATH_DATA_PRE[Dataset.TaskAttention] = pd.concat(
    [PATH_DATA_PRE[Dataset.TaskAttentionSes1], PATH_DATA_PRE[Dataset.TaskAttentionSes2]]
)
PATH_DATA_PRE[Dataset.WeeklyAttention] = pd.concat(
    [
        PATH_DATA_PRE[Dataset.WeeklyAttentionSes1],
        PATH_DATA_PRE[Dataset.WeeklyAttentionSes2],
    ]
)


class ProcessedDataset:
    def __init__(self, source: Dataset, full_pre: bool) -> None:
        self.source = source
        self.full_pre = full_pre
        data = PATH_DATA_PRE if self.full_pre else PATH_DATA
        self.path_info = data[self.source]
        self.id = f"{self.source.name}__fullpre={self.full_pre}"

    def labels(self) -> ndarray:
        return cast(ndarray, self.path_info["cls"].to_numpy())

    def eigs(self) -> list[ndarray]:
        def load(path: Path) -> ndarray:
            return cast(ndarray, np.load(path))

        return cast(list[ndarray], self.path_info["path"].apply(load).to_list())

    def unfolded(self, smoother: SmoothMethod, degree: int) -> list[Unfolded]:
        eigs: list[Eigenvalues] = [Eigenvalues(e) for e in self.eigs()]
        return [eig.unfold(smoother=smoother, degree=degree) for eig in eigs]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source.name}, full_pre={self.full_pre})"


@dataclass
class ObservableArgs:
    unfolded: ndarray
    L: ndarray


def _compute_rigidity(args: ObservableArgs) -> ndarray | None:
    """helper for `process_map`"""
    try:
        return spectral_rigidity(unfolded=args.unfolded, L=args.L, show_progress=False)[1]
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


def _compute_levelvar(args: ObservableArgs) -> ndarray | None:
    """helper for `process_map`"""
    try:
        unfolded = args.unfolded
        return level_number_variance(unfolded=unfolded, L=args.L, show_progress=False)[1]
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


# @MEMOIZER.cache
def rigidities(
    dataset: ProcessedDataset,
    degree: int,
    smoother: SmoothMethod = SmoothMethod.Polynomial,
    L: ndarray = L_VALUES,
    parallel: bool = True,
) -> DataFrame:
    L_hash = sha256(L.data.tobytes()).hexdigest()
    to_hash = ("rigidity", dataset.id, str(degree), smoother.name, L_hash)
    # quick and dirty hashing for caching  https://stackoverflow.com/a/1151705
    hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
    outfile = CACHE_DIR / f"{hsh}.json"
    if outfile.exists():
        print(f"Loading pre-computed rigidities from {outfile}")
        return pd.read_json(outfile)

    unfoldeds = dataset.unfolded(smoother=smoother, degree=degree)
    args = [ObservableArgs(unfolded=unf.vals, L=L) for unf in unfoldeds]
    if parallel:
        rigidities = process_map(
            _compute_rigidity, args, desc=f"Computing rigidities for {dataset}"
        )
    else:
        rigidities = list(
            map(_compute_rigidity, tqdm(args, desc=f"Computing rigidities for {dataset}"))
        )

    rigs, labels = [], []
    for rig, label in zip(rigidities, dataset.labels()):
        if rig is not None:
            rigs.append(rig)
            labels.append(label)
    df = DataFrame(data=np.stack(rigs, axis=0), columns=L)
    df["y"] = labels
    df.to_json(outfile, indent=2)
    print(f"Saved rigidities to {outfile}")
    return df


# @MEMOIZER.cache
def levelvars(
    dataset: ProcessedDataset,
    degree: int,
    smoother: SmoothMethod = SmoothMethod.Polynomial,
    L: ndarray = L_VALUES,
    parallel: bool = True,
) -> DataFrame:
    L_hash = sha256(L.data.tobytes()).hexdigest()
    to_hash = ("levelvar", dataset.id, str(degree), smoother.name, L_hash)
    # quick and dirty hashing for caching  https://stackoverflow.com/a/1151705
    hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
    outfile = CACHE_DIR / f"{hsh}.json"
    if outfile.exists():
        print(f"Loading pre-computed levelvars from {outfile}")
        return pd.read_json(outfile)

    unfoldeds = dataset.unfolded(smoother=smoother, degree=degree)
    args = [ObservableArgs(unfolded=unf.vals, L=L) for unf in unfoldeds]
    if parallel:
        rigidities = process_map(
            _compute_levelvar, args, desc=f"Computing level variances for {dataset}"
        )
    else:
        rigidities = list(
            map(
                _compute_levelvar,
                tqdm(args, desc=f"Computing level variances for {dataset}"),
            )
        )

    rigs, labels = [], []
    for rig, label in zip(rigidities, dataset.labels()):
        if rig is not None:
            rigs.append(rig)
            labels.append(label)
    df = DataFrame(data=np.stack(rigs, axis=0), columns=L)
    df["y"] = labels
    df.to_json(outfile, indent=2)
    print(f"Saved levelvars to {outfile}")
    return df


if __name__ == "__main__":
    for source in Dataset:
        for degree in [5, 7, 9]:
            data = ProcessedDataset(source=source, full_pre=True)
            rigs = rigidities(dataset=data, degree=degree, parallel=True)
            # level_vars = levelvars(dataset=data, degree=degree, parallel=True)