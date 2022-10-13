# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import traceback
from abc import ABC, abstractproperty
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt.dataset import ProcessedDataset, levelvars, rigidities
from rmt.enumerables import Dataset
from rmt.predict import log_normalize

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots"



class Feature(ABC):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None = None
    ) -> None:
        super().__init__()
        self.source = source
        self.full_pre = full_pre
        self.norm: bool = norm
        self.degree = degree
        self.name: str = self.__class__.__name__.lower()
        self.dataset: ProcessedDataset = ProcessedDataset(
            source=self.source,
            full_pre=self.full_pre,
        )

    @property
    def suptitle(self) -> str:
        deg = "" if self.degree is None else f" deg={self.degree}"
        return f"{self.dataset}: norm={self.norm}{deg}"

    @property
    def fname(self) -> str:
        deg = "" if self.degree is None else f"_deg={self.degree}"
        src = self.source.name
        pre = self.full_pre
        return f"{src}_fullpre={pre}_norm={self.norm}{deg}.png"

    @classmethod
    def outdir(cls) -> Path:
        out = PLOT_OUTDIR / cls.__name__.lower()
        out.mkdir(exist_ok=True, parents=True)
        return out

    @abstractproperty
    def data(self) -> DataFrame:
        ...


class Rigidities(Feature):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        assert degree is not None
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        return rigidities(dataset=self.dataset, degree=self.degree, parallel=True)


class Levelvars(Feature):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        assert degree is not None
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        return levelvars(dataset=self.dataset, degree=self.degree, parallel=True)


class Eigenvalues(Feature):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=None,
        )

    @property
    def data(self) -> DataFrame:
        return self.dataset.eigs_df()