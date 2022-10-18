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
from empyricalRMT.smoother import SmoothMethod
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
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
        self.is_combined = False
        self.feature_start_idxs = [0]

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


class EigsMinMax5(Feature):
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
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min5 = eigs.iloc[:, :5]
        max5 = eigs.drop(columns="y").iloc[:, -5:]
        df = pd.concat([min5, max5], axis=1)
        df["y"] = y
        return df


class EigsMinMax10(Feature):
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
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min10 = eigs.iloc[:, :10]
        max10 = eigs.drop(columns="y").iloc[:, -10:]
        df = pd.concat([min10, max10], axis=1)
        df["y"] = y
        return df


class EigsMinMax20(Feature):
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
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min20 = eigs.iloc[:, :20]
        max20 = eigs.drop(columns="y").iloc[:, -20:]
        df = pd.concat([min20, max20], axis=1)
        df["y"] = y
        return df


class EigsMiddle10(Feature):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=None,
        )
        self.length = 10

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigsMiddle20(EigsMiddle10):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=None,
        )
        self.length = 20

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigsMiddle40(Feature):
    def __init__(
        self, source: Dataset, full_pre: bool, norm: bool, degree: int | None
    ) -> None:
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=None,
        )
        self.length = 50

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigenvaluesSmoothed(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        df = self.dataset.eigs_df()
        y = df["y"].copy()
        df.drop(columns="y", inplace=True)
        smoothed = DataFrame(
            uniform_filter1d(df, size=self.degree, axis=-1, mode="constant")
        )
        smoothed.columns = df.columns
        smoothed["y"] = y
        return smoothed


class EigenvaluesSavGol(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        df = self.dataset.eigs_df()
        y = df["y"].copy()
        df.drop(columns="y", inplace=True)
        order = 2 if self.degree == 3 else 3
        smoothed = DataFrame(
            savgol_filter(
                df, window_length=self.degree, polyorder=order, axis=-1, mode="constant"
            )
        )
        smoothed.columns = df.columns
        smoothed["y"] = y
        return smoothed


class EigenvaluesPlusEigenvaluesSmoothed(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        df = self.dataset.eigs_df()
        df.drop(columns="y", inplace=True)
        smoothed = DataFrame(
            uniform_filter1d(df, size=self.degree, axis=-1, mode="constant")
        )
        smoothed.columns = df.columns

        df = pd.concat([eigs, smoothed], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigenvaluesPlusSavGol(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        df = self.dataset.eigs_df()
        df.drop(columns="y", inplace=True)
        order = 2 if self.degree == 3 else 3
        smoothed = DataFrame(
            savgol_filter(
                df, window_length=self.degree, polyorder=order, axis=-1, mode="constant"
            )
        )
        smoothed.columns = df.columns

        df = pd.concat([eigs, smoothed], axis=1, ignore_index=True)
        df["y"] = y
        return df


class Unfolded(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=degree,
        )

    @property
    def data(self) -> DataFrame:
        return self.dataset.unfolded_df(self.degree)


class EigPlusUnfolded(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        unfs = self.dataset.unfolded_df(self.degree)
        unfs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusRigidity(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusLevelvar(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusLevelvar(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([unfs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusRigidity(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([unfs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusUnfoldedPlusRigidity(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusUnfoldedPlusLevelvar(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class RigidityPlusLevelvar(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        y = rigs["y"].copy()
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))
        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusRigidityPlusLevelvar(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        y = rigs["y"].copy()
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))

        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([unfs, rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class AllFeatures(Feature):
    def __init__(self, source: Dataset, full_pre: bool, norm: bool, degree: int) -> None:
        self.degree: int
        super().__init__(
            source=source,
            full_pre=full_pre,
            norm=norm,
            degree=int(degree),
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(dataset=self.dataset, degree=self.degree, parallel=True)
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))

        lvars = levelvars(dataset=self.dataset, degree=self.degree, parallel=True)
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


FEATURE_OUTFILES: dict[Type[Feature], Path] = {
    AllFeatures: PROJECT / "all_combined_predictions.json",
    Eigenvalues: PROJECT / "eigenvalue_predictions.json",
    EigsMinMax5: PROJECT / "eigs-minmax-5_predictions.json",
    EigsMinMax10: PROJECT / "eigs-minmax-10_predictions.json",
    EigsMinMax20: PROJECT / "eigs-minmax-20_predictions.json",
    EigsMiddle10: PROJECT / "eigs-middle-10_predictions.json",
    EigsMiddle20: PROJECT / "eigs-middle-20_predictions.json",
    EigsMiddle40: PROJECT / "eigs-middle-40_predictions.json",
    EigenvaluesSmoothed: PROJECT / "eig_smoothed_predictions.json",
    EigenvaluesPlusEigenvaluesSmoothed: PROJECT
    / "eigenvalues+eig_smoothed_predictions.json",
    EigenvaluesPlusSavGol: PROJECT / "eigenvalues+eig_savgol_predictions.json",
    EigenvaluesSavGol: PROJECT / "eig_savgol_predictions.json",
    Rigidities: PROJECT / "rigidity_predictions.json",
    Levelvars: PROJECT / "levelvar_predictions.json",
    EigPlusLevelvar: PROJECT / "eig+levelvar_predictions.json",
    EigPlusRigidity: PROJECT / "eig+rigidity_predictions.json",
    EigPlusUnfolded: PROJECT / "eig+unfolded_predictions.json",
    EigPlusUnfoldedPlusLevelvar: PROJECT / "eig+unfolded+levelvar_predictions.json",
    EigPlusUnfoldedPlusRigidity: PROJECT / "eig+unfolded+rigidity_predictions.json",
    RigidityPlusLevelvar: PROJECT / "rigidity+levelvar_predictions.json",
    Unfolded: PROJECT / "unfolded_predictions.json",
    UnfoldedPlusLevelvar: PROJECT / "unfolded+levelvar_predictions.json",
    UnfoldedPlusRigidity: PROJECT / "unfolded+rigidity_predictions.json",
    UnfoldedPlusRigidityPlusLevelvar: PROJECT
    / "unfolded+rigidity+levelvar_predictions.json",
}

if __name__ == "__main__":
    feature = EigenvaluesSmoothed(Dataset.Learning, full_pre=False, norm=True, degree=3)
    print(feature.data)
    print(feature)
