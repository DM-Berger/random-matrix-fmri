# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import traceback
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

from rmt.dataset import ProcessedDataset, levelvars
from rmt.enumerables import Dataset

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"


def log_normalize(df: DataFrame, norm: bool) -> DataFrame:
    """Expects a `df` with final label column `y` and other columns predictors"""
    x = df.drop(columns="y")
    x[x > 0] = x[x > 0].applymap(np.log)
    if norm:
        try:
            X = DataFrame(minmax_scale(x))
        except ValueError:
            traceback.print_exc()
            print(x)
            sys.exit(1)
    else:
        X = x
    df = df.copy()
    df.iloc[:, :-1] = X
    return df


def kfold_eval(
    X: ndarray,
    y: ndarray,
    classifier: Type,
    norm: bool,
    title: str,
    **kwargs: Mapping,
) -> DataFrame:
    if norm:
        X = minmax_scale(X, axis=0)
    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(classifier(**kwargs), X, y, cv=cv, scoring="accuracy")
    guess = np.max([np.mean(y), np.mean(1 - y)])
    mean = np.mean(scores).round(3)
    mn = np.min(scores)
    mx = np.max(scores)
    # print(
    #     f"  {title:>40} {classifier.__name__:>30} accs:mean={mean:0.3f} "
    #     f"[{np.min(scores):0.3f}, {np.max(scores):0.3f}] "
    #     f"({mean - guess:+0.3f})"
    # )
    return DataFrame(
        {
            "comparison": title,
            "classifier": classifier.__name__,
            "norm": norm,
            "acc+": mean - guess,
            "mean": mean,
            "min": mn,
            "max": mx,
        },
        index=[0],
    )


def feature_label(feature_idx: int | slice | None) -> str:
    if feature_idx is None:
        feat_label = "All"
    elif isinstance(feature_idx, slice):
        start, stop = feature_idx.start, feature_idx.stop
        if start is None and stop is not None:
            feat_label = f"[:{stop}]"
        elif start is None and stop is None:
            feat_label = f"[:]"
        elif start is not None and stop is None:
            feat_label = f"[{start}:]"
        elif start is not None and stop is not None:
            feat_label = f"[{start}:{stop}]"
        else:
            raise ValueError(f"Cannot interpret slice {feature_idx}")
    else:
        feat_label = str(int(feature_idx))
    return feat_label


def select_features(
    feature: DataFrame,
    feature_idx: int | slice | None,
) -> DataFrame:
    if feature_idx is None:
        df = feature
    elif isinstance(feature_idx, slice):
        df = feature.drop(columns="y").iloc[:, feature_idx]
        df["y"] = feature["y"]
    else:
        df = feature.iloc[:, [feature_idx - 1, -1]]
    return df


def is_dud_comparison(labels: list[str], i: int, j: int) -> bool:
    DUD_PAIRS = [
        "control v control_pre",
        "control v park_pre",
        "parkinsons v control_pre",
        "parkinsons v park_pre",
    ]
    title = f"{labels[i]} v {labels[j]}"
    for dud in DUD_PAIRS:
        if dud in title:
            return True
    return False


def predict_feature(
    feature: DataFrame,
    data: ProcessedDataset,
    feature_idx: int | slice | None = None,
    norm: bool = False,
    logarithm: bool = False,
) -> DataFrame:

    if logarithm:
        feature = log_normalize(feature, norm)
    labels = feature.y.unique().tolist()

    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = select_features(feature, feature_idx)
            if is_dud_comparison(labels, i, j):
                continue
            title = f"{labels[i]} v {labels[j]}"
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            X = df.drop(columns="y").to_numpy()
            y: ndarray = LabelEncoder().fit_transform(df.y.to_numpy())  # type: ignore
            result_dfs = [
                kfold_eval(X, y, SVC, norm=norm, title=title),
                # kfold_eval(X, y, LR, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    result = pd.concat(results, axis=0, ignore_index=True)

    result["data"] = data.source.name
    result["preproc"] = "full" if data.full_pre else "minimal"
    result["idx"] = feature_label(feature_idx)
    return result.loc[
        :,
        [
            "data",
            "preproc",
            "norm",
            "idx",
            "comparison",
            "classifier",
            "acc+",
            "mean",
            "min",
            "max",
        ],
    ]