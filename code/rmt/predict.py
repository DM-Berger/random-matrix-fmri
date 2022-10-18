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
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt.enumerables import Dataset
from rmt.features import Eigenvalues, Feature, Levelvars, Rigidities

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"

# from source code
KNN_DEFAULTS = dict(
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
)


class FeatureSlice(Enum):
    All = "all"
    Min_05 = "min-05"
    Min_10 = "min-10"
    Min_25 = "min-25"
    Mid_05 = "mid-05"
    Mid_10 = "mid-10"
    Mid_25 = "mid-25"
    Max_05 = "max-05"
    Max_10 = "max-10"
    Max_25 = "max-25"

    def region(self) -> str:
        if self is FeatureSlice.All:
            return "all"
        return self.name.split("_")[0].lower()

    def slicer(self, feature_length: int) -> slice:
        if self is FeatureSlice.All:
            return slice(None)
        r = float(self.name.split("_")[1]) / 100
        base = r * feature_length
        region = self.region()
        idx = base // 2 if region == "mid" else round(base)
        if region == "min":
            idx = round(base)
            return slice(None, idx)
        elif region == "mid":
            idx = base // 2
            half = feature_length // 2
            return slice(half - idx, half + idx)
        elif region == "max":
            idx = round(base)
            return slice(-idx, None)
        else:
            raise ValueError("Impossible!")


class KNN3(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=3, **KNN_DEFAULTS)


class KNN5(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=5, **KNN_DEFAULTS)


class KNN9(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=9, **KNN_DEFAULTS)


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
    scores = cross_validate(
        classifier(**kwargs), X, y, cv=cv, scoring=["accuracy", "roc_auc", "f1"]
    )
    acc = scores["test_accuracy"]
    auc = scores["test_roc_auc"]
    f1 = np.mean(scores["test_f1"])
    guess = np.max([np.mean(y), np.mean(1 - y)])
    mean = np.mean(acc)
    auroc = np.mean(auc)
    return DataFrame(
        {
            "comparison": title,
            "classifier": classifier.__name__,
            "norm": norm,
            "acc+": mean - guess,
            "auroc": auroc,
            "acc": mean,
            "f1": f1,
        },
        index=[0],
    )


def select_features(
    feature: Feature,
    data: DataFrame,
    feature_slice: FeatureSlice,
) -> DataFrame:
    if feature_slice is FeatureSlice.All:
        return data

    if not feature.is_combined:
        length = len(data.columns) - 1
        slicer = feature_slice.slicer(length)
        df = data.drop(columns="y").iloc[:, slicer]
        df["y"] = data["y"]
        return df

    # handle combined features
    idxs = feature.feature_start_idxs
    df = data.drop(columns="y")
    sub_dfs = []
    for i in range(len(idxs) - 1):
        start = idxs[i]
        stop = idxs[i + 1]
        sub_dfs.append(df.iloc[:, start:stop])

    selecteds = []
    for sub_df in sub_dfs:
        length = len(sub_df.columns)
        slicer = feature_slice.slicer(length)
        selecteds.append(sub_df.iloc[:, slicer])
    selected = pd.concat(selecteds, axis=1, ignore_index=True)
    selected["y"] = data["y"]
    return selected


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
    feature: Feature,
    feature_slice: FeatureSlice,
    logarithm: bool = True,
) -> DataFrame:
    data = feature.data
    norm = feature.norm
    if logarithm:
        data = log_normalize(data, norm)
    labels = data.y.unique().tolist()

    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = select_features(feature, data, feature_slice)
            if is_dud_comparison(labels, i, j):
                continue
            title = f"{labels[i]} v {labels[j]}"
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            X = df.drop(columns="y").to_numpy()
            y: ndarray = LabelEncoder().fit_transform(df.y.to_numpy())  # type: ignore
            result_dfs = [
                kfold_eval(X, y, SVC, norm=norm, title=title),
                kfold_eval(X, y, RF, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
                kfold_eval(X, y, KNN3, norm=norm, title=title),
                kfold_eval(X, y, KNN5, norm=norm, title=title),
                kfold_eval(X, y, KNN9, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    result = pd.concat(results, axis=0, ignore_index=True)

    result["data"] = feature.source.name
    result["feature"] = feature.name
    result["preproc"] = "full" if feature.full_pre else "minimal"
    result["slice"] = feature_slice.value
    result["deg"] = str(feature.degree)
    return result.loc[
        :,
        [
            "data",
            "feature",
            "preproc",
            "deg",
            "norm",
            "slice",
            "comparison",
            "classifier",
            "acc+",
            "auroc",
            "acc",
            "f1"
        ],
    ]


def predict_all(args: Namespace) -> DataFrame:
    feature = args.cls(
        source=args.source, full_pre=args.full_pre, norm=args.norm, degree=args.degree
    )
    return predict_feature(
        feature=feature,
        feature_slice=args.feature_idx,
        logarithm=True,
    )


def summarize_all_predictions(
    feature_cls: Type[Feature],
    sources: Optional[list[Dataset]] = None,
    degrees: Optional[list[int]] = None,
    feature_slices: List[FeatureSlice] = [*FeatureSlice],
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 200,
) -> DataFrame:
    sources = sources or [*Dataset]
    degrees = degrees or [3, 5, 7, 9]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                cls=[feature_cls],
                source=sources,
                degree=degrees,
                feature_idx=feature_slices,
                full_pre=full_pres,
                norm=norms,
            )
        )
    ]
    # grid = grid[:100]
    dfs = process_map(predict_all, grid, desc="Predicting")
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by="acc+", ascending=False)
    print(df.iloc[:print_rows, :].to_markdown(index=False, tablefmt="simple"))

    corrs = pd.get_dummies(df.drop(columns=["data", "comparison"]))
    print("-" * 80)
    print("Spearman correlations")
    print("-" * 80)
    print(corrs.corr(method="spearman").loc["acc+"])
    corrs_pred = corrs.loc[corrs["acc+"] > 0.0]
    print("-" * 80)
    print("Spearman correlations of predictive pairs")
    print("-" * 80)
    print(corrs_pred.corr(method="spearman").loc["acc+"])
    return df