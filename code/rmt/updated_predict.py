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
from math import ceil
from pathlib import Path
from pprint import pprint
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

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_features import UpdatedFeature

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

        # We have to be careful here: the smallest feature is ndim=20. At 5%,
        #
        #   base = 0.05 * 20 = 1,
        #
        # so base // 2 = 0 (!) we do ceil(base / 2) instead.
        if region == "min":
            idx = ceil(base)
            return slice(None, idx)
        elif region == "mid":
            idx = ceil(base / 2)
            half = feature_length // 2
            return slice(half - idx, half + idx)
        elif region == "max":
            idx = ceil(base)
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
    feature: UpdatedFeature,
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


# def predict_updated_feature(
#     feature: UpdatedFeature,
#     feature_slice: FeatureSlice,
#     logarithm: bool = True,
#     debug: bool = False,
# ) -> DataFrame | None:
#     data = feature.data
#     norm = feature.norm
#     if logarithm:
#         data = log_normalize(data, norm)
#     labels = data.y.unique().tolist()

#     results = []
#     for i in range(len(labels)):
#         for j in range(i + 1, len(labels)):
#             df = select_features(feature, data, feature_slice)
#             if len(df.columns) - 1 == 0:
#                 info = "\n".join(
#                     [
#                         f"Feature: {feature}",
#                         f"Dataset: {feature.source}",
#                         f"norm: {feature.norm}",
#                         f"preproc: {feature.preproc.name}",
#                         f"feature_slice: {feature_slice.name}",
#                         f"data shape before selection: {data.shape}",
#                         f"data before selection:\n{data}",
#                         f"data shape after selection: {df.shape}",
#                         f"data after selection:\n{df}",
#                     ]
#                 )
#                 raise IndexError(f"Some bullshit.\n{info}")
#             if is_dud_comparison(labels, i, j):
#                 continue
#             if debug:
#                 continue  # just make sure no shape errors
#             title = f"{labels[i]} v {labels[j]}"
#             idx = (df.y == labels[i]) | (df.y == labels[j])
#             df = df.loc[idx]
#             X = df.drop(columns="y").to_numpy()
#             y: ndarray = LabelEncoder().fit_transform(df.y.to_numpy())  # type: ignore
#             result_dfs = [
#                 # LR never converges, pointless
#                 kfold_eval(X, y, SVC, norm=norm, title=title),
#                 kfold_eval(X, y, RF, norm=norm, title=title),
#                 kfold_eval(X, y, GBC, norm=norm, title=title),
#                 kfold_eval(X, y, KNN3, norm=norm, title=title),
#                 kfold_eval(X, y, KNN5, norm=norm, title=title),
#                 kfold_eval(X, y, KNN9, norm=norm, title=title),
#             ]
#             results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
#     if debug:
#         return None

#     result = pd.concat(results, axis=0, ignore_index=True)
#     result["data"] = feature.source.name
#     result["feature"] = feature.name
#     result["preproc"] = feature.preproc.name
#     result["slice"] = feature_slice.value
#     result["deg"] = str(feature.degree)
#     result["trim"] = feature.trim.value if feature.trim else "none"
#     return result.loc[
#         :,
#         [
#             "data",
#             "feature",
#             "preproc",
#             "deg",
#             "trim",
#             "norm",
#             "slice",
#             "comparison",
#             "classifier",
#             "acc+",
#             "auroc",
#             "acc",
#             "f1",
#         ],
#     ]


def predict_updated_feature(
    feature: UpdatedFeature,
    feature_slice: FeatureSlice,
    logarithm: bool = True,
    debug: bool = False,
) -> DataFrame | None:
    data = feature.data
    norm = feature.norm
    if logarithm:
        data = log_normalize(data, norm)
    labels = data.y.unique().tolist()
    if len(labels) == 1:
        raise ValueError("Bad labeling!")

    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = select_features(feature, data, feature_slice)
            if len(df.columns) - 1 == 0:
                info = "\n".join(
                    [
                        f"Feature: {feature}",
                        f"Dataset: {feature.source}",
                        f"norm: {feature.norm}",
                        f"preproc: {feature.preproc.name}",
                        f"feature_slice: {feature_slice.name}",
                        f"data shape before selection: {data.shape}",
                        f"data before selection:\n{data}",
                        f"data shape after selection: {df.shape}",
                        f"data after selection:\n{df}",
                    ]
                )
                raise IndexError(f"Some bullshit.\n{info}")
            if is_dud_comparison(labels, i, j):
                continue
            if debug:
                continue  # just make sure no shape errors
            title = f"{labels[i]} v {labels[j]}"
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            X = df.drop(columns="y").to_numpy()
            y: ndarray = LabelEncoder().fit_transform(df.y.to_numpy())  # type: ignore
            result_dfs = [
                # LR never converges, pointless
                kfold_eval(X, y, SVC, norm=norm, title=title),
                kfold_eval(X, y, RF, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
                kfold_eval(X, y, KNN3, norm=norm, title=title),
                kfold_eval(X, y, KNN5, norm=norm, title=title),
                kfold_eval(X, y, KNN9, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    if debug:
        return None

    try:
        result = pd.concat(results, axis=0, ignore_index=True)
    except ValueError as e:
        message = traceback.format_exc()
        if "No objects to concatenate" in message:
            raise RuntimeError(
                f"""Something went wrong:
current df.shape: {df.shape}
"""
            )

    result["data"] = feature.source.name
    result["feature"] = feature.name
    result["preproc"] = feature.preproc.name
    result["slice"] = feature_slice.value
    result["deg"] = str(feature.degree)
    result["trim"] = feature.trim.value if feature.trim else "none"
    return result.loc[
        :,
        [
            "data",
            "feature",
            "preproc",
            "deg",
            "trim",
            "norm",
            "slice",
            "comparison",
            "classifier",
            "acc+",
            "auroc",
            "acc",
            "f1",
        ],
    ]


def predict_all_updated(args: Namespace) -> DataFrame | None:
    cls: Type[UpdatedFeature] = args.cls
    try:
        feature = cls(
            source=args.source,
            preproc=args.preproc,
            norm=args.norm,
            degree=args.degree,
            trim=args.trim_method,
        )
        return predict_updated_feature(
            feature=feature,
            feature_slice=args.feature_idx,
            logarithm=True,
            debug=args.debug,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Got error {e} for following combinations of parameters:")
        pprint(args.__dict__, indent=2, depth=2)
        return None


def summarize_all_updated_predictions(
    feature_cls: Type[UpdatedFeature],
    sources: Optional[list[UpdatedDataset]] = None,
    degrees: Optional[list[int]] = None,
    trims: Optional[list[TrimMethod | None]] = None,
    feature_slices: List[FeatureSlice] = [*FeatureSlice],
    preprocs: Optional[list[PreprocLevel]] = None,
    norms: Optional[list[bool]] = None,
    debug: bool = False,
) -> DataFrame:
    sources = sources or [*UpdatedDataset]
    trims = trims or [None, *TrimMethod]
    degrees = degrees or [3, 5, 7, 9]
    preprocs = preprocs or [*PreprocLevel]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                cls=[feature_cls],
                source=sources,
                degree=degrees,
                trim_method=trims,
                feature_idx=feature_slices,
                preproc=preprocs,
                norm=norms,
                debug=[debug],
            )
        )
    ]
    dfs = process_map(predict_all_updated, grid, desc="Predicting", chunksize=1)
    dfs = [df_ for df_ in dfs if df_ is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by="acc+", ascending=False)
    return df
