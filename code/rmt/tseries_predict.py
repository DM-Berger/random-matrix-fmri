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
from scipy.ndimage import uniform_filter1d
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

from rmt.enumerables import PreprocLevel, SeriesKind, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset
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


class KNN3(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=3, **KNN_DEFAULTS)


class KNN5(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=5, **KNN_DEFAULTS)


class KNN9(KNN):
    def __init__(self) -> None:
        super().__init__(n_neighbors=9, **KNN_DEFAULTS)


def normalize(df: DataFrame, norm: bool) -> DataFrame:
    """Expects a `df` with final label column `y` and other columns predictors"""
    x = df.drop(columns="y")
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


class TimeSeriesFeature:
    def __init__(
        self,
        source: UpdatedDataset,
        kind: SeriesKind,
        preproc: PreprocLevel,
        norm: bool,
        smoothing_degree: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.kind = kind
        self.preproc = preproc
        self.norm: bool = norm
        self.smoothing_degree = int(smoothing_degree)
        self.dataset: UpdatedProcessedDataset = UpdatedProcessedDataset(
            source=self.source,
            preproc_level=self.preproc,
        )
        self.is_combined = False
        self.info = self.dataset.get_information_frame(tseries=self.kind)

    @property
    def suptitle(self) -> str:
        smooth = f" smooth={self.smoothing_degree}"
        return f"{self.dataset}: norm={self.norm}{smooth} preproc={self.preproc.name}"

    @property
    def fname(self) -> str:
        smooth = f"smooth={self.smoothing_degree}"
        src = self.source.name
        pre = self.preproc.value
        return f"{src}_preproc={pre}_norm={self.norm}_{smooth}.png"

    def labels(self) -> DataFrame:
        return cast(ndarray, self.info["label"].to_numpy())

    def series(self) -> List[ndarray]:
        return list(map(lambda p: np.load(p), self.info.index))

    def series_df(self) -> DataFrame:
        raws = self.series()
        if self.smoothing_degree > 1:
            deg = self.smoothing_degree
            smooth_args = dict(size=deg, axis=-1, mode="constant")
            raws = [uniform_filter1d(series, **smooth_args) for series in raws]
        lengths = np.array([len(s) for s in raws])
        if not np.all(lengths == lengths[0]):
            # front zero-pad
            length = np.max(lengths)
            resized = []
            for series in raws:
                padded = np.zeros(length)
                padded[-len(series) :] = series
                resized.append(padded)
        else:
            resized = raws
        vals = np.stack(resized, axis=0)
        df = DataFrame(vals, columns=range(vals.shape[1]))
        df["y"] = self.labels()
        return df

    @property
    def data(self) -> DataFrame:
        return self.series_df()


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


def predict_tseries(
    feature: TimeSeriesFeature,
    debug: bool = False,
) -> DataFrame | None:
    data = feature.data
    norm = feature.norm
    data = normalize(data, norm)
    labels = data.y.unique().tolist()
    if len(labels) == 1:
        raise ValueError("Bad labeling!")

    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = data.copy()
            if len(df.columns) - 1 == 0:
                info = "\n".join(
                    [
                        f"Feature: {feature}",
                        f"Dataset: {feature.source}",
                        f"norm: {feature.norm}",
                        f"preproc: {feature.preproc.name}",
                        f"smooth_deg: {feature.smoothing_degree}",
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
    result["feature"] = feature.kind.name
    result["preproc"] = feature.preproc.name
    result["smooth"] = str(feature.smoothing_degree)
    return result.loc[
        :,
        [
            "data",
            "feature",
            "preproc",
            "smooth",
            "norm",
            "comparison",
            "classifier",
            "acc+",
            "auroc",
            "acc",
            "f1",
        ],
    ]


def predict_all_tseries(args: Namespace) -> DataFrame | None:
    try:
        feature = TimeSeriesFeature(
            source=args.source,
            kind=args.kind,
            preproc=args.preproc,
            norm=args.norm,
            smoothing_degree=args.smoothing_degree,
        )
        return predict_tseries(
            feature=feature,
            debug=args.debug,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Got error {e} for following combinations of parameters:")
        pprint(args.__dict__, indent=2, depth=2)
        return None


def summarize_all_tseries_predictions(
    kind: SeriesKind,
    sources: Optional[list[UpdatedDataset]] = None,
    smoothing_degrees: Optional[list[int]] = None,
    preprocs: Optional[list[PreprocLevel]] = None,
    norms: Optional[list[bool]] = None,
    debug: bool = False,
) -> DataFrame:
    sources = sources or [*UpdatedDataset]
    smoothing_degrees = smoothing_degrees or [1, 2, 4, 8, 16]
    preprocs = preprocs or [*PreprocLevel]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                kind=[kind],
                source=sources,
                smoothing_degree=smoothing_degrees,
                preproc=preprocs,
                norm=norms,
                debug=[debug],
            )
        )
    ]
    dfs = process_map(predict_all_tseries, grid, desc="Predicting", chunksize=1)
    dfs = [df_ for df_ in dfs if df_ is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by="acc+", ascending=False)
    return df


TSERIES_OUTFILES = {
    SeriesKind.Mean: PROJECT / "tseries_mean_predictions.json",
    SeriesKind.Max: PROJECT / "tseries_max_predictions.json",
    SeriesKind.Min: PROJECT / "tseries_min_predictions.json",
    SeriesKind.Median: PROJECT / "tseries_median_predictions.json",
    SeriesKind.Percentile95: PROJECT / "tseries_p95_predictions.json",
    SeriesKind.Percentile05: PROJECT / "tseries_p05_predictions.json",
    SeriesKind.IQR: PROJECT / "tseries_iqr_predictions.json",
    SeriesKind.Range: PROJECT / "tseries_range_predictions.json",
    SeriesKind.RobustRange: PROJECT / "tseries_robust-range_predictions.json",
}
