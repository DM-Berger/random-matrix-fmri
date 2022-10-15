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
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt.enumerables import Dataset
from rmt.features import Eigenvalues, Feature, Levelvars, Rigidities

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
    # scores = cross_val_score(classifier(**kwargs), X, y, cv=cv, scoring="accuracy")
    scores = cross_validate(
        classifier(**kwargs), X, y, cv=cv, scoring=["accuracy", "roc_auc"]
    )
    acc = scores["test_accuracy"]
    auc = scores["test_roc_auc"]
    guess = np.max([np.mean(y), np.mean(1 - y)])
    mean = np.mean(acc).round(3)
    mn = np.min(acc)
    mx = np.max(acc)
    auroc = np.mean(auc)
    auroc_mn = np.min(auc)
    auroc_mx = np.max(auc)
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
            "auroc": auroc,
            "mean_acc": mean,
            "min_acc": mn,
            "max_acc": mx,
            "min_auroc": auroc_mn,
            "max_auroc": auroc_mx,
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
    feature: Feature,
    data: DataFrame,
    feature_idx: int | slice | None,
) -> DataFrame:
    if feature_idx is None:
        return data
    if not feature.is_combined:
        if isinstance(feature_idx, slice):
            df = data.drop(columns="y").iloc[:, feature_idx]
            df["y"] = data["y"]
            return df
        else:
            return data.iloc[:, [feature_idx - 1, -1]]

    # handle combined features
    idxs = feature.feature_start_idxs
    df = data.drop(columns="y")
    sub_dfs = []
    for i in range(len(idxs) - 1):
        start = idxs[i]
        stop = idxs[i + 1]
        sub_dfs.append(df.iloc[:, start:stop])

    if isinstance(feature_idx, slice):
        selecteds = [sub_df.iloc[:, feature_idx] for sub_df in sub_dfs]
        selected = pd.concat(selecteds, axis=1, ignore_index=True)
        selected["y"] = data["y"]
        return selected
    else:
        selecteds = [sub_df.iloc[:, feature_idx] for sub_df in sub_dfs]
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
    feature_idx: int | slice | None = None,
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
            df = select_features(feature, data, feature_idx)
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

    result["data"] = feature.source.name
    result["feature"] = feature.name
    result["preproc"] = "full" if feature.full_pre else "minimal"
    result["idx"] = feature_label(feature_idx)
    result["deg"] = str(feature.degree)
    return result.loc[
        :,
        [
            "data",
            "feature",
            "preproc",
            "deg",
            "norm",
            "idx",
            "comparison",
            "classifier",
            "acc+",
            "auroc",
            "mean_acc",
            "min_acc",
            "max_acc",
            "min_auroc",
            "max_auroc",
        ],
    ]


def predict_all(args: Namespace) -> DataFrame:
    feature = args.cls(
        source=args.source, full_pre=args.full_pre, norm=args.norm, degree=args.degree
    )
    return predict_feature(
        feature=feature,
        feature_idx=args.feature_idx,
        logarithm=True,
    )


def summarize_all_predictions(
    feature_cls: Type[Feature],
    sources: Optional[list[Dataset]] = None,
    degrees: Optional[list[int]] = None,
    feature_idxs: Optional[list[int | slice | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 200,
) -> DataFrame:
    sources = sources or [*Dataset]
    degrees = degrees or [3, 5, 7, 9]
    feature_idxs = feature_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                cls=[feature_cls],
                source=sources,
                degree=degrees,
                feature_idx=feature_idxs,
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