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

from rmt.dataset import ProcessedDataset
from rmt.predict import log_normalize


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for prod in prods:
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def is_not_dud_pairing(pair: tuple[str, str]) -> bool:
    DUD_PAIRS = [
        ("control", "control_pre"),
        ("control", "park_pre"),
        ("parkinsons", "control_pre"),
        ("parkinsons", "park_pre"),
        ("control_pre", "control"),
        ("park_pre", "control"),
        ("control_pre", "parkinsons"),
        ("park_pre", "parkinsons"),
    ]
    for dud in DUD_PAIRS:
        if dud == pair:
            return False
    return True


@dataclass
class Feature:
    df: DataFrame
    name: str


def plot_feature(
    feature: DataFrame,
    data: ProcessedDataset,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = feature.y.unique().tolist()
    feature = log_normalize(feature, norm)
    combs = list(filter(is_not_dud_pairing, combinations(labels, 2)))
    N = len(combs)
    L = np.array(feature.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(f"{data}: norm={norm}")
    for i, (label1, label2) in enumerate(combs):
        title = f"{label1} v {label2}"
        df1 = feature.loc[feature.y == label1].drop(columns="y")
        df2 = feature.loc[feature.y == label2].drop(columns="y")
        s = 2.0
        alpha = 0.3
        ax: Axes = axes.flat[i]  # type: ignore
        for k in range(len(df1)):
            ax.scatter(
                L + np.random.uniform(0, L_diff, len(L)),
                df1.iloc[k],
                s=s,
                alpha=alpha,
                color="#016afe",
                label=label1 if k == 1 else None,
            )
        for k in range(len(df2)):
            ax.scatter(
                L + np.random.uniform(0, L_diff, len(L)),
                df2.iloc[k],
                s=s,
                alpha=alpha,
                color="#fe7b01",
                label=label2 if k == 1 else None,
            )
        ax.set_title(title)
        ax.legend().set_visible(True)

    fig.set_size_inches(w=10, h=10)
    if save:
        return
    plt.show(block=False)


def count_suffix(feature_idx: int) -> str:
    s = int(str(feature_idx)[-1])
    suffixes: dict[int, str] = {i: "th" for i in range(10)}
    suffixes = {**suffixes, **{1: "st", 2: "nd", 3: "rd"}}
    return suffixes[s]


def plot_feature_separation(
    data: ProcessedDataset,
    feature: DataFrame,
    feature_name: str,
    feature_idx: int = -1,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = feature.y.unique().tolist()
    feature = log_normalize(feature, norm)
    combs = list(filter(is_not_dud_pairing, combinations(labels, 2)))
    N = len(combs)
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, squeeze=False)
    fig.suptitle(f"{data}: norm={norm}")
    ax_idx = 0
    for label1, label2 in combs:
        if feature_idx is None:
            feature_idx = -1
        df = feature.iloc[:, [feature_idx - 1, -1]]
        suffix = count_suffix(-feature_idx) if feature_idx < 0 else ""
        eig_label = (
            f"{-feature_idx}{suffix} largest" if feature_idx < 0 else f"{feature_idx}"
        )
        title = f"{label1} v {label2}: eig_idx={eig_label}"
        idx = (df.y == label1) | (df.y == label2)
        df = df.loc[idx]
        # df2 = df.drop(columns="y").applymap(np.log)
        df2 = df.drop(columns="y")
        df2[f"log({feature_name})"] = df.y
        ax1: Axes = axes.flat[ax_idx]  # type: ignore
        ax2: Axes = axes.flat[ax_idx + 1]  # type: ignore
        sbn.histplot(
            data=df,
            x=df.columns[0],
            hue="y",
            element="step",
            fill=True,
            ax=ax1,
            legend=True,
            stat="density",
            bins=20,  # type: ignore
            common_norm=False,
            common_bins=False,
            log_scale=False,
        )
        sbn.stripplot(
            data=df2,
            x=df2.columns[0],
            y=f"log({feature_name})",
            hue=f"log({feature_name})",
            legend=False,  # type: ignore
            ax=ax2,
            orient="h",
        )
        ax1.set_title(title)
        ax2.set_title(title)
        ax1.set_xlabel(f"log({feature_name})")
        ax2.set_xlabel(f"log({feature_name})")
        ax_idx += 2

    fig.set_size_inches(w=ncols * 8, h=nrows * 5)
    fig.tight_layout()
    if save:
        return
    plt.show(block=False)
