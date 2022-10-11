# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

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
from typing_extensions import Literal

from rmt.dataset import ProcessedDataset, rigidities
from rmt.enumerables import Dataset


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for prod in prods:
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def plot_rigidities(rigs: DataFrame, data: ProcessedDataset, degree: int) -> None:
    labels = rigs.y.unique().tolist()
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(rigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(f"{data}: deg={degree}")
    ax_idx = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1 = labels[i]
            label2 = labels[j]
            title = f"{label1} v {label2}"
            df1 = rigs.loc[rigs.y == label1].drop(columns="y")
            df2 = rigs.loc[rigs.y == label2].drop(columns="y")
            s = 2.0
            alpha = 0.3
            for k in range(len(df1)):
                axes.flat[ax_idx].scatter(
                    L + np.random.uniform(0, L_diff, len(L)),
                    np.log(df1.iloc[k]),
                    s=s,
                    alpha=alpha,
                    color="#016afe",
                    label=label1 if k == 1 else None,
                )
            for k in range(len(df2)):
                axes.flat[ax_idx].scatter(
                    L + np.random.uniform(0, L_diff, len(L)),
                    np.log(df2.iloc[k]),
                    s=s,
                    alpha=alpha,
                    color="#fe7b01",
                    label=label2 if k == 1 else None,
                )
            axes.flat[ax_idx].set_title(title)
            axes.flat[ax_idx].legend().set_visible(True)
            ax_idx += 1

    fig.set_size_inches(w=10, h=10)
    plt.show(block=False)


def plot_rigidity_sep(
    rigs: DataFrame, data: ProcessedDataset, degree: int, L_idx: int = -2
) -> None:
    labels = rigs.y.unique().tolist()
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(rigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, squeeze=False)
    fig.suptitle(f"{data}: deg={degree}")
    ax_idx = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = rigs.iloc[:, [L_idx, -1]]
            title = f"{labels[i]} v {labels[j]}: L={df.columns[0]}"
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            df2 = df.drop(columns="y").applymap(np.log)
            df2["log(rigidity)"] = df.y
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
                bins=20,
                common_norm=False,
                common_bins=False,
                log_scale=True,
            )
            sbn.stripplot(
                data=df2,
                x=df2.columns[0],
                y="log(rigidity)",
                hue="log(rigidity)",
                legend=False,
                ax=ax2,
                orient="h",
            )
            ax1.set_title(title)
            ax2.set_title(title)
            ax1.set_xlabel("rigidity")
            ax2.set_xlabel("log(rigidity)")
            ax_idx += 2

    fig.set_size_inches(w=ncols * 8, h=nrows * 5)
    plt.show(block=False)


if __name__ == "__main__":
    count = 0
    for source in Dataset:
        for degree in [5, 7, 9]:
            # data = ProcessedDataset(source=source, full_pre=False)
            data = ProcessedDataset(source=source, full_pre=True)
            rigs = rigidities(dataset=data, degree=degree, parallel=True)
            # level_vars = levelvars(dataset=data, degree=degree, parallel=True)
            # plot_rigidities(rigs, data, degree)
            plot_rigidity_sep(rigs, data, degree)
            count += 1
            if count % 3 == 0:
                plt.show()
            # sys.exit()