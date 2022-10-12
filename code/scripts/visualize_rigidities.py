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

from rmt.dataset import ProcessedDataset, rigidities
from rmt.enumerables import Dataset

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots/rigidities"
PLOT_OUTDIR.mkdir(exist_ok=True, parents=True)


def best_rect(m: int) -> Tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(np.floor(np.sqrt(m)))
    high = int(np.ceil(np.sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for prod in prods:
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def plot_rigidities(
    rigs: DataFrame,
    data: ProcessedDataset,
    degree: int,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = rigs.y.unique().tolist()
    if norm:
        x = rigs.drop(columns="y").applymap(np.log)
        X = DataFrame(minmax_scale(x)).applymap(np.exp)
        rigs = rigs.copy()
        rigs.iloc[:, :-1] = X
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(rigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(f"{data}: deg={degree} norm={norm}")
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
    if save:
        return
    plt.show(block=False)


def plot_rigidity_sep(
    rigs: DataFrame,
    data: ProcessedDataset,
    degree: int,
    L_idx: int = -2,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = rigs.y.unique().tolist()
    if norm:
        x = rigs.drop(columns="y").applymap(np.log)
        X = DataFrame(minmax_scale(x)).applymap(np.exp)
        rigs = rigs.copy()
        rigs.iloc[:, :-1] = X
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(rigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, squeeze=False)
    fig.suptitle(f"{data}: deg={degree} norm={norm}")
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
    if save:
        return
    plt.show(block=False)


def kfold_eval(
    X: ndarray, y: ndarray, classifier: Type, norm: bool, title: str, **kwargs: Mapping
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


def predict_rigidity_sep(
    rigs: DataFrame,
    data: ProcessedDataset,
    degree: int,
    L_idx: int | None = None,
    norm: bool = False,
) -> DataFrame:
    DUDS = [
        "control v control_pre",
        "control v park_pre",
        "parkinsons v control_pre",
        "parkinsons v park_pre",
    ]

    labels = rigs.y.unique().tolist()
    L_label = rigs.columns[L_idx - 1] if L_idx is not None else "All"
    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = rigs if L_idx is None else rigs.iloc[:, [L_idx - 1, -1]]
            title = f"{labels[i]} v {labels[j]}"
            skip = False
            for dud in DUDS:
                if dud in title:
                    skip = True
                    break
            if skip:
                continue
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            X = df.drop(columns="y").applymap(np.log).to_numpy()
            y = LabelEncoder().fit_transform(df.y.to_numpy())
            result_dfs = [
                kfold_eval(X, y, SVC, norm=norm, title=title),
                kfold_eval(X, y, LR, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    result = pd.concat(results, axis=0, ignore_index=True)

    result["deg"] = str(degree)
    result["data"] = data.source.name
    result["preproc"] = "full" if data.full_pre else "minimal"
    result["L"] = str(L_label)
    return result.loc[
        :,
        [
            "data",
            "preproc",
            "deg",
            "norm",
            "L",
            "comparison",
            "classifier",
            "acc+",
            "mean",
            "min",
            "max",
        ],
    ]


def plot_all_rigidities(
    sources: Optional[list[Dataset]] = None,
    degrees: Optional[list[int]] = None,
    L_idxs: Optional[list[int | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    plot_separations: bool = False,
    save: bool = False,
) -> None:
    sources = sources or [*Dataset]
    degrees = degrees or [3, 5, 7, 9]
    L_idxs = L_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                L_idx=L_idxs,
                full_pre=full_pres,
                degree=degrees,  # important this is last
                norm=norms,
            )
        )
    ]

    count = 0
    for args in tqdm(grid):
        # data = ProcessedDataset(source=source, full_pre=False)
        data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
        rigs = rigidities(dataset=data, degree=args.degree, parallel=True)
        # level_vars = levelvars(dataset=data, degree=degree, parallel=True)
        if plot_separations:
            plot_rigidity_sep(rigs, data, args.degree, norm=args.norm, save=save)
        else:
            plot_rigidities(rigs, data, args.degree, norm=args.norm, save=save)
        count += 1
        if save:
            fname = f"{args.source.name}_fullpre={args.full_pre}_norm={args.norm}_deg={args.degree}.png"
            plt.savefig(PLOT_OUTDIR / fname, dpi=300)
            plt.close()
            continue
        if count % 5 == 0:
            plt.show()
    if save:
        print(f"Plots saved to {PLOT_OUTDIR}")


def predict_data(args: Namespace) -> DataFrame:
    data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
    rigs = rigidities(dataset=data, degree=args.degree, parallel=True)
    return predict_rigidity_sep(
        rigs,
        data,
        degree=args.degree,
        L_idx=args.L_idx,
        norm=args.norm,
    )


def summarize_all_predictions(
    sources: Optional[list[Dataset]] = None,
    degrees: Optional[list[int]] = None,
    L_idxs: Optional[list[int | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 200,
) -> None:
    sources = sources or [*Dataset]
    degrees = degrees or [3, 5, 7, 9]
    L_idxs = L_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                degree=degrees,
                L_idx=L_idxs,
                full_pre=full_pres,
                norm=norms,
            )
        )
    ]
    # grid = grid[:100]
    dfs = process_map(predict_data, grid, desc="Predicting")
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by="acc+", ascending=False)
    print(df.iloc[:print_rows, :].to_markdown(index=False, tablefmt="simple"))

    corrs = pd.get_dummies(df.drop(columns=["data", "comparison"]))
    print("-" * 80)
    print("Spearman correlations")
    print("-" * 80)
    print(corrs.corr(method="spearman").loc["acc+"])
    corrs = corrs.loc[corrs["acc+"] > 0.0]
    print("-" * 80)
    print("Spearman correlations of actual predictive pairs")
    print("-" * 80)
    print(corrs.corr(method="spearman").loc["acc+"])


if __name__ == "__main__":
    DEGREES = [5, 7, 9]
    L_IDXS: List[int | None] = [None]
    plot_all_rigidities(
        degrees=DEGREES,
        plot_separations=True,
        norms=[True],
        save=True,
    )
    sys.exit()
    summarize_all_predictions(
        degrees=DEGREES,
        L_idxs=L_IDXS,
    )
