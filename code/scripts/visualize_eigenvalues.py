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
PLOT_OUTDIR = RESULTS / "plots/eigenvalues"
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


def log_normalize_eigs(eigs: DataFrame, norm: bool) -> DataFrame:
    x = eigs.drop(columns="y") + 1
    x = x.applymap(np.log)
    if norm:
        try:
            X = DataFrame(minmax_scale(x))
        except ValueError:
            traceback.print_exc()
            print(x)
            sys.exit(1)
    else:
        X = x
    eigs = eigs.copy()
    eigs.iloc[:, :-1] = X
    return eigs


def plot_eigenvalues(
    eigs: DataFrame,
    data: ProcessedDataset,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = eigs.y.unique().tolist()
    eigs = log_normalize_eigs(eigs, norm)
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(eigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(f"{data}: norm={norm}")
    ax_idx = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1 = labels[i]
            label2 = labels[j]
            title = f"{label1} v {label2}"
            df1 = eigs.loc[eigs.y == label1].drop(columns="y")
            df2 = eigs.loc[eigs.y == label2].drop(columns="y")
            s = 2.0
            alpha = 0.3
            ax: Axes = axes.flat[ax_idx]  # type: ignore
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
            ax_idx += 1

    fig.set_size_inches(w=10, h=10)
    if save:
        return
    plt.show(block=False)


def plot_eigvals_sep(
    eigs: DataFrame,
    data: ProcessedDataset,
    eig_idx: int = -2,
    norm: bool = False,
    save: bool = False,
) -> None:
    labels = eigs.y.unique().tolist()
    eigs = log_normalize_eigs(eigs, norm)
    combs = list(combinations(labels, 2))
    N = len(combs)
    L = np.array(eigs.drop(columns="y").columns.to_list(), dtype=np.float64)
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, squeeze=False)
    fig.suptitle(f"{data}: norm={norm}")
    ax_idx = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if eig_idx is None:
                eig_idx = -2
            df = eigs.iloc[:, [eig_idx - 1, -1]]
            eig_label = f"{-eig_idx}th largest" if eig_idx < 0 else f"{eig_idx}"
            title = f"{labels[i]} v {labels[j]}: eig_idx={eig_label}"
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            # df2 = df.drop(columns="y").applymap(np.log)
            df2 = df.drop(columns="y")
            df2["log(eigenvalue + 1)"] = df.y
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
                y="log(eigenvalue + 1)",
                hue="log(eigenvalue + 1)",
                legend=False,  # type: ignore
                ax=ax2,
                orient="h",
            )
            ax1.set_title(title)
            ax2.set_title(title)
            ax1.set_xlabel("log(eigenvalue + 1)")
            ax2.set_xlabel("log(eigenvalue + 1)")
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


def predict_eigval_sep(
    eigs: DataFrame,
    data: ProcessedDataset,
    eig_idx: int | None = None,
    norm: bool = False,
) -> DataFrame:
    DUDS = [
        "control v control_pre",
        "control v park_pre",
        "parkinsons v control_pre",
        "parkinsons v park_pre",
    ]

    labels = eigs.y.unique().tolist()
    eig_label = eigs.columns[eig_idx - 1] if eig_idx is not None else "All"
    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = eigs if eig_idx is None else eigs.iloc[:, [eig_idx - 1, -1]]
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
            y: ndarray = LabelEncoder().fit_transform(df.y.to_numpy())  # type: ignore
            result_dfs = [
                kfold_eval(X, y, SVC, norm=norm, title=title),
                kfold_eval(X, y, LR, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    result = pd.concat(results, axis=0, ignore_index=True)

    result["data"] = data.source.name
    result["preproc"] = "full" if data.full_pre else "minimal"
    result["idx"] = str(eig_label)
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


def plot_all_eigvals(
    sources: Optional[list[Dataset]] = None,
    eig_idxs: Optional[list[int | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    plot_separations: bool = False,
    save: bool = False,
) -> None:
    sources = sources or [*Dataset]
    eig_idxs = eig_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                eig_idx=eig_idxs,
                full_pre=full_pres,
                norm=norms,
            )
        )
    ]

    count = 0
    for args in tqdm(grid):
        data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
        eigs = data.eigs_df()
        if plot_separations:
            plot_eigvals_sep(eigs, data, norm=args.norm, save=save, eig_idx=args.eig_idx)
        else:
            plot_eigenvalues(eigs, data, norm=args.norm, save=save)
        count += 1
        if save:
            fname = f"{args.source.name}_fullpre={args.full_pre}_norm={args.norm}.png"
            plt.savefig(PLOT_OUTDIR / fname, dpi=300)
            plt.close()
            continue
        if count % 5 == 0:
            plt.show()
    if save:
        print(f"Plots saved to {PLOT_OUTDIR}")


def predict_data(args: Namespace) -> DataFrame:
    data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
    eigs = data.eigs_df()
    return predict_eigval_sep(
        eigs,
        data,
        eig_idx=args.eig_idx,
        norm=args.norm,
    )


def summarize_all_predictions(
    sources: Optional[list[Dataset]] = None,
    eig_idxs: Optional[list[int | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 200,
) -> None:
    sources = sources or [*Dataset]
    eig_idxs = eig_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                eig_idx=eig_idxs,
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
    # EIG_IDXS: List[int | None] = [None]
    EIG_IDXS = [None, -2]
    plot_all_eigvals(
        plot_separations=False,
        save=False,
    )
    sys.exit()
    summarize_all_predictions(
        eig_idxs=EIG_IDXS,
    )
