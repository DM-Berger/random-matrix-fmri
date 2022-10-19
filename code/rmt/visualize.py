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
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from matplotlib.axes import Axes
from matplotlib.patches import Patch
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

from rmt.enumerables import Dataset
from rmt.features import Eigenvalues, Feature, Levelvars, Rigidities
from rmt.predict import log_normalize

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots"
BLUE = tuple(np.array((1, 70, 198)) / 255)

TITLE_ARGS = dict(fontsize=10)

Colors = dict[str, tuple[float, float, float]]


def outdir(feature_name: str) -> Path:
    out = PLOT_OUTDIR / feature_name
    out.mkdir(exist_ok=True, parents=True)
    return out


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


def flattenize(df: DataFrame) -> DataFrame:
    """make single-column, but retain labels"""
    dfs = []
    y = df["y"].copy()
    df = df.drop(columns="y")
    for row in range(len(df)):
        flat = DataFrame(
            data=df.iloc[row, :].to_numpy(), columns=["all"], index=range(len(df.columns))
        )
        flat["y"] = y.iloc[row]
        dfs.append(flat)
    return pd.concat(dfs, axis=0, ignore_index=True)


def get_alpha(df: DataFrame) -> float:
    a = 2 / len(df)
    a = float(np.clip(a, a_min=0.3, a_max=0.5))
    return a


def get_plot_dfs(
    data: DataFrame, label1: str, label2: str
) -> tuple[DataFrame, DataFrame]:
    df = data
    idx = (df.y == label1) | (df.y == label2)
    df = df.loc[idx]
    df_flat = flattenize(df)
    return df, df_flat


def plot_multi_raw(
    df: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
) -> None:
    capitalized = feature_name.capitalize()
    plural = f"{capitalized[:-1]}ies" if "rigidity" in feature_name else f"{capitalized}s"
    ax.set_title(f"Raw {plural}", **TITLE_ARGS)
    ax.set_xlabel(f"{capitalized} Index")
    ax.set_ylabel(f"log({feature_name.lower()})")
    y = df["y"].copy().to_numpy().ravel()

    for row in range(len(df)):
        cols = df.iloc[:, :-1].columns
        x = df.iloc[row, :-1].to_numpy()
        x[x >= 1.0] = np.nan
        ax.scatter(cols, x, color=colors[y[row]], s=0.25, alpha=get_alpha(df))


def plot_multi_raw_observable(
    df: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
    degree: float,
) -> None:
    capitalized = feature_name.capitalize()
    plural = f"{capitalized[:-1]}ies" if "rigidity" in feature_name else f"{capitalized}s"
    ax.set_title(f"Raw {plural}", **TITLE_ARGS)
    ax.set_xlabel("L")
    ax.set_ylabel(f"Unfolding Degree + log({feature_name.lower()})")
    y = df["y"].copy().to_numpy().ravel()
    df = df.copy()
    values = df.iloc[:, :-1].to_numpy()
    values += degree - 0.5
    df.iloc[:, :-1] = values

    for row in range(len(df)):
        cols = df.iloc[:, :-1].columns
        x = df.iloc[row, :-1].to_numpy()
        x[x >= 1.0 + degree - 0.5] = np.nan
        ax.plot(cols, x, color=colors[y[row]], lw=0.25, alpha=get_alpha(df))
    xlabels = list(df.iloc[:, :-1].columns)[::2]
    ylabels = [3, 5, 7, 9]
    ax.set_xticks(ticks=xlabels, labels=xlabels)
    ax.set_yticks(ticks=ylabels, labels=ylabels)


def plot_multi_hist_grouped(
    df: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
) -> None:
    capitalized = feature_name.capitalize()

    ax.set_title(f"{capitalized} Distributions (per sample)", **TITLE_ARGS)
    ax.set_xlabel(f"{capitalized}")

    y = df["y"].copy().to_numpy().ravel()

    for row in range(len(df)):
        x = df.iloc[row, :-1].to_numpy().astype(np.float64)
        x[x >= 1.0] = np.nan
        if np.sum(~np.isnan(x)) == 0 or np.nanvar(x) == 0:
            continue  # this bungles the KDE
        sbn.kdeplot(
            x=x,
            color=colors[y[row]],
            alpha=get_alpha(df),
            fill=False,
            ax=ax,
            legend=True,
            # common_norm=False,
            log_scale=False,
            lw=1.0,
        )
        # sbn.histplot(
        #     x=x,
        #     color=colors[y[row]],
        #     alpha=0.10,
        #     element="step",
        #     fill=True,
        #     ax=ax,
        #     legend=True,
        #     stat="density",
        #     bins=np.linspace(0, 1.0, 20),  # type: ignore
        #     common_norm=False,
        #     common_bins=False,
        #     log_scale=False,
        # )
    ymin, ymax = ax.get_ylim()
    if ymax > 100:
        ax.set_ylim(ymin, 100)


def plot_multi_hist_grouped_observables(
    df: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
    degree: float,
) -> None:
    capitalized = feature_name.capitalize()

    ax.set_title(f"{capitalized} Distributions (per sample)", **TITLE_ARGS)
    ax.set_xlabel(f"Unfolding Degree + {capitalized}")

    dodge = degree - 0.5
    values = df.drop(columns="y").to_numpy() + dodge
    df = df.copy()
    df.iloc[:, :-1] = values

    y = df["y"].copy().to_numpy().ravel()

    for row in range(len(df)):
        x = df.iloc[row, :-1].to_numpy().astype(np.float64)
        x[x >= 1.0 + dodge] = np.nan
        if np.sum(~np.isnan(x)) == 0 or np.nanvar(x) == 0:
            continue
        sbn.kdeplot(
            x=x,
            color=colors[y[row]],
            alpha=get_alpha(df),
            # fill=True,
            ax=ax,
            legend=True,
            # common_norm=False,
            log_scale=False,
            lw=1.0,
        )
        # sbn.histplot(
        #     x=x,
        #     color=colors[y[row]],
        #     alpha=0.10,
        #     element="step",
        #     fill=True,
        #     kde=True,
        #     ax=ax,
        #     legend=True,
        #     stat="density",
        #     bins=np.linspace(0 + dodge, 1.0 + dodge, 20),  # type: ignore
        #     common_norm=False,
        #     common_bins=False,
        #     log_scale=False,
        #     line_kws=dict(lw=0.5),
        #     lw=0.5,
        # )
    ymin, ymax = ax.get_ylim()
    if ymax > 100:
        ax.set_ylim(ymin, 100)
    xlabels = [3, 5, 7, 9]
    ax.set_xticks(ticks=xlabels, labels=xlabels)


def plot_multi_flat_hist(
    df_flat: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
) -> None:
    sbn.histplot(
        data=df_flat,
        x="all",
        hue="y",
        hue_order=sorted(colors.keys()),
        element="step",
        fill=True,
        ax=ax,
        legend=False,
        stat="density",
        bins=20,  # type: ignore
        common_norm=False,
        common_bins=False,
        log_scale=False,
    )
    ax.set_xlabel(f"log({feature_name.lower()})")
    ax.set_title(f"{feature_name.capitalize()} Distributions", **TITLE_ARGS)


def plot_multi_flat_hist_observables(
    df_flat: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
    degree: float,
) -> None:
    values = df_flat.drop(columns="y").to_numpy() + degree - 0.5
    df = df_flat.copy()
    df.iloc[:, :-1] = values

    sbn.histplot(
        data=df,
        x="all",
        hue="y",
        hue_order=sorted(colors.keys()),
        element="step",
        fill=True,
        ax=ax,
        legend=False,
        stat="density",
        bins=20,  # type: ignore
        common_norm=False,
        common_bins=False,
        log_scale=False,
    )
    ax.set_xlabel(f"Unfolding degree + log({feature_name.lower()})")
    ax.set_title(f"{feature_name.capitalize()} Distributions", **TITLE_ARGS)
    xlabels = [3, 5, 7, 9]
    ax.set_xticks(ticks=xlabels, labels=xlabels)


def plot_multi_flat_strip(
    df_flat: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
) -> None:
    sbn.stripplot(
        data=df_flat,
        x="all",
        y="y",
        hue="y",
        hue_order=sorted(colors.keys()),
        legend=False,  # type: ignore
        ax=ax,
        orient="h",
        s=1.0,
        alpha=0.2,
    )
    ax.set_xlabel(f"log({feature_name.lower()})")
    ax.set_title(f"{feature_name.capitalize()} Distributions", **TITLE_ARGS)


def plot_multi_flat_strip_observables(
    df_flat: DataFrame,
    ax: Axes,
    feature_name: str,
    colors: Colors,
    degree: float,
) -> None:
    values = df_flat.drop(columns="y").to_numpy() + degree - 0.5
    df = df_flat.copy()
    df.iloc[:, :-1] = values
    sbn.stripplot(
        data=df,
        x="all",
        y="y",
        hue="y",
        hue_order=sorted(colors.keys()),
        legend=False,  # type: ignore
        ax=ax,
        orient="h",
        s=1.0,
        alpha=0.2,
    )
    ax.set_xlabel(f"Unfolding degree + log({feature_name.lower()})")
    ax.set_title(f"{feature_name.capitalize()} Distributions", **TITLE_ARGS)
    xlabels = [3, 5, 7, 9]
    ax.set_xticks(ticks=xlabels, labels=xlabels)


def plot_feature(
    feature: Feature,
    save: bool = False,
) -> None:
    data = feature.data
    norm = feature.norm

    labels = data.y.unique().tolist()
    data = log_normalize(data, norm)
    combs = list(filter(is_not_dud_pairing, combinations(labels, 2)))
    N = len(combs)
    L = np.array(data.drop(columns="y").columns.to_list(), dtype=np.float64)
    L_diff = np.min(np.diff(L)) * 0.95
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(feature.suptitle)
    for i, (label1, label2) in enumerate(combs):
        title = f"{label1} v {label2}"
        df1 = data.loc[data.y == label1].drop(columns="y")
        df2 = data.loc[data.y == label2].drop(columns="y")
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
    feature: Feature,
    feature_idx: int = -1,
    save: bool = False,
) -> None:
    data = feature.data
    norm = feature.norm
    feature_name = feature.name

    labels = data.y.unique().tolist()
    data = log_normalize(data, norm)
    combs = list(filter(is_not_dud_pairing, combinations(labels, 2)))
    N = len(combs)
    nrows, ncols = best_rect(N)
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols * 2, squeeze=False)
    fig.suptitle(feature.suptitle)
    ax_idx = 0
    for label1, label2 in combs:
        if feature_idx is None:
            feature_idx = -1
        df = data.iloc[:, [feature_idx - 1, -1]]
        suffix = count_suffix(-feature_idx) if feature_idx < 0 else ""
        idx_label = (
            f"{-feature_idx}{suffix} largest" if feature_idx < 0 else f"{feature_idx}"
        )
        title = f"{label1} v {label2}: idx={idx_label}"
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


def plot_feature_multi(
    source: Dataset,
    full_pre: bool,
    norm: bool,
    save: bool = False,
) -> None:
    """Plot all features in various ways to visualize separations"""
    eigs = Eigenvalues(source=source, full_pre=full_pre, norm=norm, degree=None)
    rigs: dict[int, Rigidities] = {
        deg: Rigidities(source=source, full_pre=full_pre, norm=norm, degree=deg)
        for deg in [3, 5, 7, 9]
    }
    lvars: dict[int, Levelvars] = {
        deg: Levelvars(source=source, full_pre=full_pre, norm=norm, degree=deg)
        for deg in [3, 5, 7, 9]
    }

    eigs_data = eigs.data
    labels = eigs_data.y.unique().tolist()
    outdir = eigs.outdir().parent / "all_in_one"
    outdir.mkdir(exist_ok=True, parents=True)

    eigs_data = log_normalize(eigs_data, norm)
    rigs_data = {deg: log_normalize(rig.data, norm) for deg, rig in rigs.items()}
    lvars_data = {deg: log_normalize(lvar.data, norm) for deg, lvar in lvars.items()}

    combs = list(filter(is_not_dud_pairing, combinations(labels, 2)))
    N = len(combs)
    sbn.set_style("darkgrid")
    palette = [(0.0, 0.0, 0.0)] + sbn.color_palette("dark")  # type: ignore
    palette[1] = BLUE
    sbn.set_palette(palette)
    # nrows, ncols = best_rect(N)
    pbar = tqdm(total=N * (4 + 2 * 4 * 4))  # 4 eigs, 2 obs, 4 plots per obs, 4 degrees
    for k in range(N):
        label1, label2 = sorted(combs[k])  # need sorted for hue_order
        if not is_not_dud_pairing(combs[k]):
            continue
        if source is Dataset.Osteo:
            if label1 != "duloxetine":
                continue
            if label2 != "nopain":
                continue
        colors: Colors = {label1: palette[0], label2: palette[1]}
        fig, axes = plt.subplots(nrows=3, ncols=4, squeeze=False)
        desc = f"{source.name}: {label1} v {label2}"
        pbdesc = f"{label1} v {label2}"
        fig.suptitle(f"{desc} - All Features")
        pbar.set_description(pbdesc)

        pbar.set_description(f"{pbdesc} - Eigenvalues")
        df_eigs, df_flat = get_plot_dfs(eigs_data, label1, label2)
        plot_multi_raw(df_eigs, axes[0][0], "eigenvalue", colors)
        pbar.update()
        plot_multi_flat_hist(df_flat, axes[0][1], "eigenvalue", colors)
        pbar.update()
        plot_multi_flat_strip(df_flat, axes[0][2], "eigenvalue", colors)
        pbar.update()
        plot_multi_hist_grouped(df_eigs, axes[0][3], "eigenvalue", colors)
        pbar.update()

        for degree, rig_data in rigs_data.items():
            pbar.set_description(f"{pbdesc} - Rigidities [deg={degree}]")
            df_rigs, df_flat = get_plot_dfs(rig_data, label1, label2)
            args: Mapping[str, Any] = dict(feature_name="rigidity", degree=degree)
            cargs: Mapping[str, Any] = {**args, **dict(colors=colors)}
            plot_multi_raw_observable(df_rigs, axes[1][0], **cargs)
            pbar.update()
            plot_multi_flat_hist_observables(df_flat, axes[1][1], **cargs)
            pbar.update()
            plot_multi_flat_strip_observables(df_flat, axes[1][2], **cargs)
            pbar.update()
            plot_multi_hist_grouped_observables(df_rigs, axes[1][3], **cargs)
            pbar.update()

        for degree, lvar_data in lvars_data.items():
            pbar.set_description(f"{pbdesc} - Levelvars [deg={degree}]")
            df_lvar, df_flat = get_plot_dfs(lvar_data, label1, label2)
            args = dict(feature_name="level variance", degree=degree)
            cargs = {**args, **dict(colors=colors)}
            plot_multi_raw_observable(df_lvar, axes[2][0], **cargs)
            pbar.update()
            plot_multi_flat_hist_observables(df_flat, axes[2][1], **cargs)
            pbar.update()
            plot_multi_flat_strip_observables(df_flat, axes[2][2], **cargs)
            pbar.update()
            plot_multi_hist_grouped_observables(df_lvar, axes[2][3], **cargs)
            pbar.update()

        pbar.set_description(f"{pbdesc} - Saving")
        color1 = colors[label1]
        color1_alpha = (*color1, 0.3)
        color2 = colors[label2]
        color2_alpha = (*color2, 0.3)
        patches = [
            Patch(facecolor=color1_alpha, edgecolor=color1, label=label1),
            Patch(facecolor=color2_alpha, edgecolor=color2, label=label2),
        ]
        fig.legend(handles=patches, loc="upper right")

        fig.set_size_inches(w=18, h=8)
        fig.tight_layout()
        # plt.close()
        if save:
            normed = "_normed" if norm else ""
            fname = f"{source.name.lower()}_{label1}_v_{label2}_fullpre={full_pre}{normed}.png"
            outfile = outdir / fname
            pbar.set_description(f"{pbdesc} - Saving to {outfile}")
            fig.savefig(str(outfile), dpi=300)
            pbar.set_description(f"{pbdesc} - Saved to {outfile}")
            plt.close()
        else:
            plt.show()
        # plt.show(block=False)
    pbar.close()


def plot_all_features(
    sources: Optional[list[Dataset]] = None,
    feature_idxs: Optional[list[int | None]] = None,
    feature_cls: Type[Rigidities] | Type[Levelvars] | Type[Eigenvalues] = Eigenvalues,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    degrees: Optional[list[int]] = None,
    plot_separations: bool = False,
    save: bool = False,
) -> None:
    sources = sources or [*Dataset]
    feature_idxs = feature_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    _degrees = degrees or [3, 5, 7, 9]
    if feature_cls is Eigenvalues:
        _degrees = [None]
    norms = norms or [True, False]
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                feature_idx=feature_idxs,
                full_pre=full_pres,
                norm=norms,
                degree=_degrees,
            )
        )
    ]

    count = 0
    for args in tqdm(grid):
        feature = feature_cls(
            source=args.source, full_pre=args.full_pre, norm=args.norm, degree=args.degree
        )
        if plot_separations:
            plot_feature_separation(
                feature=feature,
                save=save,
                feature_idx=args.feature_idx,
            )
        else:
            plot_feature(feature=feature, save=save)
        count += 1
        if save:
            plt.savefig(feature_cls.outdir() / feature.fname, dpi=300)
            plt.close()
            continue
        if count % 5 == 0:
            plt.show()
    if save:
        print(f"Plots saved to {feature.outdir}")  # type: ignore


def plot_all_features_multi(
    sources: Optional[list[Dataset]] = None,
    full_pres: Optional[list[bool]] = None,
    save: bool = False,
) -> None:
    sources = sources or [*Dataset]
    full_pres = full_pres or [True, False]
    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                full_pre=full_pres,
                norm=[True],  # visualization too hard without
            )
        )
    ]

    pbar = tqdm(grid, leave=True)
    for args in pbar:
        fullpre = "fullpre" if args.full_pre else ""
        pbar.set_description(f"{args.source.name}: {fullpre}")
        plot_feature_multi(
            source=args.source,
            full_pre=args.full_pre,
            norm=args.norm,
            save=save,
        )
    pbar.close()
