# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

import re
from shutil import copyfile
from typing import Literal
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Memory
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numba import njit
from numpy import ndarray
from pandas import DataFrame
from pandas.errors import PerformanceWarning
from seaborn import FacetGrid
from tqdm import tqdm

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_features import Eigenvalues, Levelvars, Rigidities, Unfolded
from rmt.summary.tables import print_correlations
from rmt.summary.plotting.kde import kde_plot, Grouping
from rmt.visualize import UPDATED_PLOT_OUTDIR as PLOT_OUTDIR
from rmt.visualize import best_rect
from rmt.summary.plotting.utils import (
    Metric,
    s_fnmae,
    s_title,
    s_xlim,
    make_legend,
    clean_titles,
    rotate_labels,
    add_auroc_lines,
    despine,
    thinify_lines,
    dashify_gross,
    dashify_trims,
    make_row_labels,
    savefig,
    resize_fig,
)
from rmt.summary.loading import (
    load_combined,
    get_described,
    get_described_w_classifier,
)
from rmt.summary.groupings import (
    fine_feature_grouping,
    slice_grouping,
    get_feature_ordering,
    make_palette,
    is_rmt,
)
from rmt.summary.constants import (
    get_aggregates,
    BLUE,
    LBLUE,
    ORNG,
    GREY,
    BLCK,
    PURP,
    RED,
    PINK,
    SPIE_OUTDIR,
    SPIE_PAPER_OUTDIR,
    SPIE_MIN_LINE_WEIGHT,
    SPIE_JMI_MAX_COL_WIDTH_INCHES,
    SPIE_JMI_MAX_WIDTH_INCHES,
    HEADER,
    FOOTER,
    DROPS,
    SUBGROUPERS,
    RMT_FEATURE_PALETTE,
    RMT_FEATURE_ORDER,
    FEATURE_GROUP_PALETTE,
    NON_BASELINE_PALETTE,
    GROSS_FEATURE_PALETTE,
    TRIM_ORDER,
    SLICE_ORDER,
    DEGREE_ORDER,
    SUBGROUP_ORDER,
    OVERALL_PREDICTIVE_GROUP_ORDER,
    CLASSIFIER_ORDER,
    PREPROC_ORDER,
    NORM_ORDER,
    AGGREGATES,
)


def plot_unfolded_duloxetine() -> None:
    fig: Figure
    ax: Axes
    args = dict(
        source=UpdatedDataset.Osteo,
        preproc=PreprocLevel.MotionCorrect,
        norm=True,
    )
    deg = 9
    trims = [TrimMethod.Precision, TrimMethod.Largest, TrimMethod.Middle]
    unfs = [Unfolded(degree=deg, trim=trim, **args).data for trim in trims]
    eigs = Eigenvalues(**args).data

    fig, axes = plt.subplots(ncols=len(unfs) + 1, nrows=1, sharex=True, sharey=False)
    for i, (unf, trim) in enumerate(zip(unfs, trims)):
        ax = axes[i + 1]
        ax.set_yscale("log")
        ax.set_title(f"trim = {trim.name}", fontsize=9)
        dulox = unf.drop(columns="y").loc[unf["y"] == "duloxetine"]
        nopain = unf.drop(columns="y").loc[unf["y"] == "nopain"]
        for k in range(len(dulox)):
            vals = dulox.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals,
                color=BLCK,
                lw=0.5,
                label="duloxetine" if k == 0 else None,
                alpha=0.5,
            )
        for k in range(len(nopain)):
            vals = nopain.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals, color=BLUE, lw=0.5, label="nopain" if k == 0 else None, alpha=0.5
            )

    ax = axes[0]
    ax.set_yscale("log")
    ax.set_title("Raw Eigenvalues", fontsize=9)
    dulox = eigs.drop(columns="y").loc[eigs["y"] == "duloxetine"]
    nopain = eigs.drop(columns="y").loc[eigs["y"] == "nopain"]
    for k in range(len(dulox)):
        vals = dulox.iloc[k, :]
        vals[vals <= 1e-3] = np.nan
        ax.plot(
            vals,
            color=BLCK,
            lw=0.5,
            label="duloxetine" if k == 0 else None,
            alpha=0.5,
        )
    for k in range(len(nopain)):
        vals = nopain.iloc[k, :]
        vals[vals <= 1e-3] = np.nan
        ax.plot(vals, color=BLUE, lw=0.5, label="nopain" if k == 0 else None, alpha=0.5)

    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

    fig.text(x=0.5, y=0.02, s="Feature Index", ha="center", fontsize=8)
    fig.text(
        y=0.5, x=0.02, s="Feature Value", va="center", rotation="vertical", fontsize=8
    )
    fig.set_size_inches(w=6.5, h=2)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.889, bottom=0.179, left=0.099, right=0.977, hspace=0.2, wspace=0.35
    )
    plt.show()
    savefig(fig, "unfolded_duloxetine_nopain.png")


def plot_unfolded(
    source: UpdatedDataset, preproc: PreprocLevel, group1: str, group2: str, degree: int
) -> None:
    fig: Figure
    ax: Axes
    args = dict(
        source=source,
        preproc=preproc,
        norm=True,
    )
    trims = [TrimMethod.Precision, TrimMethod.Largest, TrimMethod.Middle]
    unfs = [Unfolded(degree=degree, trim=trim, **args).data for trim in trims]
    eigs = Eigenvalues(**args).data

    fig, axes = plt.subplots(ncols=len(unfs) + 1, nrows=1, sharex=True, sharey=False)
    for i, (unf, trim) in enumerate(zip(unfs, trims)):
        ax = axes[i + 1]
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"trim = {trim.name}", fontsize=9)
        g1 = unf.drop(columns="y").loc[unf["y"] == group1]
        g2 = unf.drop(columns="y").loc[unf["y"] == group2]
        for k in range(len(g1)):
            vals = g1.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals,
                color=BLCK,
                lw=0.5,
                label=group1 if k == 0 else None,
                alpha=0.5,
            )
        for k in range(len(g2)):
            vals = g2.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals, color=LBLUE, lw=0.5, label=group2 if k == 0 else None, alpha=0.5
            )

    ax = axes[0]
    ax.set_yscale("log")
    ax.set_title("Raw Eigenvalues", fontsize=9)
    g1 = eigs.drop(columns="y").loc[eigs["y"] == group1]
    g2 = eigs.drop(columns="y").loc[eigs["y"] == group2]
    for k in range(len(g1)):
        vals = g1.iloc[k, :]
        vals[vals <= 1e-3] = np.nan
        ax.plot(
            vals,
            color=BLCK,
            lw=0.5,
            label=group1 if k == 0 else None,
            alpha=0.5,
        )
    for k in range(len(g2)):
        vals = g2.iloc[k, :]
        vals[vals <= 1e-3] = np.nan
        ax.plot(vals, color=LBLUE, lw=0.5, label=group2 if k == 0 else None, alpha=0.5)

    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    axes.flat[0].legend(frameon=False, fontsize=8).set_visible(True)

    fig.text(x=0.5, y=0.02, s="Feature Index", ha="center", fontsize=8)
    fig.text(
        y=0.5, x=0.02, s="Feature Value", va="center", rotation="vertical", fontsize=8
    )
    fig.set_size_inches(w=6.5, h=2)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.889, bottom=0.179, left=0.099, right=0.977, hspace=0.2, wspace=0.35
    )
    savefig(fig, f"unfolded_{source.value.lower()}_{group1}_v_{group2}.png")


def plot_observables(
    source: UpdatedDataset, preproc: PreprocLevel, group1: str, group2: str, degree: int
) -> None:
    fig: Figure
    ax: Axes
    args = dict(source=source, preproc=preproc, norm=True, degree=degree)
    trims = [TrimMethod.Precision, TrimMethod.Largest, TrimMethod.Middle]

    rigs = [Rigidities(trim=trim, **args).data for trim in trims]
    lvars = [Levelvars(trim=trim, **args).data for trim in trims]

    fig, axes = plt.subplots(ncols=len(trims), nrows=2, sharex=True, sharey=False)
    for i, (rig, trim) in enumerate(zip(rigs, trims)):
        ax = axes[0][i]
        ax.set_title(f"trim = {trim.name}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Rigidity", fontsize=9)
        g1 = rig.drop(columns="y").loc[rig["y"] == group1]
        g2 = rig.drop(columns="y").loc[rig["y"] == group2]
        for k in range(len(g1)):
            vals = g1.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals,
                color=BLCK,
                lw=0.5,
                label=group1 if k == 0 else None,
                alpha=0.5,
            )
        for k in range(len(g2)):
            vals = g2.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals, color=LBLUE, lw=0.5, label=group2 if k == 0 else None, alpha=0.5
            )

    for i, (lvar, trim) in enumerate(zip(lvars, trims)):
        ax = axes[1][i]
        ax.set_title(f"trim = {trim.name}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Level Variance", fontsize=9)
        g1 = lvar.drop(columns="y").loc[lvar["y"] == group1]
        g2 = lvar.drop(columns="y").loc[lvar["y"] == group2]
        for k in range(len(g1)):
            vals = g1.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals,
                color=BLCK,
                lw=0.5,
                label=group1 if k == 0 else None,
                alpha=0.5,
            )
        for k in range(len(g2)):
            vals = g2.iloc[k, :]
            vals[vals <= 0] = np.nan
            ax.plot(
                vals, color=LBLUE, lw=0.5, label=group2 if k == 0 else None, alpha=0.5
            )

    for ax in axes.flat:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=8).set_visible(True)

    fig.text(x=0.5, y=0.02, s="L", ha="center", fontsize=8)
    fig.text(
        y=0.5, x=0.02, s="Feature Value", va="center", rotation="vertical", fontsize=8
    )
    fig.set_size_inches(w=6.5, h=4)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.924, bottom=0.089, left=0.124, right=0.977, hspace=0.2, wspace=0.225
    )
    savefig(fig, f"observables_{source.value.lower()}_{group1}_v_{group2}.png")