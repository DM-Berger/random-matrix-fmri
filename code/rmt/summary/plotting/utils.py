# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

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

from rmt.summary.constants import (
    BLUE,
    ORNG,
    GREY,
    BLCK,
    PURP,
    RED,
    PINK,
    SPIE_MIN_LINE_WEIGHT,
    SPIE_OUTDIR,
    SPIE_PAPER_OUTDIR,
)

Metric = Literal["auroc", "f1", "acc+"]


def s_title(summary: Metric) -> str:
    return {"auroc": "AUROCs", "f1": "F1-Scores", "acc+": "Adjusted Accuracies"}[summary]


def s_fnmae(summary: Metric) -> str:
    return {"auroc": "aurocs", "f1": "f1s", "acc+": "accs"}[summary]


def s_xlim(
    summary: Metric, kind: Literal["all", "smallest", "largest"] = "all"
) -> tuple[float, float]:
    if summary == "auroc":
        if kind == "all":
            return (0.1, 0.9)
        elif kind == "largest":
            return (0.4, 1.0)
        else:
            return (0.1, 0.6)
    elif summary == "acc+":
        if kind == "all":
            return (-0.4, 0.4)
        elif kind == "largest":
            return (-0.1, 0.3)
        else:
            return (-0.4, 0.1)
    else:
        raise NotImplementedError()


def make_legend(fig: Figure, position: str | tuple[float, float] = "upper right") -> None:
    patches = [
        Patch(facecolor=RED, edgecolor="white", label="timeseries location feature"),
        Patch(facecolor=PINK, edgecolor="white", label="timeseries scale feature"),
        Patch(facecolor=GREY, edgecolor="white", label="smoothed eigenvalues feature"),
        Patch(facecolor=ORNG, edgecolor="white", label="max eigenvalues feature"),
        Patch(facecolor=BLCK, edgecolor="white", label="eigenvalues only feature"),
        Patch(facecolor=BLUE, edgecolor="white", label="RMT-only feature"),
        Patch(facecolor=PURP, edgecolor="white", label="RMT + eigenvalues feature"),
    ]

    fig.legend(handles=patches, loc=position)


def clean_titles(
    grid: FacetGrid,
    text: str = "subgroup = ",
    replace: str = "",
    split_at: Literal["-", "|"] | None = None,
) -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        axtitle = ax.get_title()
        if split_at is not None:
            ax.set_title(
                re.sub(text, replace, axtitle).replace(f" {split_at} ", "\n"), fontsize=8
            )
        else:
            ax.set_title(re.sub(text, replace, axtitle), fontsize=8)


def rotate_labels(grid: FacetGrid, axis: Literal["x", "y"] = "x") -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        if axis == "x":
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        else:
            plt.setp(ax.get_yticklabels(), rotation=40, ha="right")


def add_auroc_lines(
    grid: FacetGrid, kind: Literal["vline", "hline"], summary: Metric = "auroc"
) -> None:
    fig: Figure
    fig = grid.fig
    if summary == "f1":
        return
    guess = 0.5 if summary == "auroc" else 0.0

    for i, ax in enumerate(fig.axes):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        if kind == "vline":
            ax.vlines(
                x=guess,
                ymin=ymin,
                ymax=ymax,
                colors=["black"],
                linestyles="dotted",
                alpha=0.5,
                lw=SPIE_MIN_LINE_WEIGHT,
                label="guess" if i == 0 else None,
            )
        else:
            ax.hlines(
                y=guess,
                xmin=xmin,
                xmax=xmax,
                colors=["black"],
                linestyles="dotted",
                lw=SPIE_MIN_LINE_WEIGHT,
                alpha=0.5,
                label="guess" if i == 0 else None,
            )


def despine(grid: FacetGrid) -> None:
    sbn.despine(grid.fig, left=True)
    for ax in grid.fig.axes:
        ax.set_yticks([])


def thinify_lines(grid: FacetGrid) -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_linewidth(1.0)

    for line in grid.legend.legendHandles:
        line.set_linewidth(1.0)


def dashify_gross(grid: FacetGrid) -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_linewidth(1.0)
        try:
            ax.get_lines()[0].set_linestyle("solid")
            ax.get_lines()[1].set_linestyle("-.")
            ax.get_lines()[2].set_linestyle("--")
        except IndexError:
            pass

    for line in grid.legend.legendHandles:
        line.set_linewidth(1.0)
    try:
        grid.legend.legendHandles[0].set_linestyle("--")
        grid.legend.legendHandles[1].set_linestyle("-.")
        grid.legend.legendHandles[2].set_linestyle("solid")
    except IndexError:
        pass


def dashify_trims(grid: FacetGrid) -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_linewidth(1.0)
        try:
            ax.get_lines()[0].set_linestyle("solid")
            ax.get_lines()[1].set_linestyle("-.")
            ax.get_lines()[2].set_linestyle("--")
            ax.get_lines()[3].set_linestyle(":")
        except IndexError:
            pass

    for line in grid.legend.legendHandles:
        line.set_linewidth(1.0)
    try:
        grid.legend.legendHandles[0].set_linestyle(":")
        grid.legend.legendHandles[1].set_linestyle("--")
        grid.legend.legendHandles[2].set_linestyle("-.")
        grid.legend.legendHandles[3].set_linestyle("solid")
    except IndexError:
        pass


def make_row_labels(grid: FacetGrid, col_order: list[str], row_order: list[str]) -> None:
    ncols = len(col_order)
    row = 0
    for i, ax in enumerate(grid.fig.axes):
        if i == 0:
            ax.set_ylabel(row_order[row])
            row += 1
        elif i % ncols == 0 and i >= ncols:
            ax.set_ylabel(row_order[row])
            row += 1
        else:
            ax.set_ylabel("")


def savefig(fig: Figure, filename: str, show: bool = False) -> None:
    if show:
        print(f"Would have saved figure to file named: {filename}")
        plt.show()
        return
    print("Saving...", end="", flush=True)
    outfile = SPIE_OUTDIR / filename
    paper_outfile = SPIE_PAPER_OUTDIR / filename
    fig.savefig(outfile, dpi=600)
    print(f" saved figure to {outfile}")
    copyfile(outfile, paper_outfile)
    print(f"Copied saved figure to {paper_outfile}")
    plt.close()


def resize_fig() -> None:
    fig = plt.gcf()
    fig.set_size_inches(w=40, h=26)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, hspace=0.2)