# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

import re
from shutil import copyfile
from enum import Enum
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

LETTER_WIDTH = 10.0
LETTER_HEIGHT = 8.0


class Grouping(Enum):
    CoarseFeature = "coarse_feature"
    FineFeature = "fine_feature"
    Classifier = "classifier"
    Subgroup = "subgroup"
    Degree = "deg"
    Trim = "trim"
    Preprocessing = "preproc"
    Slice = "slice"

    def order(self) -> list[str]:
        orders: dict[Grouping, list[str]] = {
            Grouping.CoarseFeature: list(GROSS_FEATURE_PALETTE.keys()),
            Grouping.FineFeature: list(FEATURE_GROUP_PALETTE.keys()),
            Grouping.Classifier: CLASSIFIER_ORDER,
            Grouping.Subgroup: SUBGROUP_ORDER,
            Grouping.Degree: DEGREE_ORDER,
            Grouping.Trim: TRIM_ORDER,
            Grouping.Preprocessing: PREPROC_ORDER,
            Grouping.Slice: SLICE_ORDER,
        }
        return orders[self]

    def palette(self) -> list[tuple[float, float, float]]:
        palettes: dict[Grouping, list[tuple[float, float, float]]] = {
            Grouping.CoarseFeature: GROSS_FEATURE_PALETTE,
            Grouping.FineFeature: FEATURE_GROUP_PALETTE,
        }
        if self not in palettes:
            raise ValueError(f"Grouping {self} does not have a palette defined")
        return palettes[self]


def kde_plot(
    data: DataFrame,
    metric: Metric,
    hue: Grouping | None,
    column: Grouping | None,
    row: Grouping | None,
    col_wrap: int | None,
    row_wrap: int | None,
    bw_adjust: float = 1.0,
    alpha: float = 0.8,
    # xlims: tuple[float, float] | None = None,
    # ylims: tuple[float, float] | None = None,
    add_lines: Literal["hlines", "vlines", False] | None = None,
    suptitle_desc: str = "",
    title_clean: list[dict[str, str]] | None = None,
    add_row_labels: bool = True,
    fix_xticks: bool = False,
    thinify: bool = True,
    w: float = LETTER_WIDTH,
    h: float = LETTER_HEIGHT,
    subplots_adjust: dict[str, float] | None = None,
    legend_pos: tuple[float, float] | str = "lower right",
) -> None:
    stitle = s_title(metric)
    sfname = s_fnmae(metric)

    print("Plotting...", end="", flush=True)
    grid = sbn.displot(
        data=data,
        x=metric,
        kind="kde",
        hue=hue.value if hue is not None else None,
        hue_order=hue.order() if hue is not None else None,
        palette=hue.palette() if hue is not None else None,
        fill=False,
        common_norm=False,
        row=row.value if row is not None else None,
        row_order=row.order() if row is not None else None,
        row_wrap=row_wrap,
        col=column.value if column is not None else None,
        col_order=column.order() if column is not None else None,
        col_wrap=col_wrap,
        bw_adjust=bw_adjust,
        alpha=alpha,
        facet_kws=dict(xlim=s_xlim(metric, kind="all"), sharey=False),
    )
    print("done")

    despine(grid)

    if title_clean is not None:
        for clean_args in title_clean:
            clean_titles(grid, **clean_args)
            # clean_titles(grid, ".*\n")
            # clean_titles(grid, "subgroup = ", split_at="-")
            # clean_titles(grid, "task_attend", "high")
            # clean_titles(grid, "task_nonattend", "low")
            # clean_titles(grid, "nonvigilant", "low")
            # clean_titles(grid, "vigilant", "high")
            # clean_titles(grid, "younger", "young")
            # clean_titles(grid, "duloxetine", "dlxtn")
    if add_row_labels:
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=CLASSIFIER_ORDER
        )
    if thinify:
        thinify_lines(grid)
    if add_lines:
        add_auroc_lines(grid, "vline", summary=metric)

    fig = grid.fig
    fig.suptitle(f"{stitle} {suptitle_desc}", fontsize=10)
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    fig.subplots_adjust(**subplots_adjust)
    sbn.move_legend(grid, loc=legend_pos)

    if fix_xticks:
        for i, ax in enumerate(fig.axes):
            if metric == "auroc":
                ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")

    savefig(fig, f"fine_feature_group_{sfname}_by_predictive_subgroup_classifier.png")
