# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from enum import Enum
from typing import Literal, Mapping

import seaborn as sbn
from pandas import DataFrame

from rmt.summary.constants import (
    CLASSIFIER_ORDER,
    DEGREE_ORDER,
    FEATURE_GROUP_PALETTE,
    GROSS_FEATURE_PALETTE,
    NORM_ORDER,
    OVERALL_PREDICTIVE_GROUP_ORDER,
    PREPROC_ORDER,
    SLICE_ORDER,
    SUBGROUP_ORDER,
    TRIM_ORDER,
)
from rmt.summary.plotting.utils import (
    Metric,
    add_auroc_lines,
    clean_titles,
    dashify_gross,
    despine,
    make_row_labels,
    s_fnmae,
    s_title,
    s_xlim,
    savefig,
    thinify_lines,
)

LETTER_WIDTH = 10.0
LETTER_HEIGHT = 8.0


class Grouping(Enum):
    CoarseFeature = "coarse_feature"
    FineFeature = "fine_feature"
    Classifier = "classifier"
    Subgroup = "subgroup"
    PredictiveSubgroup = "predictive_subgroup"
    Norm = "norm"
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
            Grouping.PredictiveSubgroup: OVERALL_PREDICTIVE_GROUP_ORDER,
            Grouping.Norm: NORM_ORDER,
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


def get_fname(
    metric: Metric,
    modifier: str,
    hue: Grouping | None,
    column: Grouping | None,
    row: Grouping | None = None,
) -> str:
    metricname = s_fnmae(metric)
    colname = "" if column is None else f"_{column.value}"
    rowname = "" if row is None else f"_{row.value}"
    if colname == "" and rowname == "":
        return f"{hue.value}_{modifier}_{metricname}.png"
    return f"{hue.value}_{modifier}_{metricname}_by{colname}{rowname}.png"


def kde_plot(
    data: DataFrame,
    metric: Metric,
    hue: Grouping | None,
    col: Grouping | None,
    row: Grouping | None = None,
    col_wrap: int | None = None,
    bw_adjust: float = 1.0,
    alpha: float = 0.8,
    # xlims: tuple[float, float] | None = None,
    # ylims: tuple[float, float] | None = None,
    add_lines: Literal["hlines", "vlines", False] | None = None,
    suptitle_fmt: str | None = None,
    fname_modifier: str = "",
    title_clean: list[dict[str, str]] | None = None,
    add_row_labels: bool = True,
    fix_xticks: bool = False,
    thinify: bool = True,
    dashify: bool = False,
    w: float = LETTER_WIDTH,
    h: float = LETTER_HEIGHT,
    xlims: Literal["all", "smallest", "largest"] = "all",
    subplots_adjust: dict[str, float] | None = None,
    legend_pos: tuple[float, float] | str = "lower right",
    facet_kwargs: Mapping | None = None,
    show: bool = False,
) -> None:
    stitle = s_title(metric)
    if row is None:
        add_row_labels = False

    print("Plotting...", end="", flush=True)
    column = col.value if col is not None else None
    if "predictive" in column:
        data = data[data.subgroup.isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        column = "subgroup"

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
        col=column,
        col_order=col.order() if col is not None else None,
        col_wrap=col_wrap,
        bw_adjust=bw_adjust,
        alpha=alpha,
        facet_kws=dict(xlim=s_xlim(metric, kind=xlims), **facet_kwargs),
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
        if row is None:
            raise ValueError("`row` must not be none if you want row labels")
        make_row_labels(grid, col_order=col.order(), row_order=row.order())
    if thinify:
        thinify_lines(grid)
    if dashify:
        # add condition for feature or trims
        dashify_gross(grid)
    if add_lines:
        add_auroc_lines(grid, "vline", summary=metric)

    fig = grid.fig
    if suptitle_fmt is not None:
        fig.suptitle(suptitle_fmt.format(metric=stitle), fontsize=10)
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    fig.subplots_adjust(**subplots_adjust)
    sbn.move_legend(grid, loc=legend_pos)

    if fix_xticks:
        for i, ax in enumerate(fig.axes):
            if metric == "auroc":
                ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if col_wrap is None:
                if i >= len(col.order()):
                    ax.set_title("")

    savefig(
        fig,
        get_fname(metric, modifier=fname_modifier, hue=hue, column=col, row=row),
        show=show,
    )
