# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sbn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from seaborn import FacetGrid

from rmt.summary.constants import (
    CLASSIFIER_ORDER,
    FEATURE_GROUP_PALETTE,
    PREPROC_ORDER,
    SLICE_ORDER,
    SUBGROUP_ORDER,
)
from rmt.summary.groupings import fine_feature_grouping, slice_grouping
from rmt.summary.loading import load_combined
from rmt.summary.plotting.utils import resize_fig


def summarize_performance_by_aggregation(
    metric: Literal["auroc", "f1", "acc+"], summarizer: Literal["median", "best"]
) -> None:
    ax: Axes
    fig: Figure

    sbn.set_style("darkgrid")
    # df = load_all_renamed()
    df = load_combined()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["fine_feature"] = df["feature"].apply(fine_feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)

    df = df.loc[~df.data.str.contains("Reflect")]
    df = df.loc[~df.data.str.contains("Ses")]

    # cleaner view
    # """
    sbn.displot(
        data=df,
        x="auroc",
        kind="kde",
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        col="subgroup",
        col_wrap=6,
        facet_kws=dict(ylim=(0.0, 0.5)),
    )
    fig = plt.gcf()
    for i, ax in enumerate(fig.axes):
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=0.5,
            ymin=ymin,
            ymax=ymax,
            colors=["black"],
            linestyles="dashed",
            alpha=0.7,
            label="guess" if i == 0 else None,
        )
    plt.show()
    # """

    # violin plot
    # """
    sbn.catplot(
        data=df,
        y="auroc",
        x="subgroup",
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="violin",
        linewidth=1.0,
    )
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle("Feature Distributions")
    for ax in fig.axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        xmin, xmax = ax.get_xlim()
        ax.hlines(
            y=0.5,
            xmin=xmin,
            xmax=xmax,
            colors=["black"],
            linestyles="dashed",
        )
    resize_fig()
    plt.show()
    # """

    # # Very useful: shows best RMT performance due to outlier performances
    sbn.catplot(
        data=df,
        y="auroc",
        x="subgroup",
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="box",
        col="preproc",
        col_order=PREPROC_ORDER,
        linewidth=0.5,
        fliersize=0.75,  # outlier markers
        legend_out=False,
    )
    fig = plt.gcf()
    # sbn.move_legend(fig, loc="upper right")
    for ax in fig.axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    resize_fig()
    plt.show()

    # Compare by classifier: AUROC < 0.5 = overfitting
    sbn.catplot(
        data=df,
        x="auroc",
        y="subgroup",
        order=SUBGROUP_ORDER,
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="box",
        row="preproc",
        row_order=PREPROC_ORDER,
        col="classifier",
        col_order=CLASSIFIER_ORDER,
        linewidth=0.25,
        fliersize=0.5,  # outlier markers
        legend_out=True,
    )
    fig = plt.gcf()
    # sbn.move_legend(fig, loc="upper right")
    for ax in fig.axes:
        # plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=0.5,
            ymin=ymin,
            ymax=ymax,
            colors=["black"],
            linestyles="dashed",
        )
    resize_fig()
    plt.show()

    # Also useful if you re-order hue and cols to show max_eigs most important
    sbn.catplot(
        data=df,
        y="auroc",
        x="slice",
        order=SLICE_ORDER,
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="box",
        col="preproc",
        col_order=PREPROC_ORDER,
        legend_out=False,
        linewidth=0.5,
        fliersize=0.5,  # outlier markers
    )
    resize_fig()
    plt.show()
    fig = plt.gcf()
    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    # This is important: Shows clear interaction between preproc and slice
    # """
    sbn.catplot(
        data=df,
        x="auroc",
        y="slice",
        order=SLICE_ORDER,
        hue="fine_feature",
        palette=FEATURE_GROUP_PALETTE,
        hue_order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="box",
        row="preproc",
        row_order=PREPROC_ORDER,
        col="subgroup",
        linewidth=0.25,
        fliersize=0.5,  # outlier markers
    )
    fig = plt.gcf()
    for ax in fig.axes:
        axtitle = ax.get_title()
        ax.set_title(
            axtitle[axtitle.find("|") + 1 :]
            .replace("subgroup = ", "")
            .replace(" - ", "\n"),
            fontsize=10,
        )
    resize_fig()
    fig.subplots_adjust(top=0.93, right=0.93)
    fig.text(x=0.45, y=0.5, s="Minimal preprocessing")
    fig.text(x=0.45, y=0.98, s="More preprocessing")
    plt.show()
    # """

    # This shows preproc / slice / feature interaction even more clearly than above
    # """
    sbn.catplot(
        data=df,
        x="auroc",
        col="subgroup",
        col_order=SUBGROUP_ORDER,
        y="fine_feature",
        kind="box",
        hue="slice",
        hue_order=SLICE_ORDER,
        row="preproc",
        row_order=PREPROC_ORDER,
        linewidth=0.25,
        fliersize=0.5,
    )
    fig = plt.gcf()
    for ax in fig.axes:
        axtitle = ax.get_title()
        ax.set_title(
            axtitle[axtitle.find("|") + 1 :]
            .replace("subgroup = ", "")
            .replace(" - ", "\n"),
            fontsize=10,
        )
    resize_fig()
    fig.subplots_adjust(top=0.93, right=0.93, left=0.055)
    fig.text(x=0.45, y=0.5, s="Minimal preprocessing")
    fig.text(x=0.45, y=0.98, s="More preprocessing")
    plt.show()
    # """

    # This one very clearly shows effect of preprocessing
    # """
    grid: FacetGrid = sbn.catplot(
        data=df,
        y="auroc",
        col="subgroup",
        col_order=SUBGROUP_ORDER,
        col_wrap=4,
        x="fine_feature",
        order=list(FEATURE_GROUP_PALETTE.keys()),
        kind="box",
        hue="preproc",
        hue_order=PREPROC_ORDER,
        linewidth=0.5,
        fliersize=0.75,
    )
    fig = plt.gcf()
    fig.suptitle("Effect of Preprocessing by Subgroup")
    for ax in fig.axes:
        axtitle = ax.get_title()
        ax.set_title(axtitle.replace("subgroup = ", ""), fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    grid.set_axis_labels(rotation=30)
    resize_fig()
    fig.subplots_adjust(top=0.93, right=0.945, bottom=0.105)
    plt.show()
