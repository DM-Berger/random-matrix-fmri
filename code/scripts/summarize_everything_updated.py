# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
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
from rmt.summary.plotting.kde import kde_plot
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


PROJECT = ROOT.parent
MEMORY = Memory(PROJECT / "__JOBLIB_CACHE__")


def make_kde_plots() -> None:
    ax: Axes
    fig: Figure
    sbn.set_style("ticks")

    print("Loading data...", end="", flush=True)
    df = load_combined()
    print(" done")

    def plot_by_coarse_feature() -> None:
        ax: Axes
        fig: Figure
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            fill=False,
            common_norm=False,
            palette=GROSS_FEATURE_PALETTE,
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(ylim=(0.0, 15.0), xlim=(0.2, 0.9)),
        )
        add_auroc_lines(grid, kind="vline")
        clean_titles(grid, split_at="-")
        fig = grid.fig
        fig.suptitle(
            "Distribution of AUROCs by Coarse Feature Group",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        despine(grid)
        dashify_gross(grid)
        savefig(fig, "coarse_feature_overall_by_subgroup.png")

    def plot_largest_by_coarse_feature() -> None:
        """Too big / complicated"""
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            # hue="fine_feature",
            # hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            # palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            row="classifier",
            row_order=CLASSIFIER_ORDER,
            # row_order=[PREPROC_ORDER[0], PREPROC_ORDER[-1]],
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            # col_wrap=5,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(ylim=(0.0, 75.0), xlim=(0.5, 1.0)),
        )
        add_auroc_lines(grid, kind="vline")
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        dashify_gross(grid)
        fig = grid.fig
        fig.suptitle(
            "Distributions of Largest 500 AUROCs for each combination "
            "of Coarse Feature Group, Dataset, and Classifier",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "coarse_feature_largest_by_subgroup_data.png")

    def plot_largest_by_coarse_feature_subgroup() -> None:
        """THIS IS GOOD. LOOK AT MODES. In only ony case are eigs or rmt mode auroc
        worse than tseries alone, i.e. modally, RMT or eigs are better than tseries.
        """
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            # facet_kws=dict(ylim=(0.0, 75.0), xlim=(0.5, 1.0), sharey=False),
            facet_kws=dict(xlim=(0.5, 1.0), sharey=False),
        )
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        dashify_gross(grid)
        fig = grid.fig
        fig.suptitle(
            "Distributions of Largest 500 AUROCs for each Combination "
            "of Coarse Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "coarse_feature_largest_by_subgroup.png")

    def plot_largest_by_fine_feature_subgroup(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nlargest(500, summary)
        )
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            # facet_kws=dict(ylim=(0.0, 75.0), xlim=(0.5, 1.0), sharey=False),
            facet_kws=dict(xlim=s_xlim(summary, kind="largest"), sharey=False),
        )
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        thinify_lines(grid)
        fig = grid.fig
        fig.suptitle(
            f"Distributions of Largest 500 {stitle} "
            "for each Combination of Fine Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, f"fine_feature_largest_{sfname}_by_subgroup.png")

    def plot_smallest_by_coarse_feature_subgroup() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=(0.1, 0.6), sharey=False),
        )
        print("done")
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        dashify_gross(grid)
        fig = grid.fig
        fig.suptitle(
            "Distributions of Smallest 500 AUROCs for each Combination "
            "of Coarse Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "coarse_feature_smallest_by_subgroup.png")

    def plot_smallest_by_fine_feature_subgroup(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nsmallest(500, summary)
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, "smallest"), sharey=False),
        )
        print("done")
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        thinify_lines(grid)
        fig = grid.fig
        fig.suptitle(
            f"Distributions of Smallest 500 {stitle} "
            "for each Combination of Fine Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, f"fine_feature_smallest_{sfname}_by_subgroup.png")

    def plot_largest_by_fine_feature_groups() -> None:
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=(0.2, 1.0)),
        )
        add_auroc_lines(grid, kind="vline")
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        # dashify_gross(grid)
        fig = grid.fig
        fig.suptitle("Distribution of Largest 500 AUROCs for Fine Feature Groups")
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=10)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.03, right=0.985, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.72, 0.08))
        savefig(fig, "fine_feature_group_largests_by_subgroup.png")

    def plot_smallest_by_fine_feature_groups() -> None:
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(ylim=(0.0, 75.0), xlim=(0.1, 0.6)),
        )
        add_auroc_lines(grid, kind="vline")
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        despine(grid)
        # dashify_gross(grid)
        fig = grid.fig
        fig.suptitle("Distribution of Smallest 500 AUROCs for each Fine Feature Grouping")
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=10)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.05, right=0.985, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.76, 0.11))
        savefig(fig, "fine_feature_group_smallest_by_subgroup.png")

    def plot_all_by_fine_feature_groups(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        grid = sbn.displot(
            data=df,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=s_xlim(summary, kind="all")),
        )
        add_auroc_lines(grid, kind="vline", summary=summary)
        despine(grid)
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        fig = grid.fig
        fig.suptitle(f"Overall Distribution of {stitle}s for each Fine Feature Group")
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.03, right=0.995, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.76, 0.11))
        savefig(fig, f"all_{sfname}_by_fine_feature_groups.png")

    def plot_all_by_fine_feature_groups_best_params() -> None:
        rmt = df.loc[df.coarse_feature.isin(["rmt"])]
        rmt = rmt.loc[rmt.trim.isin(["middle", "largest"])]
        rmt = rmt.loc[rmt.deg.isin([3, 9])]
        rmt = rmt.loc[rmt.preproc.isin(["MotionCorrect", "MNIRegister"])]
        rmt = rmt.loc[rmt.slice.isin(["max-10", "mid-25"])]
        idx = rmt.feature.isin(["unfolded"])
        rmt = rmt[idx]

        other = df.loc[df.coarse_feature.isin(["eigs", "tseries"])]
        data = pd.concat([rmt, other], axis=0)

        grid = sbn.displot(
            data=data,
            x="auroc",
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=(0.1, 1.0)),
        )
        thinify_lines(grid)
        add_auroc_lines(grid, kind="vline")
        despine(grid)
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        fig = grid.fig
        fig.suptitle("Overall Distribution of AUROCs for each Fine Feature Group")
        fig.tight_layout()
        fig.set_size_inches(w=10, h=8)
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.03, right=0.995, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.76, 0.09))
        savefig(fig, "best_rmt_params_by_subgroup.png")

    def plot_all_by_fine_feature_groups_slice(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        data = df.loc[df.coarse_feature.isin(["rmt", "eigs"])]
        grid = sbn.displot(
            data=data,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(NON_BASELINE_PALETTE.keys()),
            palette=NON_BASELINE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="slice",
            row_order=SLICE_ORDER,
            bw_adjust=0.8,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=s_xlim(summary, kind="all")),
        )
        thinify_lines(grid)
        add_auroc_lines(grid, kind="vline", summary=summary)
        despine(grid)
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, r"slice = .+\| ")
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=SLICE_ORDER
        )
        fig = grid.fig
        fig.suptitle(
            f"Overall Distribution of {stitle} for each Fine Feature Group by Slicing"
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.07, left=0.036, right=0.99, hspace=1.0, wspace=0.128
        )
        sbn.move_legend(grid, loc=(0.01, 0.85))
        savefig(fig, f"rmt_eigs_{sfname}_by_subgroup_and_slicing.png")

    def plot_all_by_fine_feature_groups_best_params_best_slice() -> None:
        rmt = df.loc[df.coarse_feature.isin(["rmt", "eigs"])]
        rmt = rmt.loc[rmt.trim.isin(["middle", "largest"])]
        rmt = rmt.loc[rmt.deg.isin([3, 9])]
        rmt = rmt.loc[rmt.preproc.isin(["MotionCorrect", "MNIRegister"])]
        rmt = rmt.loc[rmt.slice.isin(["max-05", "max-10", "mid-25"])]
        idx = rmt.feature.isin(["unfolded"])
        rmt = rmt[idx]

        # other = df.loc[df.coarse_feature.isin(["eigs"])]
        # data = pd.concat([rmt, other], axis=0)
        data = rmt

        palette = {**FEATURE_GROUP_PALETTE}
        palette.pop("tseries loc")
        palette.pop("tseries scale")

        grid = sbn.displot(
            data=data,
            x="auroc",
            kind="kde",
            hue="fine_feature",
            hue_order=list(palette.keys()),
            palette=palette,
            fill=False,
            common_norm=False,
            col="subgroup",
            # col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            col_order=SUBGROUP_ORDER,
            col_wrap=3,
            # row="slice",
            # row_order=SLICE_ORDER,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=(0.1, 1.0)),
        )
        thinify_lines(grid)
        add_auroc_lines(grid, kind="vline")
        despine(grid)
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, r"slice = .+\| ")
        # make_row_labels(
        #     grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=SLICE_ORDER
        # )
        fig = grid.fig
        fig.suptitle("Overall Distribution of AUROCs Using Best Analytic Choices")
        fig.tight_layout()
        fig.set_size_inches(w=10, h=8)
        fig.subplots_adjust(
            top=0.892, bottom=0.05, left=0.03, right=0.995, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.70, 0.04))
        savefig(fig, "best_rmt_params_by_subgroup.png")

    def plot_largest_by_coarse_feature_preproc() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            # col="subgroup",
            # col_order=SUBGROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            # col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=(0.5, 1.0), sharey=False),
        )
        print("done")
        clean_titles(grid, "preproc = ")
        despine(grid)
        dashify_gross(grid)
        fig = grid.fig
        fig.suptitle(
            "Largest 500 AUROCs by Preprocessing Degree",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_COL_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.05, 0.75))
        for ax in fig.axes:
            ax.set_ylabel("")
        savefig(fig, "coarse_feature_largest_by_preproc.png")

    def plot_smallest_by_coarse_feature_preproc() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "coarse_feature"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            # col="subgroup",
            # col_order=SUBGROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            # col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 0.5), sharey=False),
        )
        print("done")
        clean_titles(grid, "preproc = ")
        despine(grid)
        dashify_gross(grid)
        fig = grid.fig
        fig.suptitle(
            "Smallest 500 AUROCs by Preprocessing Degree",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.05, 0.75))
        for ax in fig.axes:
            ax.set_ylabel("")
        savefig(fig, "coarse_feature_smallest_by_preproc.png")

    def plot_by_coarse_feature_preproc_subgroup() -> None:
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            # col_wrap=4,
            bw_adjust=2.0,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 1.0), sharey=False),
        )
        print("done")
        clean_titles(grid, "preproc = ", split_at="|")
        clean_titles(grid, "subgroup = ")
        despine(grid)
        dashify_gross(grid)
        add_auroc_lines(grid, "vline")
        fig = grid.fig
        fig.suptitle(
            "AUROCs by Preprocessing Degree",
            fontsize=10,
        )
        fig.set_size_inches(w=11, h=8.5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.05, 0.75))
        for ax in fig.axes:
            ax.set_ylabel("")
        plt.show()
        savefig(fig, "coarse_feature_by_preproc_subgroup.png")

    def plot_by_coarse_predictive_feature_preproc_subgroup() -> None:
        dfp = df.loc[df["subgroup"].isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfp,
            x="auroc",
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            # col_wrap=4,
            bw_adjust=2.0,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 1.0), sharey=False),
        )
        print("done")
        clean_titles(grid, "preproc = ", split_at="|")
        clean_titles(grid, ".*\n")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "task_attend", "high")
        clean_titles(grid, "task_nonattend", "low")
        clean_titles(grid, "nonvigilant", "low")
        clean_titles(grid, "vigilant", "high")
        clean_titles(grid, "younger", "young")
        clean_titles(grid, "duloxetine", "dlxtn")
        despine(grid)
        dashify_gross(grid)
        add_auroc_lines(grid, "vline")
        fig = grid.fig
        fig.suptitle(
            "AUROCs by Preprocessing for Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.085, left=0.055, right=0.98, hspace=0.2, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.01, 0.88))
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=PREPROC_ORDER
        )
        for i, ax in enumerate(fig.axes):
            ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")
        savefig(fig, "coarse_feature_by_preproc_predictive_subgroup.png")

    def plot_by_fine_predictive_feature_preproc_subgroup(
        summary: Metric = "auroc",
    ) -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        dfp = df.loc[df["subgroup"].isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfp,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            bw_adjust=2.0,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, kind="all"), sharey=False),
        )
        print("done")
        clean_titles(grid, "preproc = ", split_at="|")
        clean_titles(grid, ".*\n")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "task_attend", "high")
        clean_titles(grid, "task_nonattend", "low")
        clean_titles(grid, "nonvigilant", "low")
        clean_titles(grid, "vigilant", "high")
        clean_titles(grid, "younger", "young")
        clean_titles(grid, "duloxetine", "dlxtn")
        despine(grid)
        thinify_lines(grid)
        add_auroc_lines(grid, "vline", summary=summary)
        fig = grid.fig
        fig.suptitle(
            f"{stitle} by Preprocessing for Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.085, left=0.1, right=0.98, hspace=0.2, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.01, 0.80))
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=PREPROC_ORDER
        )
        for i, ax in enumerate(fig.axes):
            if summary == "auroc":
                ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            else:
                pass
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")
        savefig(fig, f"fine_feature_{sfname}_by_preproc_predictive_subgroup.png")

    def plot_by_fine_feature_group_predictive_norm_subgroup() -> None:
        dfp = df.loc[df["subgroup"].isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfp,
            x="auroc",
            kind="kde",
            # hue="coarse_feature",
            # hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            # palette=GROSS_FEATURE_PALETTE,
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="norm",
            row_order=NORM_ORDER,
            # col_wrap=4,
            bw_adjust=1.5,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 1.0), sharey=False),
        )
        print("done")
        clean_titles(grid, "norm = ", split_at="|")
        clean_titles(grid, ".*\n")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "task_attend", "high")
        clean_titles(grid, "task_nonattend", "low")
        clean_titles(grid, "nonvigilant", "low")
        clean_titles(grid, "vigilant", "high")
        clean_titles(grid, "younger", "young")
        clean_titles(grid, "duloxetine", "dlxtn")
        make_row_labels(
            grid,
            col_order=list(FEATURE_GROUP_PALETTE.keys()),
            row_order=NORM_ORDER,  # type: ignore
        )
        despine(grid)
        dashify_gross(grid)
        add_auroc_lines(grid, "vline")
        fig = grid.fig
        fig.suptitle(
            "AUROCs by Normalization for Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.085, left=0.055, right=0.98, hspace=0.2, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.01, 0.88))
        for i, ax in enumerate(fig.axes):
            ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")
        savefig(fig, "fine_feature_group_by_norm_predictive_subgroup.png")

    def plot_by_coarse_feature_group_predictive_classifier_subgroup(
        summary: Metric = "auroc",
    ) -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        dfp = df.loc[df["subgroup"].isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfp,
            x=summary,
            kind="kde",
            hue="coarse_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            row="classifier",
            row_order=CLASSIFIER_ORDER,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            bw_adjust=1.5,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, kind="all"), sharey=False),
        )
        print("done")
        clean_titles(grid, "norm = ", split_at="|")
        clean_titles(grid, ".*\n")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "task_attend", "high")
        clean_titles(grid, "task_nonattend", "low")
        clean_titles(grid, "nonvigilant", "low")
        clean_titles(grid, "vigilant", "high")
        clean_titles(grid, "younger", "young")
        clean_titles(grid, "duloxetine", "dlxtn")
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=CLASSIFIER_ORDER
        )
        despine(grid)
        dashify_gross(grid)
        add_auroc_lines(grid, "vline", summary=summary)
        fig = grid.fig
        fig.suptitle(
            f"{stitle} by Classifier for Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=20, h=15)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.085, left=0.055, right=0.98, hspace=0.2, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.01, 0.88))
        for i, ax in enumerate(fig.axes):
            if summary == "auroc":
                ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")
        savefig(
            fig, f"coarse_feature_group_{sfname}_by_predictive_subgroup_classifier.png"
        )

    def plot_by_fine_feature_group_predictive_classifier_subgroup(
        summary: Metric = "auroc",
    ) -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        dfp = df.loc[df["subgroup"].isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfp,
            x=summary,
            kind="kde",
            hue="fine_feature",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            row="classifier",
            row_order=CLASSIFIER_ORDER,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            bw_adjust=1.5,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, kind="all"), sharey=False),
        )
        print("done")
        clean_titles(grid, "norm = ", split_at="|")
        clean_titles(grid, ".*\n")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "task_attend", "high")
        clean_titles(grid, "task_nonattend", "low")
        clean_titles(grid, "nonvigilant", "low")
        clean_titles(grid, "vigilant", "high")
        clean_titles(grid, "younger", "young")
        clean_titles(grid, "duloxetine", "dlxtn")
        make_row_labels(
            grid, col_order=OVERALL_PREDICTIVE_GROUP_ORDER, row_order=CLASSIFIER_ORDER
        )
        despine(grid)
        thinify_lines(grid)
        add_auroc_lines(grid, "vline", summary=summary)
        fig = grid.fig
        fig.suptitle(
            f"{stitle} by Classifier for Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=20, h=15)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.82, bottom=0.085, left=0.055, right=0.98, hspace=0.2, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.01, 0.88))
        for i, ax in enumerate(fig.axes):
            if summary == "auroc":
                ax.set_xticks([0.25, 0.5, 0.75], [0.25, "", 0.75], fontsize=8)
            if i >= len(OVERALL_PREDICTIVE_GROUP_ORDER):
                ax.set_title("")
        savefig(fig, f"fine_feature_group_{sfname}_by_predictive_subgroup_classifier.png")

    def plot_rmt_by_trim(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        data = df.loc[
            df.feature.apply(
                lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                and ("eigs" not in s)
            )
        ]
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=data,
            x=summary,
            kind="kde",
            hue="feature",
            hue_order=RMT_FEATURE_ORDER,
            palette=RMT_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="trim",
            row_order=TRIM_ORDER,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, kind="all"), sharey=False),
        )
        print("done")
        clean_titles(grid, "trim = ")
        clean_titles(grid, "none")
        clean_titles(grid, "precision")
        clean_titles(grid, "largest")
        clean_titles(grid, "middle")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "feature = ", split_at="|")
        despine(grid)
        # dashify_trims(grid)
        thinify_lines(grid)
        add_auroc_lines(grid, "vline", summary=summary)
        make_row_labels(
            grid,
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row_order=TRIM_ORDER,
        )
        fig = grid.fig
        fig.suptitle(
            f"RMT Feature {stitle} by Trimming on Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.795, bottom=0.085, left=0.04, right=0.96, hspace=0.35, wspace=0.22
        )
        sbn.move_legend(grid, loc=(0.01, 0.825))
        savefig(fig, f"rmt_feature_{sfname}_by_trim.png")

    def plot_rmt_by_degree(summary: Metric = "auroc") -> None:
        stitle = s_title(summary)
        sfname = s_fnmae(summary)
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df.loc[
                df.feature.apply(
                    lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                    and ("eigs" not in s)
                )
            ],
            x=summary,
            kind="kde",
            hue="feature",
            hue_order=RMT_FEATURE_ORDER,
            palette=RMT_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="deg",
            row_order=DEGREE_ORDER,
            # col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=s_xlim(summary, kind="all"), sharey=False),
        )
        print("done")
        # clean_titles(grid, " = ", split_at="|")
        clean_titles(grid, r"deg = [0-9]")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "feature = ", split_at="|")
        despine(grid)
        thinify_lines(grid)
        add_auroc_lines(grid, "vline", summary=summary)
        fig = grid.fig
        fig.suptitle(
            f"RMT Feature {stitle} by Degree on Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.79, bottom=0.085, left=0.04, right=0.96, hspace=0.35, wspace=0.22
        )
        sbn.move_legend(grid, loc=(0.01, 0.825))
        make_row_labels(
            grid,
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row_order=[f"degree = {d}" for d in DEGREE_ORDER],
        )
        savefig(fig, f"rmt_feature_{sfname}_by_degree.png")

    def plot_rmt_largest_by_degree() -> None:
        print("Plotting...", end="", flush=True)
        data = df.loc[
            df.feature.apply(
                lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                and ("eigs" not in s)
            )
        ]
        df_largest = data.groupby(["deg", "feature"]).apply(
            lambda grp: grp.nlargest(2000, "auroc")
        )
        grid = sbn.displot(
            data=df_largest,
            x="auroc",
            kind="kde",
            hue="feature",
            hue_order=RMT_FEATURE_ORDER,
            palette=RMT_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row="deg",
            row_order=DEGREE_ORDER,
            # col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 1.0), sharey=False),
        )
        print("done")
        # clean_titles(grid, " = ", split_at="|")
        clean_titles(grid, r"deg = [0-9]")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "feature = ", split_at="|")
        despine(grid)
        thinify_lines(grid)
        add_auroc_lines(grid, "vline")
        fig = grid.fig
        fig.suptitle(
            "RMT Feature Largest 2000 AUROCs by Degree on Predictable Data",
            fontsize=10,
        )
        fig.set_size_inches(w=10, h=8)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.79, bottom=0.085, left=0.04, right=0.96, hspace=0.35, wspace=0.22
        )
        sbn.move_legend(grid, loc=(0.01, 0.83))
        make_row_labels(
            grid,
            col_order=OVERALL_PREDICTIVE_GROUP_ORDER,
            row_order=[f"degree = {d}" for d in DEGREE_ORDER],
        )
        savefig(fig, "rmt_feature_auroc_largest_by_degree.png")

    # these are not great
    # plot_overall()

    # plot_largest_by_coarse_feature()
    # plot_by_coarse_feature()
    # plot_largest_by_coarse_feature_subgroup()
    # plot_smallest_by_coarse_feature_subgroup()
    # plot_largest_by_fine_feature_groups()
    # plot_smallest_by_fine_feature_groups()

    # plot_by_coarse_feature_preproc()
    # plot_largest_by_coarse_feature_preproc()
    # plot_smallest_by_coarse_feature_preproc()
    # plot_by_coarse_feature_preproc_subgroup()
    # plot_by_coarse_predictive_feature_preproc_subgroup()
    # plot_by_fine_feature_group_predictive_norm_subgroup()
    # plot_rmt_largest_by_degree()
    # plot_all_by_fine_feature_groups_best_params()
    # plot_all_by_fine_feature_groups_best_params_slice()
    # plot_all_by_fine_feature_groups_best_params()
    # plot_all_by_fine_feature_groups_best_params_best_slice()

    simplefilter("ignore", UserWarning)
    ##########################################################################
    # Supplementary Figures
    ##########################################################################
    # plot_all_by_fine_feature_groups(summary="auroc")
    # plot_largest_by_fine_feature_subgroup(summary="auroc")
    # plot_smallest_by_fine_feature_subgroup(summary="auroc")
    # plot_by_fine_predictive_feature_preproc_subgroup(summary="auroc")
    # plot_by_coarse_feature_group_predictive_classifier_subgroup(summary="auroc")
    # plot_by_fine_feature_group_predictive_classifier_subgroup(summary="auroc")
    # plot_rmt_by_trim(summary="auroc")
    # plot_rmt_by_degree(summary="auroc")
    # plot_all_by_fine_feature_groups_slice(summary="auroc")

    # plot_all_by_fine_feature_groups(summary="acc+")
    # plot_largest_by_fine_feature_subgroup(summary="acc+")
    # plot_smallest_by_fine_feature_subgroup(summary="acc+")
    # plot_by_fine_predictive_feature_preproc_subgroup(summary="acc+")
    # plot_by_coarse_feature_group_predictive_classifier_subgroup(summary="acc+")
    # plot_by_fine_feature_group_predictive_classifier_subgroup(summary="acc+")
    # plot_rmt_by_trim(summary="acc+")
    # plot_rmt_by_degree(summary="acc+")
    # plot_all_by_fine_feature_groups_slice(summary="acc+")


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


if __name__ == "__main__":
    simplefilter(action="ignore", category=PerformanceWarning)
    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000
    # feature_superiority()
    # df_ses = load_combined(drop_ses=False)
    # df.to_json(PROJECT / "EVERYTHING.json")
    # print(f"Saved all combined data to {PROJECT / 'EVERYTHING.json'}")

    # make_kde_plots()
    # sys.exit()

    df = load_combined()
    feature_summary = (
        df[df.subgroup.isin(OVERALL_PREDICTIVE_GROUP_ORDER)]
        .groupby("feature")
        .describe(percentiles=[0.05, 0.5, 0.95])["f1"]
        .sort_values(by=["95%", "feature"], ascending=[False, True])
        .drop(columns=["count"])
        .loc[:, ["mean", "min", "5%", "50%", "95%", "max", "std"]]
        .round(3)
    )
    print(feature_summary.to_markdown(tablefmt="simple"))

    preds = df[df.subgroup.isin(OVERALL_PREDICTIVE_GROUP_ORDER)].copy()
    largest_aurocs = (
        preds.groupby(["subgroup", "fine_feature"])[["auroc"]]
        .quantile(0.95)
        .groupby("subgroup")
        .apply(lambda grp: grp.nlargest(3, columns=["auroc"]))
        .droplevel(1)
    )
    largest_accs = (
        preds.groupby(["subgroup", "fine_feature"])[["acc+"]]
        .quantile(0.95)
        .groupby("subgroup")
        .apply(lambda grp: grp.nlargest(3, columns=["acc+"]))
        .droplevel(1)
    )
    largests = pd.concat([largest_aurocs, largest_accs], axis=1)
    largests = largests.sort_values(
        by=["subgroup", "auroc", "acc+"], ascending=[True, False, False]
    )
    print(largests.round(3).to_latex())

    # plot_unfolded_duloxetine()
    # plot_unfolded_aging()
    simplefilter("ignore", UserWarning)
    # plot_unfolded(UpdatedDataset.Vigilance, group1="vigilant", group2="nonvigilant")
    # plot_unfolded(UpdatedDataset.Older, group1="younger", group2="older")
    # plot_unfolded(
    #     UpdatedDataset.Osteo,
    #     preproc=PreprocLevel.SliceTimeAlign,
    #     group1="duloxetine",
    #     group2="nopain",
    #     degree=9,
    # )
    # plot_observables(
    #     UpdatedDataset.Older,
    #     preproc=PreprocLevel.BrainExtract,
    #     group1="younger",
    #     group2="older",
    #     degree=7,
    # )
    # summary_stats_and_tables()

    # ts = load_tseries()
    # df = load_combined()

    # print_correlations(by=["data", "classifier", "preproc", "feature"])

    # print("\n" * 50)

    # get_overfit_scores()
    # summarize_performance_by_aggregation(metric="auroc", summarizer="median")

    # generate_all_topk_plots()
    sys.exit()

    # df = load_all_renamed()
    # df = df.loc[df.preproc != "minimal"]
    # df = df.loc[df.preproc == "minimal"]
    # naive_describe(df)

    # naive_describe(df)
    # print(f"Summarized {len(df)} 5-fold runs")
