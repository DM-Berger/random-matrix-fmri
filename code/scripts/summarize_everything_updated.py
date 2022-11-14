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
from rmt.summary.plotting.kde import kde_plot, Grouping
from rmt.summary.plotting.features import (
    plot_observables,
    plot_unfolded,
    plot_unfolded_duloxetine,
    plt,
)
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

    def plot_all_by_coarse_feature(metric: Metric = "auroc", show: bool = False) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            bw_adjust=1.2,
            add_lines="vlines",
            suptitle_fmt="Distribution of {metric} by Coarse Feature Group",
            fname_modifier="overall",
            title_clean=[dict(split_at="-")],
            fix_xticks=False,
            w=SPIE_JMI_MAX_WIDTH_INCHES,
            h=5,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.09),
            xlims="all",
            facet_kwargs=dict(ylim=(0.0, 15.0)),
            show=show,
        )

    def plot_all_by_fine_feature_groups(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            bw_adjust=1.2,
            add_lines="vlines",
            suptitle_fmt="Distribution of {metric} by Fine Feature Group",
            fname_modifier="overall",
            title_clean=[dict(split_at="-")],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.08),
            xlims="all",
            facet_kwargs=dict(ylim=(0.0, 15.0)),
            show=show,
        )

    def plot_all_by_fine_feature_groups_best_params(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        rmt = df.loc[df.coarse_feature.isin(["rmt"])]
        rmt = rmt.loc[rmt.trim.isin(["middle", "largest"])]
        rmt = rmt.loc[rmt.deg.isin([3, 9])]
        rmt = rmt.loc[rmt.preproc.isin(["MotionCorrect", "MNIRegister"])]
        rmt = rmt.loc[rmt.slice.isin(["max-10", "mid-25"])]
        idx = rmt.feature.isin(["unfolded"])
        rmt = rmt[idx]
        other = df.loc[df.coarse_feature.isin(["eigs", "tseries"])]
        data = pd.concat([rmt, other], axis=0)

        kde_plot(
            data=data,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            bw_adjust=1.2,
            add_lines="vlines",
            suptitle_fmt="Distribution of {metric} by Fine Feature Group - Best Params",
            fname_modifier="best_params",
            title_clean=[dict(split_at="-")],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.08),
            xlims="all",
            facet_kwargs=dict(ylim=(0.0, 15.0)),
            show=show,
        )

    def plot_largest_by_coarse_feature(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        """Too big / complicated"""

        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(500, metric)
        )
        print("done")
        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.Subgroup,
            row=Grouping.Classifier,
            add_lines=True,
            dashify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt=(
                "Distributions of Largest 500 {metric} for each combination "
                "of Coarse Feature Group, Dataset, and Classifier"
            ),
            fname_modifier="largest",
            title_clean=[
                dict(text="classifier = "),
                dict(text="subgroup = ", split_at="-"),
            ],
            fix_xticks=True,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.09),
            facet_kwargs=dict(ylim=(0.0, 75.0)),
            xlims="largest",
            show=show,
        )

    def plot_largest_by_coarse_feature_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        """THIS IS GOOD. LOOK AT MODES. In only ony case are eigs or rmt mode auroc
        worse than tseries alone, i.e. modally, RMT or eigs are better than tseries.
        """
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(500, metric)
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            add_lines=True,
            dashify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt=(
                "Distributions of Largest 500 {metric} for each combination "
                "of Coarse Feature Group and Dataset"
            ),
            fname_modifier="largest",
            title_clean=[
                dict(text="classifier = "),
                dict(text="subgroup = ", split_at="-"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.09),
            facet_kwargs=dict(sharey=False),
            xlims="largest",
            show=show,
        )

    def plot_largest_by_fine_feature_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nlargest(500, metric)
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            add_lines=True,
            dashify=False,
            thinify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt=(
                "Distributions of Largest 500 {metric} for each combination "
                "of Fine Feature Group and Dataset"
            ),
            fname_modifier="largest",
            title_clean=[
                dict(text="classifier = "),
                dict(text="subgroup = ", split_at="-"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.09),
            facet_kwargs=dict(sharey=False),
            xlims="largest",
            show=show,
        )

    def plot_smallest_by_coarse_feature_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "coarse_feature"]).apply(
            lambda grp: grp.nsmallest(500, metric)
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            add_lines=True,
            dashify=False,
            thinify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt=(
                "Distributions of Smallest 500 {metric} for each combination "
                "of Coarse Feature Group and Dataset"
            ),
            fname_modifier="smallest",
            title_clean=[
                dict(text="classifier = "),
                dict(text="subgroup = ", split_at="-"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.09),
            facet_kwargs=dict(sharey=False),
            xlims="smallest",
            show=show,
        )

    def plot_smallest_by_fine_feature_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "fine_feature"]).apply(
            lambda grp: grp.nsmallest(500, metric)
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            add_lines=True,
            dashify=False,
            thinify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt=(
                "Distributions of Smallest 500 {metric} for each combination "
                "of Fine Feature Group and Dataset"
            ),
            fname_modifier="smallest",
            title_clean=[
                dict(text="classifier = "),
                dict(text="subgroup = ", split_at="-"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
            ),
            legend_pos=(0.79, 0.08),
            facet_kwargs=dict(sharey=False),
            xlims="smallest",
            show=show,
        )

    def plot_all_by_fine_feature_groups_slice(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        data = df.loc[df.coarse_feature.isin(["rmt", "eigs"])]  # sliceables
        stitle = s_title(metric)
        sfname = s_fnmae(metric)

        grid = sbn.displot(
            data=data,
            x=metric,
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
            facet_kws=dict(sharey=False, xlim=s_xlim(metric, kind="all")),
        )
        thinify_lines(grid)
        add_auroc_lines(grid, kind="vline", summary=metric)
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

    def plot_largest_by_coarse_feature_preproc(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "coarse_feature"]).apply(
            lambda grp: grp.nlargest(2000, "auroc")
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Preprocessing,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="Largest 500 {metric} by Preprocessing Degree",
            fname_modifier="largest_predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text=r"BrainExtract \| "),
                dict(text=r"SliceTimeAlign \| "),
                dict(text=r"MotionCorrect \| "),
                dict(text=r"MNIRegister \| "),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.05, 0.75),
            facet_kwargs=dict(sharey=False),
            xlims="largest",
            show=show,
        )

    def plot_smallest_by_coarse_feature_preproc(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "coarse_feature"]).apply(
            lambda grp: grp.nsmallest(500, metric)
        )
        print("done")

        kde_plot(
            data=dfg,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Preprocessing,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="Smallest 500 {metric} by Preprocessing Degree",
            fname_modifier="smallest_predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text=r"BrainExtract \| "),
                dict(text=r"SliceTimeAlign \| "),
                dict(text=r"MotionCorrect \| "),
                dict(text=r"MNIRegister \| "),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.05, 0.75),
            facet_kwargs=dict(sharey=False),
            xlims="smallest",
            show=show,
        )

    def plot_by_coarse_feature_preproc_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.Subgroup,
            row=Grouping.Preprocessing,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Preprocessing Degree",
            fname_modifier="all",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text=r"BrainExtract \| "),
                dict(text=r"SliceTimeAlign \| "),
                dict(text=r"MotionCorrect \| "),
                dict(text=r"MNIRegister \| "),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.90),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_by_coarse_predictive_feature_preproc_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Preprocessing,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Preprocessing Degree",
            fname_modifier="predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text=r"BrainExtract \| "),
                dict(text=r"SliceTimeAlign \| "),
                dict(text=r"MotionCorrect \| "),
                dict(text=r"MNIRegister \| "),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.90),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_by_fine_predictive_feature_preproc_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Preprocessing,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Preprocessing Degree",
            fname_modifier="predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text=r"BrainExtract \| "),
                dict(text=r"SliceTimeAlign \| "),
                dict(text=r"MotionCorrect \| "),
                dict(text=r"MNIRegister \| "),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.82, bottom=0.085, left=0.09, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.80),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_by_fine_feature_group_predictive_norm_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Norm,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Normaliztion",
            fname_modifier="predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
                dict(text=r"norm = .+\|"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.91, bottom=0.085, left=0.09, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.80),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_by_coarse_feature_group_predictive_classifier_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.CoarseFeature,
            col=Grouping.PredictiveSubgroup,
            col_wrap=4,
            add_lines=True,
            dashify=False,
            thinify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} for Predictive Classification Tasks",
            fname_modifier="all_predictive",
            title_clean=[
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.79, 0.08),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_by_fine_feature_group_predictive_classifier_subgroup(
        metric: Metric = "auroc", show: bool = False
    ) -> None:
        kde_plot(
            data=df,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.Subgroup,
            col_wrap=4,
            add_lines=True,
            dashify=False,
            thinify=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} for Predictive Classification Tasks",
            fname_modifier="all_predictive",
            title_clean=[
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.79, 0.08),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_rmt_by_trim(metric: Metric = "auroc", show: bool = False) -> None:
        data = df.loc[
            df.feature.apply(
                lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                and ("eigs" not in s)
            )
        ]
        kde_plot(
            data=data,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Trim,
            add_lines=True,
            dashify=True,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Trimming",
            fname_modifier="predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
                dict(text=r"norm = .+\|"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.91, bottom=0.085, left=0.09, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.80),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_rmt_by_degree(metric: Metric = "auroc", show: bool = False) -> None:
        data = df.loc[
            df.feature.apply(
                lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                and ("eigs" not in s)
            )
        ]
        kde_plot(
            data=data,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Degree,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="{metric} by Degree",
            fname_modifier="predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
                dict(text=r"norm = .+\|"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.91, bottom=0.085, left=0.09, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.80),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    def plot_rmt_largest_by_degree(metric: Metric = "auroc", show: bool = False) -> None:
        data = df.loc[
            df.feature.apply(
                lambda s: (("rigid" in s) or ("levelvar" in s) or ("unfolded" in s))
                and ("eigs" not in s)
            )
        ]
        print("Grouping...", end="", flush=True)
        df_largest = data.groupby(["deg", "feature"]).apply(
            lambda grp: grp.nlargest(2000, "auroc")
        )
        print("done")

        kde_plot(
            data=df_largest,
            metric=metric,
            hue=Grouping.FineFeature,
            col=Grouping.PredictiveSubgroup,
            row=Grouping.Degree,
            add_lines=True,
            dashify=False,
            thinify=True,
            add_row_labels=True,
            bw_adjust=1.2,
            alpha=0.8,
            suptitle_fmt="Largst {metric} by Unfolding Degree",
            fname_modifier="largest_predictive",
            title_clean=[
                dict(text="preproc = "),
                dict(text="subgroup = ", split_at="-"),
                dict(text="task_attend", replace="high"),
                dict(text="task_nonattend", replace="low"),
                dict(text="nonvigilant", replace="low"),
                dict(text="vigilant", replace="high"),
                dict(text="younger", replace="young"),
                dict(text="duloxetine", replace="dlxtn"),
                dict(text=r"norm = .+\|"),
            ],
            fix_xticks=False,
            subplots_adjust=dict(
                top=0.91, bottom=0.085, left=0.09, right=0.955, hspace=0.35, wspace=0.3
            ),
            legend_pos=(0.00, 0.80),
            facet_kwargs=dict(sharey=False),
            xlims="all",
            show=show,
        )

    # silence constant KDE warnings for some largest / smallest plotting cases
    simplefilter("ignore", UserWarning)
    SHOW = False
    for metric in ["auroc", "acc+"]:
        # Everything, too busy sometimes
        plot_all_by_coarse_feature(metric, show=SHOW)
        plot_all_by_fine_feature_groups(metric, show=SHOW)
        plot_all_by_fine_feature_groups_best_params(metric, show=SHOW)

        # Everything, but just predictive tasks
        plot_by_coarse_feature_group_predictive_classifier_subgroup(metric, show=SHOW)
        plot_by_fine_feature_group_predictive_classifier_subgroup(metric, show=SHOW)
        plot_by_coarse_predictive_feature_preproc_subgroup(metric, show=SHOW)
        plot_by_fine_predictive_feature_preproc_subgroup(metric, show=SHOW)

        # Largest and Smallest plots
        plot_largest_by_coarse_feature(metric, show=SHOW)
        plot_largest_by_coarse_feature_subgroup(metric, show=SHOW)
        plot_largest_by_fine_feature_subgroup(metric, show=SHOW)
        plot_smallest_by_coarse_feature_subgroup(metric, show=SHOW)
        plot_smallest_by_fine_feature_subgroup(metric, show=SHOW)

        # Main effects of Analytic Choices
        plot_by_coarse_feature_preproc_subgroup(metric, show=SHOW)
        plot_all_by_fine_feature_groups_slice(metric, show=SHOW)
        plot_rmt_by_trim(metric, show=SHOW)
        plot_rmt_by_degree(metric, show=SHOW)
        plot_by_fine_feature_group_predictive_norm_subgroup(metric, show=SHOW)

        # Main effects of Analytic Choices on Largest / Smallest
        plot_rmt_largest_by_degree(metric, show=SHOW)
        plot_largest_by_coarse_feature_preproc(metric, show=SHOW)
        plot_smallest_by_coarse_feature_preproc(metric, show=SHOW)


if __name__ == "__main__":
    simplefilter(action="ignore", category=PerformanceWarning)
    simplefilter(action="ignore", category=UserWarning)

    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000
    # df_ses = load_combined(drop_ses=False)

    # generate_all_topk_plots()
    make_kde_plots()
    plot_unfolded_duloxetine()
    plot_unfolded(UpdatedDataset.Vigilance, group1="vigilant", group2="nonvigilant")
    plot_unfolded(UpdatedDataset.Older, group1="younger", group2="older")
    plot_unfolded(
        UpdatedDataset.Osteo,
        preproc=PreprocLevel.SliceTimeAlign,
        group1="duloxetine",
        group2="nopain",
        degree=9,
    )
    plot_observables(
        UpdatedDataset.Older,
        preproc=PreprocLevel.BrainExtract,
        group1="younger",
        group2="older",
        degree=7,
    )
    sys.exit()

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

    # naive_describe(df)
    # feature_superiority()
    # summary_stats_and_tables()
    # print_correlations(by=["data", "classifier", "preproc", "feature"])
    # get_overfit_scores()
    # summarize_performance_by_aggregation(metric="auroc", summarizer="median")
