# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from typing import Literal
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Memory
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas import DataFrame
from pandas.errors import PerformanceWarning
from seaborn import FacetGrid
from tqdm import tqdm

from rmt.updated_features import FEATURE_OUTFILES as PATHS
from rmt.visualize import UPDATED_PLOT_OUTDIR as PLOT_OUTDIR
from rmt.visualize import best_rect

PROJECT = ROOT.parent
MEMORY = Memory(PROJECT / "__JOBLIB_CACHE__")
SPIE_OUTDIR = PROJECT / "results/plots/figures/SPIE"
if not SPIE_OUTDIR.exists():
    SPIE_OUTDIR.mkdir(exist_ok=True, parents=True)

SPIE_JMI_MAX_WIDTH_INCHES = 6.75
SPIE_JMI_MAX_COL_WIDTH_INCHES = 3 + 5 / 16
SPIE_MIN_LINE_WEIGHT = 0.5

HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)
DROPS = [
    "acc+",
    "auroc",
    "classifier_GradientBoostingClassifier",
]
BLUE = "#004cc7"
ORNG = "#f68a0e"
GREEN = "#01f91e"
GREY = "#5c5c5c"
BLCK = "#000000"
PURP = "#8e02c5"
RED = "#de0202"
PINK = "#ff6bd8"

ALL_GROUPERS = [
    "data",
    "comparison",
    "classifier",
    "preproc",
    "deg",
    "trim",
    "slice",
    "norm",
]

SUBGROUPERS = [
    ["data"],
    ["data", "comparison"],
    ["data", "comparison", "classifier"],
    ["data", "comparison", "classifier", "preproc"],
    ["data", "comparison", "classifier", "preproc", "deg"],
    ["data", "comparison", "classifier", "preproc", "trim"],
    ["data", "comparison", "classifier", "preproc", "slice"],
    ["data", "comparison", "classifier", "preproc", "deg", "trim"],
    ["data", "comparison", "classifier", "preproc", "deg", "norm"],
    ["data", "comparison", "classifier", "preproc", "deg", "trim", "norm"],
]

FEATURE_GROUP_PALETTE = {
    "tseries loc": RED,
    "tseries scale": PINK,
    "eigs": BLCK,
    "eigs max": ORNG,
    "eigs smooth": GREEN,
    "eigs middle": GREY,
    "rmt + eigs": PURP,
    "rmt only": BLUE,
}

GROSS_FEATURE_PALETTE = {
    "tseries": ORNG,
    "eigs": BLCK,
    "rmt": BLUE,
}

TRIM_PALETTE = {
    "none": BLCK,
    "precision": PURP,
    "largest": BLUE,
    "middle": ORNG,
}

TRIM_ORDER = list(TRIM_PALETTE.keys())

SLICE_ORDER = [
    "all",
    "max-25",
    "max-10",
    "max-05",
    "mid-25",
    "mid-10",
    "mid-05",
    "min-25",
    "min-10",
    "min-05",
]

SUBGROUP_ORDER = [
    "Bilinguality - monolingual v bilingual",
    "Depression - depress v control",
    "Learning - rest v task",
    "Aging - younger v older",
    "Osteo - nopain v duloxetine",
    "Osteo - nopain v pain",
    "Osteo - pain v duloxetine",
    "Parkinsons - ctrl v park",
    "TaskAttention - task_attend v task_nonattend",
    # "TaskAttentionSes1 - task_attend v task_nonattend",
    # "TaskAttentionSes2 - task_attend v task_nonattend",
    "Vigilance - vigilant v nonvigilant",
    # "VigilanceSes1 - vigilant v nonvigilant",
    # "VigilanceSes2 - vigilant v nonvigilant",
    "WeeklyAttention - trait_nonattend v trait_attend",
    # "WeeklyAttentionSes1 - trait_nonattend v trait_attend",
    # "WeeklyAttentionSes2 - trait_attend v trait_nonattend",
]


CLASSIFIER_ORDER = [
    # "RandomForestClassifier",
    # "GradientBoostingClassifier",
    "RF",
    "GBDT",
    "KNN3",
    "KNN5",
    "KNN9",
    "SVC",
]
PREPROC_ORDER = [
    "BrainExtract",
    "SliceTimeAlign",
    "MotionCorrect",
    "MNIRegister",
]


def get_aggregates(subgroupers: list[list[str]]) -> list[list[str]]:
    aggregates = [  # just excludes the column used for grouping and keeps ordering
        [colname for colname in ALL_GROUPERS if colname not in subgrouper]
        for subgrouper in subgroupers
    ]
    return aggregates


AGGREGATES = get_aggregates(SUBGROUPERS)


def topk_outdir(k: int) -> Path:
    outdir = PLOT_OUTDIR / f"top-{k}_plots"
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


def corr_renamer(s: str) -> str:
    if "feature_" in s:
        return s.replace("feature_", "")
    return s


@MEMORY.cache
def load_all_renamed() -> DataFrame:
    df = pd.concat(
        [
            pd.read_json(path)
            for path in tqdm(PATHS.values(), desc="loading")
            if path.exists()
        ],
        axis=0,
        ignore_index=True,
    )
    df.loc[:, "data"] = df["data"].str.replace("Older", "Aging")
    df.loc[:, "feature"] = df["feature"].str.replace("plus", " + ").copy()
    df.loc[:, "feature"] = df["feature"].str.replace("eig ", "eigs ").copy()
    df.loc[:, "feature"] = df["feature"].str.replace("eigenvalues", "eigs").copy()
    df.loc[:, "feature"] = (
        df["feature"].str.replace("eigssmoothed", "eigs_smoothed").copy()
    )
    df.loc[:, "feature"] = df["feature"].str.replace("eigssavgol", "eigs_savgol").copy()
    df.loc[:, "feature"] = df["feature"].str.replace("smoothed", "smooth").copy()
    df.loc[:, "feature"] = (
        df["feature"].str.replace("allfeatures", "eigs + rigidity + levelvar").copy()
    )
    df.loc[:, "feature"] = df["feature"].str.replace("rigidities", "rigidity").copy()
    df.loc[:, "feature"] = df["feature"].str.replace("levelvars", "levelvar").copy()
    df.loc[:, "classifier"] = (
        df["classifier"].str.replace("GradientBoostingClassifier", "GBDT").copy()
    )
    df.loc[:, "classifier"] = (
        df["classifier"].str.replace("RandomForestClassifier", "RF").copy()
    )
    dupes = (df.data == "TaskAttentionSes2") & (
        df.comparison == "task_nonattend v task_attend"
    )
    df = df.loc[~dupes]
    dupes = (df.data == "WeeklyAttentionSes2") & (
        df.comparison == "trait_nonattend v trait_attend"
    )
    df = df.loc[~dupes]
    return df


@MEMORY.cache
def load_tseries() -> DataFrame:
    renames = {
        "Percentile05": "p05",
        "Percentile95": "p95",
        "Median": "med",
        "Range": "rng",
        "RobustRange": "rrng",
        "StdDev": "std",
    }
    df = pd.concat([pd.read_json(path) for path in sorted(PROJECT.glob("tseries*.json"))])
    df.rename(columns={"smooth": "deg"}, inplace=True)
    df["feature"] = df.feature.apply(lambda s: renames[s] if s in renames else s)
    df["feature"] = df.feature.apply(lambda s: f"T-{s.lower()}")
    df["slice"] = "all"
    df["trim"] = "none"
    df.loc[:, "data"] = df["data"].str.replace("Older", "Aging")
    return df


def load_combined() -> DataFrame:
    df = load_all_renamed()
    ts = load_tseries()
    return pd.concat([df, ts], axis=0)


@MEMORY.cache
def get_described(metric: Literal["auroc", "f1"] = "auroc") -> DataFrame:
    df = load_all_renamed()
    return (
        df.groupby(["feature", "data", "comparison", "slice"])
        .describe(percentiles=[0.05, 0.95])
        .rename(columns={"max": "best", "50%": "median"})
        .loc[:, metric]
        .drop(columns=["count"])
    )


@MEMORY.cache
def get_described_w_classifier(metric: Literal["auroc", "f1"] = "auroc") -> DataFrame:
    df = load_all_renamed()
    return (
        df.groupby(["feature", "data", "comparison", "classifier", "slice"])
        .describe(percentiles=[0.05, 0.95])
        .rename(columns={"max": "best", "50%": "median"})
        .loc[:, metric]
        .drop(columns=["count"])
    )


def print_correlations(df: DataFrame, sorter: str) -> None:
    print(
        pd.get_dummies(df.reset_index())
        .corr(method="spearman")
        .round(3)
        .loc[sorter]
        .sort_values(ascending=False)
    )


def is_tseries(s: str) -> bool:
    return "T-" in s


def is_ts_loc(s: str) -> bool:
    locs = ["T-max", "T-mean", "T-med", "T-min", "T-p05", "T-p95"]
    return s in locs


def is_ts_scale(s: str) -> bool:
    scales = ["T-iqr", "T-rng", "T-rrng", "T-std"]
    return s in scales


def is_rmt(s: str) -> bool:
    return ("rig" in s) or ("level" in s) or ("unf" in s)


def is_rmt_plus(s: str) -> bool:
    return is_rmt(s) and ("eig" in s)


def is_rmt_only(s: str) -> bool:
    return is_rmt(s) and ("eigs" not in s)


def is_max(s: str) -> bool:
    return ("max" in s) and (not is_tseries(s))


def is_smoothed(s: str) -> bool:
    return ("savgol" in s) or ("smooth" in s)


def is_eigs_only(s: str) -> bool:
    return s == "eigs" or ("middle" in s)


def feature_grouping(s: str) -> str:
    if is_ts_loc(s):
        return "tseries loc"
    if is_ts_scale(s):
        return "tseries scale"
    if is_smoothed(s):
        return "eigs smooth"
    if is_max(s):
        return "eigs max"
    if "middle" in s:
        return "eigs middle"
    if s == "eigs":
        return "eigs"
    if is_rmt_only(s):
        return "rmt only"
    if is_rmt_plus(s):
        return "rmt + eigs"


def gross_feature_grouping(s: str) -> str:
    if is_ts_loc(s) or is_ts_scale(s):
        return "tseries"
    if is_rmt_only(s) or is_rmt_plus(s):
        return "rmt"
    return "eigs"


def slice_grouping(s: str) -> str:
    if "min" in s:
        return "min"
    if "max" in s:
        return "max"
    if "mid" in s:
        return "mid"
    return s


def make_palette(features: list[str]) -> dict[str, str]:
    palette = {}
    for feature in np.unique(features):
        if is_rmt_plus(feature):
            palette[feature] = PURP
        elif is_ts_loc(feature):
            palette[feature] = RED
        elif is_ts_scale(feature):
            palette[feature] = PINK
        elif is_rmt_only(feature):
            palette[feature] = BLUE
        elif is_max(feature):
            palette[feature] = ORNG
        elif is_smoothed(feature):
            palette[feature] = GREY
        else:
            palette[feature] = BLCK
    return palette


def get_feature_ordering(features: list[str]) -> list[str]:
    ts_loc = sorted(filter(is_ts_loc, features))
    ts_scale = sorted(filter(is_ts_scale, features))
    rmt_only = sorted(filter(is_rmt_only, features))
    rmt_plus = sorted(filter(is_rmt_plus, features))
    eigs_smooth = sorted(filter(lambda s: is_smoothed(s), features))
    eigs_max = sorted(filter(is_max, features))
    eigs_only = sorted(filter(lambda s: is_eigs_only(s), features))
    ordering = (
        ts_loc + ts_scale + eigs_smooth + eigs_max + eigs_only + rmt_only + rmt_plus
    )
    return ordering


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


def resize_fig() -> None:
    fig = plt.gcf()
    fig.set_size_inches(w=40, h=26)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, hspace=0.2)


def summarize_performance_by_aggregation(
    metric: Literal["auroc", "f1", "acc+"], summarizer: Literal["median", "best"]
) -> None:
    ax: Axes
    fig: Figure

    sbn.set_style("darkgrid")
    # df = load_all_renamed()
    df = load_combined()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["feature_group"] = df["feature"].apply(feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)

    df = df.loc[~df.data.str.contains("Reflect")]
    df = df.loc[~df.data.str.contains("Ses")]

    # cleaner view
    # """
    sbn.displot(
        data=df,
        x="auroc",
        kind="kde",
        hue="feature_group",
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
        hue="feature_group",
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
        hue="feature_group",
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
        hue="feature_group",
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
        hue="feature_group",
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
        hue="feature_group",
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
        y="feature_group",
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
        x="feature_group",
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
    # """

    return
    """
    # The more levels we include, the less we "generalize" our claims
    for grouper, aggregates in zip(SUBGROUPERS, AGGREGATES):
        min_summarized_per_feature = np.unique(df.groupby(grouper + ["feature"]).count())[
            0
        ]
        grouped = df.groupby(grouper)
        # if you do:
        #   grouped.hist()
        # here, you get a plot for each sub-frame induced by "grouper"
        # I think if you reset_index, or just leave it as is, seaborn is also
        # going to be able to do a decent job here
        if summarizer == "median":
            summary = df.groupby(grouper + ["feature"]).median(numeric_only=True)

        else:
            summary = df.groupby(grouper + ["feature"]).max(numeric_only=True)

        summary.hist(color="black")
        plt.gcf().suptitle(f"Grouping by {grouper}\n(aggregating across: {aggregates})")
        plt.show()

        print()
        # bests = (
        #     summary.reset_index()
        #     .groupby(grouper)
        #     .apply(lambda g: g.nlargest(k, columns="auroc"))
        # )
    """


def plot_topk_features_by_aggregation(sorter: str, k: int = 5) -> None:
    # df = load_all_renamed()
    df = load_combined()
    # The more levels we include, the less we "generalize" our claims
    for grouper, aggregates in zip(SUBGROUPERS, AGGREGATES):
        min_summarized_per_feature = np.unique(df.groupby(grouper + ["feature"]).count())[
            0
        ]
        if sorter == "median":
            summary = df.groupby(grouper + ["feature"]).median(numeric_only=True)

        else:
            summary = df.groupby(grouper + ["feature"]).max(numeric_only=True)
        bests = (
            summary.reset_index()
            .groupby(grouper)
            .apply(lambda g: g.nlargest(k, columns="auroc"))
        )
        best_feats = bests["feature"]
        counts = best_feats.value_counts()
        sbn.set_style("darkgrid")
        fig, ax = plt.subplots()
        sbn.countplot(
            y=best_feats,
            order=counts.index,
            color="black",
            palette=make_palette(counts.index),
            ax=ax,
        )
        Sorter = sorter.capitalize() if sorter == "median" else "Max"
        suptitle = f"Total Number of Instances where Feature Yields One of the Top-{k} {Sorter} AUROCs"
        title = f"Summarizing / Grouping at level of: {grouper}"
        if len(aggregates) > 0:
            title += f"\n(i.e. expected {sorter} performance across all variations of choice of {aggregates})"
        title += (
            f"\n[Number of 5-fold runs summarized by {Sorter} "
            f"per feature grouping: {min_summarized_per_feature}+]"
        )
        ax.set_title(title, fontsize=10)
        make_legend(fig, position=(0.825, 0.15))
        fig.suptitle(suptitle, fontsize=12)
        fig.set_size_inches(w=16, h=6)
        fig.tight_layout()
        outdir = topk_outdir(k)
        fname = f"{sorter}_" + "_".join([g[:4] for g in grouper]) + f"_top{k}.png"
        outfile = outdir / fname
        fig.savefig(outfile, dpi=300)  # type: ignore
        print(f"Saved top-{k} plot to {outfile}")
        plt.close()


def plot_topk_features_by_preproc(sorter: str, k: int = 5) -> None:
    # df = load_all_renamed()
    df = load_combined()
    # The more levels we include, the less we "generalize" our claims
    groupers = [grouper for grouper in SUBGROUPERS if "preproc" not in grouper]
    aggregates = get_aggregates(groupers)
    for agg in aggregates:  # since we split the visuals to cover this
        if "preproc" in agg:
            agg.remove("preproc")
    df_nopre = df.loc[df.preproc == "minimal"]
    df_pre = df.loc[df.preproc != "minimal"]
    df_brain = df.loc[df.preproc == "BrainExtract"]
    df_slice = df.loc[df.preproc == "SliceTimeAlign"]
    df_motion = df.loc[df.preproc == "MotionCorrect"]
    df_reg = df.loc[df.preproc == "MNIRegister"]
    preproc_dfs = [df_brain, df_slice, df_motion, df_reg]
    preproc_labels = [
        "Brain Extract Only",
        "Slicetime Correct",
        "Slicetime + Motion Correct",
        "All Correction + Register",
    ]

    unq = df.feature.unique()
    ordering = get_feature_ordering(unq)
    palette = make_palette(unq)

    for grouper, aggregate in zip(groupers, aggregates):
        min_summarized_per_feature = np.unique(df.groupby(grouper + ["feature"]).count())[
            0
        ]
        if sorter == "median":
            summaries = [
                df_.groupby(grouper + ["feature"]).median(numeric_only=True)
                for df_ in preproc_dfs
            ]
            # summary_pre = df_pre.groupby(grouper + ["feature"]).median(numeric_only=True)
            # summary_nopre = df_nopre.groupby(grouper + ["feature"]).median(
            #     numeric_only=True
            # )

        else:
            summaries = [
                df_.groupby(grouper + ["feature"]).max(numeric_only=True)
                for df_ in preproc_dfs
            ]
            # summary_pre = df_pre.groupby(grouper + ["feature"]).max(numeric_only=True)
            # summary_nopre = df_nopre.groupby(grouper + ["feature"]).max(numeric_only=True)
        bests = [
            (
                summary.reset_index()
                .groupby(grouper, group_keys=False)
                .apply(lambda g: g.nlargest(k, columns="auroc"))
            )
            for summary in summaries
        ]
        # bests_pre = (
        #     summary_pre.reset_index()
        #     .groupby(grouper)
        #     .apply(lambda g: g.nlargest(k, columns="auroc"))
        # )
        # bests_nopre = (
        #     summary_nopre.reset_index()
        #     .groupby(grouper)
        #     .apply(lambda g: g.nlargest(k, columns="auroc"))
        # )

        best_feats = [best["feature"] for best in bests]
        # best_feats_pre = bests_pre["feature"]
        # best_feats_nopre = bests_nopre["feature"]

        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(nrows=len(preproc_dfs), sharex=True, sharey=True)
        ax: Axes
        for i, ax in enumerate(axes):
            sbn.countplot(
                y=best_feats[i],
                order=ordering,
                color="black",
                palette=palette,
                ax=ax,
            )
        Sorter = sorter.capitalize() if sorter == "median" else "Max"
        suptitle = f"Total Number of Instances where Feature Yields One of the Top-{k} {Sorter} AUROCs"
        title = f"Summarizing / Grouping at level of: {grouper}"

        if len(aggregate) > 0:
            title += f"\n(i.e. expected {sorter} performance across all variations of choices of {aggregate})"

        title += (
            f"\n[Number of 5-fold runs summarized by {Sorter} "
            f"per feature grouping: {min_summarized_per_feature}+]"
        )

        for i, ax in enumerate(axes):
            ax.set_title(title if i == 0 else "", fontsize=10)
            ax.set_ylabel(preproc_labels[i], fontsize=14)

        make_legend(fig, position=(0.87, 0.05))
        fig.suptitle(suptitle, fontsize=12)
        fig.set_size_inches(w=16, h=12)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.9, bottom=0.049, left=0.139, right=0.991, hspace=0.18, wspace=0.2
        )
        outdir = topk_outdir(k) / "preproc_compare"
        outdir.mkdir(exist_ok=True, parents=True)
        fname = (
            f"{sorter}_"
            + "_".join([g[:4] for g in grouper])
            + f"_top{k}_preproc-compare.png"
        )
        outfile = outdir / fname
        fig.savefig(outfile, dpi=300)  # type: ignore
        print(f"Saved top-{k} plot to {outfile}")
        plt.close()


def plot_topk_features_by_grouping(
    sorter: str,
    k: int,
    by: Literal["data", "classifier"],
    position: str | tuple[float, float],
) -> None:
    # df = load_all_renamed()
    df = load_combined()

    fine_grouper = ["data", "comparison", "classifier", "preproc", "deg", "norm"]
    if sorter == "best":
        bestk = df.groupby(fine_grouper).apply(lambda g: g.nlargest(k, columns="auroc"))
    else:
        bestk = (
            df.groupby(fine_grouper + ["feature"])
            .median(numeric_only=True)
            .reset_index()
            .groupby(fine_grouper)
            .apply(lambda g: g.nlargest(5, columns="auroc"))
        )
    if by == "classifier":
        bestk = bestk.droplevel(["data", "comparison"])

    features = bestk.loc[:, "feature"].unique().tolist()
    order = get_feature_ordering(features)
    palette = make_palette(features)

    sbn.set_style("darkgrid")
    instances = df[by].unique()
    nrows, ncols = best_rect(len(instances))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)

    for i, instance in enumerate(instances):
        ax: Axes = axes.flat[i]  # type: ignore
        counts = bestk.loc[instance, "feature"]
        sbn.countplot(y=counts, order=order, color="black", palette=palette, ax=ax)
        ax.set_title(f"{str(instance)}")
    for i in range(len(instances), len(axes.flat)):  # remove dangling axes
        fig.delaxes(axes.flat[i])
    suptitle = f"Overall Frequency of Feature Producing one of Top {k} {sorter.capitalize()} AUROCs"

    fig.suptitle(suptitle)
    make_legend(fig, position)
    fig.set_size_inches(w=16, h=16)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.93, bottom=0.056, left=0.126, right=0.988, hspace=0.25, wspace=0.123
    )
    outdir = topk_outdir(k)
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / f"{sorter}_AUROC_by_{by}.png"
    fig.savefig(outfile, dpi=300)  # type: ignore
    print(f"Saved plot to {outfile}")
    plt.close()


def plot_feature_counts_grouped(
    bests3: DataFrame, sorter: str, by: Literal["data", "classifier"]
) -> None:
    # df = load_all_renamed()
    df = load_combined()

    df = bests3.reset_index()
    order = df["feature"].unique()
    rmt_special = sorted(filter(is_rmt, order))
    eigs_only = sorted(filter(lambda s: not is_rmt(s), order))
    order = eigs_only + rmt_special
    palette = make_palette(order)
    drop = "classifier" if by == "data" else "data"
    df = df.drop(columns=drop)
    groups = df[by].unique().tolist()
    nrows, ncols = best_rect(len(groups))
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    ax: Axes
    for group, ax in zip(groups, axes.flat):
        bests = df.loc[df[by] == group, "feature"]
        sbn.countplot(y=bests, order=order, palette=palette, color="black", ax=ax)
        ax.set_title(str(group))
    make_legend(fig)
    fig.suptitle(
        f"Frequency of Feature Producing one of Top 3 {sorter.capitalize()} AUROCs by {by.capitalize()}"
    )
    fig.set_size_inches(w=16, h=16)
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.92, bottom=0.05, left=0.119, right=0.991, hspace=0.175, wspace=0.107
    )
    outdir = PLOT_OUTDIR / "top3_counts"
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / f"{sorter}_AUROC_by_{by}.png"
    fig.savefig(outfile, dpi=300)
    print(f"Saved plot to {outfile}")
    plt.close()


def auroc_correlations(
    df: DataFrame, subset: Literal["all", "features"], predictive_only: bool
) -> DataFrame:
    dummies = pd.get_dummies(df)
    if predictive_only:
        dummies = dummies.loc[dummies["acc+"] > 0.0]
    corrs: DataFrame = (
        dummies.corr(method="spearman").loc["auroc"].drop(index=DROPS)  # type: ignore
    )
    corrs = corrs.filter(like="feature_") if subset == "all" else corrs

    if subset == "all":
        print(f"{HEADER}Spearman correlations with AUROC{FOOTER}")
    else:
        print(
            f"{HEADER}Correlation (Spearman) between feature inclusion and AUROC{FOOTER}"
        )
    result: DataFrame = corrs.rename(index=corr_renamer).sort_values(
        ascending=False
    )  # type: ignore
    print(result)
    return result


def feature_aurocs(df: DataFrame, sorter: str = "best") -> DataFrame:
    if sorter == "best":
        print(f"{HEADER}Feature best AUROCs overall:{FOOTER}")
    elif sorter == "median":
        print(f"{HEADER}Feature mean and median AUROCs overall (median-sorted):{FOOTER}")
    grouped = (
        df.groupby("feature")
        .describe(percentiles=[0.05, 0.95])
        .rename(columns={"max": "best", "50%": "median"})
        .loc[:, "auroc"]
        .drop(columns="count")
        .sort_values(by=sorter, ascending=False)  # type: ignore
        .round(3)
    )
    if sorter == "best":
        ordering = ["best", "median", "mean", "std", "5%", "95%"]
    elif sorter == "median":
        ordering = ["median", "best", "mean", "std", "5%", "95%"]
    else:
        raise NotImplementedError()
    result = grouped.loc[:, ordering]
    print(result)
    return result


# Pandas multi-indexes are hot garbage, nothing works in any sensible kind of way.
# Better to just write your own looping code (as usual), OR, give each grouped DF
# row an index, use group operations (e.g. .loc[...].max()) to get ids, and then
# subset by ids, and group again (lol). But basically better to ungroup and do
# manual table creation
def feature_dataset_aurocs(sorter: str = "best") -> DataFrame:

    desc = get_described(metric="auroc").loc[:, sorter].reset_index()
    bests = (
        (
            desc.sort_values(
                by=["data", "comparison", "feature", sorter], ascending=False
            )
            .groupby(["data", "comparison", "feature"])
            .apply(lambda grp: grp.nlargest(1, columns=sorter))
        )
        .loc[:, ["slice", sorter]]
        .reset_index()
        .drop(columns="level_3")
        .sort_values(by=["data", "comparison", sorter])
        .groupby(["data", "comparison", "feature"], group_keys=True)
        .max()
        .sort_values(by=["data", "comparison", sorter], ascending=False)
    )

    summary = sorter.capitalize()
    groupers = "feature, dataset"
    # CORRELATIONS
    print(f"{HEADER}Correlations of {summary} AUROCs by {groupers}:{FOOTER}")
    print_correlations(bests, sorter)

    print(f"{HEADER}Top 3 {summary} AUROCs by {groupers}:{FOOTER}")
    bests3 = (
        bests.reset_index()
        .groupby(["data", "comparison"], group_keys=True)
        .apply(lambda grp: grp.nlargest(3, columns=[sorter]))
        .loc[:, ["feature", "slice", sorter]]
    )
    print(bests3)
    plot_topk_features_by_aggregation(bests3, sorter)

    # CORRELATIONS
    print(f"{HEADER}Correlations of Top 3 {summary} AUROCs by {groupers}:{FOOTER}")
    print_correlations(bests3, sorter)
    return bests3


def feature_dataset_classifier_aurocs(sorter: str = "best") -> DataFrame:
    desc = get_described_w_classifier(metric="auroc").loc[:, sorter].reset_index()
    bests = (
        (
            desc.sort_values(
                by=["data", "comparison", "feature", "classifier", sorter],
                ascending=False,
            )
            .groupby(["data", "comparison", "feature", "classifier"])
            .apply(lambda grp: grp.nlargest(1, columns=sorter))
        )
        .loc[:, ["slice", sorter]]
        .reset_index()
        .sort_values(by=["data", "comparison", "classifier", sorter])
        .groupby(["data", "comparison", "feature", "classifier"], group_keys=True)
        .max()
        .sort_values(by=["data", "comparison", "classifier", sorter], ascending=False)
    )

    # CORRELATIONS
    summary = sorter.capitalize()
    groupers = "feature, dataset, classifier"
    print(f"{HEADER}Correlations of {summary} AUROCs by {groupers}:{FOOTER}")
    print_correlations(bests, sorter)

    print(f"{HEADER}Top 3 {summary} AUROCs by {groupers}:{FOOTER}")
    bests3 = (
        bests.reset_index()
        .groupby(["data", "comparison", "classifier"], group_keys=True)
        .apply(lambda grp: grp.nlargest(3, columns=[sorter]))
        .loc[:, ["feature", "slice", sorter]]
    )
    print(bests3)
    plot_feature_counts_grouped(bests3, sorter, by="data")
    plot_feature_counts_grouped(bests3, sorter, by="classifier")

    # CORRELATIONS
    print(f"{HEADER}Correlations of Top 3 {summary} AUROCs by {groupers}:{FOOTER}")
    print_correlations(bests3, sorter)
    return bests3


def naive_describe(df: DataFrame) -> None:
    # pd.options.display. = True
    pd.options.display.max_info_rows = None
    pd.options.display.max_rows = None
    pd.options.display.expand_frame_repr = True

    # df = df.loc[df.classifier != "SVC"].copy()

    # auroc_correlations(df, subset="all", predictive_only=False)
    # auroc_correlations(df, subset="features", predictive_only=False)
    # auroc_correlations(df, subset="all", predictive_only=True)
    # auroc_correlations(df, subset="features", predictive_only=True)

    # feature_aurocs(df, sorter="best")
    # feature_aurocs(df, sorter="median")

    # FOR FAST TESTING
    # df = df.loc[df.data == "Osteo"]

    feature_dataset_aurocs(sorter="best")
    feature_dataset_classifier_aurocs(sorter="best")
    feature_dataset_aurocs(sorter="median")
    feature_dataset_classifier_aurocs(sorter="median")

    # osteo = (
    #     df.loc[df.data == "Osteo"]
    #     .loc[df.comparison == "nopain v duloxetine"]
    #     .drop(columns=["data", "comparison"])
    # )
    # print(
    #     f"{HEADER}Summary stats of AUROCs for 'nopain v duloxetine' comparison (max-sorted):{FOOTER}"
    # )
    # print(
    #     osteo.groupby(["feature"])
    #     .describe()
    #     .loc[:, "auroc"]
    #     .round(3)
    #     .rename(columns={"50%": "median"})  # type: ignore
    #     .loc[:, ["mean", "median", "min", "max", "std"]]
    #     .sort_values(by=["max", "median"], ascending=False)
    # )


def generate_all_topk_plots() -> None:
    K = 5
    plot_topk_features_by_aggregation(sorter="median", k=K)
    plot_topk_features_by_aggregation(sorter="best", k=K)
    position1 = (0.82, 0.11)
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="data",
        position=position1,
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="data",
        position=position1,
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="classifier",
        position=(0.83, 0.07),
    )
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="classifier",
        position=(0.83, 0.28),
    )
    plot_topk_features_by_preproc(sorter="median", k=5)
    plot_topk_features_by_preproc(sorter="best", k=5)


def get_overfit_scores() -> None:
    # df = load_all_renamed()
    df = load_combined()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["feature_group"] = df["feature"].apply(feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)
    df["is_overfit"] = (df["auroc"] < 0.5).astype(int)
    df["overfit_score"] = df["auroc"].apply(lambda auroc: 2 * (auroc - 0.5))
    classifiers = pd.get_dummies(df["classifier"])
    df_cls = df.drop(columns="classifier")
    df_cls = pd.concat([df_cls, classifiers], axis=1).drop(
        columns=[
            "acc+",
            "acc",
            "f1",
            "is_overfit",
            "norm",
            "auroc",
            "data",
            "comparison",
            "feature",
            "slice",
        ]
    )
    p95 = df.groupby(
        ["subgroup", "feature_group", "classifier", "preproc", "trim", "slice_group"]
    ).quantile(0.95)
    good = p95["overfit_score"].loc[p95["overfit_score"] < 0].reset_index()
    print("")
    print(good.feature_group.value_counts())

    corrs = (
        df_cls.groupby("subgroup")
        .corr("spearman", numeric_only=True)["overfit_score"]  # type: ignore
        .reset_index()
        .rename(columns={"level_1": "classifier", "overfit_score": "correlation"})
    )
    corrs = corrs[corrs.classifier != "overfit_score"]

    grid: FacetGrid = sbn.catplot(
        data=corrs,
        y="correlation",
        x="classifier",
        order=CLASSIFIER_ORDER,
        col="subgroup",
        col_order=SUBGROUP_ORDER,
        col_wrap=3,
        kind="bar",
    )
    fig = plt.gcf()
    fig.suptitle("Correlation with Overfitting (based on AUROC)")

    plt.show()

    print(
        df.groupby(["subgroup", "classifier"])
        .describe()["is_overfit"]
        .drop(columns=["std", "min", "max", "25%", "75%", "count"])
    )


def print_correlations(by: list[str]) -> None:
    # df = load_all_renamed()
    df = load_combined()
    df.drop(columns=["acc+", "acc", "f1"], inplace=True)
    corrs = df.groupby(by).apply(
        lambda grp: pd.get_dummies(grp.drop(columns=["auroc"] + by)).corrwith(
            grp["auroc"], method="spearman"
        )
    )
    print(corrs)
    print(corrs.reset_index().groupby(by).describe())

    # spearman = dummies.corrwith(df.auroc, method="spearman")
    # spearman.name = "spearman"
    # pearson = dummies.corrwith(df.auroc, method="pearson")
    # pearson.name = "pearson"
    # corrs = pd.concat([spearman, pearson]).sort_values(by="spearman", ascending=False)
    # print(corrs)


def clean_titles(
    grid: FacetGrid, text: str = "subgroup = ", split_at: Literal["-", "|"] | None = None
) -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        axtitle = ax.get_title()
        if split_at is not None:
            ax.set_title(
                axtitle.replace(text, "").replace(f" {split_at} ", "\n"), fontsize=8
            )
        else:
            ax.set_title(axtitle.replace(text, ""), fontsize=8)


def rotate_labels(grid: FacetGrid, axis: Literal["x", "y"] = "x") -> None:
    fig: Figure = grid.fig
    for ax in fig.axes:
        if axis == "x":
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        else:
            plt.setp(ax.get_yticklabels(), rotation=40, ha="right")


def add_auroc_lines(grid: FacetGrid, kind: Literal["vline", "hline"]) -> None:
    fig: Figure
    fig = grid.fig
    for i, ax in enumerate(fig.axes):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        if kind == "vline":
            ax.vlines(
                x=0.5,
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
                y=0.5,
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


def savefig(fig: Figure, filename: str) -> None:
    print("Saving...", end="", flush=True)
    outfile = SPIE_OUTDIR / filename
    fig.savefig(outfile, dpi=600)
    print(f" saved figure to {outfile}")
    plt.close()


def make_kde_plots() -> None:
    ax: Axes
    fig: Figure
    sbn.set_style("ticks")

    print("Loading data...", end="", flush=True)
    print(" done")
    df = load_combined()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["feature_group"] = df["feature"].apply(feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)
    df["gross_feature"] = df["feature"].apply(gross_feature_grouping)
    df = df.loc[~df.data.str.contains("Ses")]

    def plot_by_gross_feature() -> None:
        ax: Axes
        fig: Figure
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df,
            x="auroc",
            kind="kde",
            hue="gross_feature",
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
            "Distribution of AUROCs by Feature Group",
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
        savefig(fig, "gross_feature_overall_by_subgroup.png")

    def plot_largest_by_gross_feature() -> None:
        """Too big / complicated"""
        dfg = df.groupby(["subgroup", "gross_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="gross_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            # hue="feature_group",
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
            "Distributions of Largest 500 AUROCs for each combination of Gross Feature Group, Dataset, and Classifier",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.992, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "gross_feature_largest_by_subgroup_data.png")

    def plot_largest_by_gross_feature_subgroup() -> None:
        """THIS IS GOOD. LOOK AT MODES. In only ony case are eigs or rmt mode auroc
        worse than tseries alone, i.e. modally, RMT or eigs are better than tseries.
        """
        dfg = df.groupby(["subgroup", "gross_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="gross_feature",
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
            "Distributions of Largest 500 AUROCs for each Combination of Gross Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "gross_feature_largest_by_subgroup.png")

    def plot_smallest_by_gross_feature_subgroup() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["subgroup", "gross_feature"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="gross_feature",
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
            "Distributions of Smallest 500 AUROCs for each Combination of Gross Feature Group and Dataset",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.08, left=0.04, right=0.98, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.79, 0.09))
        savefig(fig, "gross_feature_smallest_by_subgroup.png")

    def plot_largest_by_feature_groups() -> None:
        dfg = df.groupby(["subgroup", "feature_group"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="feature_group",
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
        fig.suptitle("Distribution of Largest 500 AUROCs for each Feature Group")
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=10)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.03, right=0.985, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.72, 0.08))
        savefig(fig, "feature_group_largests_by_subgroup.png")

    def plot_smallest_by_feature_groups() -> None:
        dfg = df.groupby(["subgroup", "feature_group"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="feature_group",
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
        fig.suptitle("Distribution of Smallest 500 AUROCs for each Feature Group")
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=10)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.05, right=0.985, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.76, 0.11))
        savefig(fig, "feature_group_smallest_by_subgroup.png")

    def plot_all_by_feature_groups() -> None:
        grid = sbn.displot(
            data=df,
            x="auroc",
            kind="kde",
            hue="feature_group",
            hue_order=list(FEATURE_GROUP_PALETTE.keys()),
            palette=FEATURE_GROUP_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(sharey=False, xlim=(0.1, 0.9)),
        )
        add_auroc_lines(grid, kind="vline")
        despine(grid)
        clean_titles(grid, "classifier = ")
        clean_titles(grid, "subgroup = ", split_at="-")
        fig = grid.fig
        fig.suptitle("Overall Distribution of AUROCs for each Feature Group")
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.92, bottom=0.05, left=0.03, right=0.995, hspace=0.3, wspace=0.091
        )
        sbn.move_legend(grid, loc=(0.76, 0.11))
        savefig(fig, "all_by_feature_groups.png")

    def plot_largest_by_gross_feature_preproc() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "gross_feature"]).apply(
            lambda grp: grp.nlargest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="gross_feature",
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
        for ax in fig.axes:
            ax.set_ylabel("")
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
        savefig(fig, "gross_feature_largest_by_preproc.png")

    def plot_smallest_by_gross_feature_preproc() -> None:
        print("Grouping...", end="", flush=True)
        dfg = df.groupby(["preproc", "gross_feature"]).apply(
            lambda grp: grp.nsmallest(500, "auroc")
        )
        print("done")
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=dfg,
            x="auroc",
            kind="kde",
            hue="gross_feature",
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
        fig.set_size_inches(w=SPIE_JMI_MAX_COL_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.05, 0.75))
        for ax in fig.axes:
            ax.set_ylabel("")
        savefig(fig, "gross_feature_smallest_by_preproc.png")

    def plot_by_gross_feature_preproc() -> None:
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df,
            x="auroc",
            kind="kde",
            hue="gross_feature",
            hue_order=list(GROSS_FEATURE_PALETTE.keys()),
            palette=GROSS_FEATURE_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            row="preproc",
            row_order=PREPROC_ORDER,
            # col_wrap=4,
            bw_adjust=1.2,
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
        fig.set_size_inches(w=SPIE_JMI_MAX_COL_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.055, right=0.955, hspace=0.35, wspace=0.086
        )
        sbn.move_legend(grid, loc=(0.05, 0.75))
        for ax in fig.axes:
            ax.set_ylabel("")
        savefig(fig, "gross_feature_smallest_by_preproc.png")

    def plot_rmt_by_trim() -> None:
        print("Plotting...", end="", flush=True)
        grid = sbn.displot(
            data=df.loc[
                df.feature.apply(
                    lambda s: (("rigid" in s) or ("levelvar" in s)) and ("eigs" not in s)
                )
            ],
            x="auroc",
            kind="kde",
            hue="trim",
            hue_order=TRIM_ORDER,
            palette=TRIM_PALETTE,
            fill=False,
            common_norm=False,
            col="subgroup",
            col_order=SUBGROUP_ORDER,
            # row="feature",
            # row_order=FEATUR,
            col_wrap=4,
            bw_adjust=1.2,
            alpha=0.8,
            facet_kws=dict(xlim=(0.0, 1.0), sharey=False),
        )
        print("done")
        # clean_titles(grid, " = ", split_at="|")
        clean_titles(grid, "subgroup = ", split_at="-")
        clean_titles(grid, "feature = ", split_at="|")
        despine(grid)
        dashify_trims(grid)
        add_auroc_lines(grid, "vline")
        fig = grid.fig
        fig.suptitle(
            "Eigenfeature AUROCs by Trimming",
            fontsize=10,
        )
        fig.set_size_inches(w=SPIE_JMI_MAX_WIDTH_INCHES, h=5)
        fig.tight_layout()
        fig.subplots_adjust(
            top=0.87, bottom=0.085, left=0.04, right=0.96, hspace=0.35, wspace=0.22
        )
        sbn.move_legend(grid, loc=(0.77, 0.07))
        for ax in fig.axes:
            ax.set_ylabel("")
        savefig(fig, "eigenfeature_auroc_by_trim.png")

    # these are not great
    # plot_overall()
    # plot_largest_by_gross_feature()

    # plot_by_gross_feature()
    # plot_largest_by_gross_feature_subgroup()
    # plot_smallest_by_gross_feature_subgroup()
    # plot_largest_by_feature_groups()
    # plot_smallest_by_feature_groups()
    # plot_all_by_feature_groups()

    plot_by_gross_feature_preproc()
    plot_largest_by_gross_feature_preproc()
    plot_rmt_by_trim()


    return


if __name__ == "__main__":
    simplefilter(action="ignore", category=PerformanceWarning)
    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000
    # df = load_all_renamed()
    # df.to_json(PROJECT / "EVERYTHING.json")
    # print(f"Saved all combined data to {PROJECT / 'EVERYTHING.json'}")

    make_kde_plots()

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
