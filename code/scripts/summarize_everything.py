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
from matplotlib.patches import Patch
from pandas import DataFrame
from pandas.errors import PerformanceWarning
from tqdm import tqdm

from rmt.features import FEATURE_OUTFILES as PATHS
from rmt.visualize import PLOT_OUTDIR, best_rect

PROJECT = ROOT.parent
MEMORY = Memory(PROJECT / "__JOBLIB_CACHE__")

HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)
DROPS = [
    "acc+",
    "auroc",
    "classifier_GradientBoostingClassifier",
]
BLUE = "#004cc7"
ORNG = "#f68a0e"
GREY = "#5c5c5c"
BLCK = "#000000"
PURP = "#8e02c5"

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
def load_all_renamed(remove_reflect: bool = False) -> DataFrame:
    df = pd.concat(
        [pd.read_json(path) for path in tqdm(PATHS.values(), desc="loading")],
        axis=0,
        ignore_index=True,
    )
    if remove_reflect:
        print("\n\nRemoving meaningless 'REFLECT' data:\n")
        non_reflects = df.data.apply(lambda s: "Reflect" not in s)
        df = df.loc[non_reflects]
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
    return df


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


def is_rmt(s: str) -> bool:
    return ("rig" in s) or ("level" in s) or ("unf" in s)


def is_rmt_plus(s: str) -> bool:
    return is_rmt(s) and ("eig" in s)


def is_rmt_only(s: str) -> bool:
    return is_rmt(s) and ("eigs" not in s)


def is_max(s: str) -> bool:
    return "max" in s


def is_smoothed(s: str) -> bool:
    return ("savgol" in s) or ("smooth" in s)


def is_eigs_only(s: str) -> bool:
    return s == "eigs" or ("middle" in s)


def feature_grouping(s: str) -> str:
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


def make_palette(features: list[str]) -> dict[str, str]:
    palette = {}
    for feature in np.unique(features):
        if is_rmt_plus(feature):
            palette[feature] = PURP
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
    rmt_only = sorted(filter(is_rmt_only, features))
    rmt_plus = sorted(filter(is_rmt_plus, features))
    eigs_smooth = sorted(filter(lambda s: is_smoothed(s), features))
    eigs_max = sorted(filter(lambda s: "max" in s, features))
    eigs_only = sorted(filter(lambda s: is_eigs_only(s), features))
    ordering = eigs_smooth + eigs_max + eigs_only + rmt_only + rmt_plus
    return ordering


def make_legend(fig: Figure, position: str | tuple[float, float] = "upper right") -> None:
    patches = [
        Patch(facecolor=GREY, edgecolor="white", label="smoothed eigenvalues feature"),
        Patch(facecolor=ORNG, edgecolor="white", label="max eigenvalues feature"),
        Patch(facecolor=BLCK, edgecolor="white", label="eigenvalues only feature"),
        Patch(facecolor=BLUE, edgecolor="white", label="RMT-only feature"),
        Patch(facecolor=PURP, edgecolor="white", label="RMT + eigenvalues feature"),
    ]
    fig.legend(handles=patches, loc=position)


def summarize_performance_by_aggregation(
    metric: Literal["auroc", "f1", "acc+"], summarizer: Literal["median", "best"]
) -> None:
    def resize_fig() -> None:
        fig = plt.gcf()
        fig.set_size_inches(w=40, h=26)
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, hspace=0.2)

    sbn.set_style("darkgrid")
    df = load_all_renamed()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["feature_group"] = df["feature"].apply(feature_grouping)

    df = df.loc[~df.data.str.contains("Reflect")]
    df = df.loc[~df.data.str.contains("Ses")]

    # not bad view of features overall
    sbn.displot(
        data=df,
        x="auroc",
        hue="feature_group",
        col="subgroup",
        col_wrap=6,
        element="step",
        facet_kws=dict(ylim=(0.0, 0.5)),
        stat="density",
    )
    plt.show()

    # cleaner view
    sbn.displot(
        data=df,
        x="auroc",
        kind="kde",
        hue="feature_group",
        col="subgroup",
        col_wrap=6,
        facet_kws=dict(ylim=(0.0, 0.5)),
    )
    plt.show()

    # violin plot
    sbn.catplot(data=df, y="auroc", x="subgroup", hue="feature_group", kind="violin")
    resize_fig()
    plt.show()

    # Very useful: shows best RMT performance due to outlier performances
    sbn.catplot(
        data=df, y="auroc", x="subgroup", hue="feature_group", kind="box", col="preproc"
    )
    resize_fig()
    plt.show()

    # Also useful if you re-order hue and cols to show max_eigs most important
    sbn.catplot(
        data=df, y="auroc", x="slice", hue="feature_group", kind="box", col="preproc"
    )
    resize_fig()
    plt.show()

    # This is important: Shows clear interaction between preproc and slice
    sbn.catplot(
        data=df,
        y="auroc",
        x="slice",
        hue="feature_group",
        kind="box",
        row="preproc",
        col="subgroup",
    )
    resize_fig()
    plt.show()

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


def plot_topk_features_by_aggregation(sorter: str, k: int = 5) -> None:
    df = load_all_renamed()
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
    df = load_all_renamed()
    # The more levels we include, the less we "generalize" our claims
    groupers = [grouper for grouper in SUBGROUPERS if "preproc" not in grouper]
    aggregates = get_aggregates(groupers)
    for agg in aggregates:  # since we split the visuals to cover this
        if "preproc" in agg:
            agg.remove("preproc")
    df_nopre = df.loc[df.preproc == "minimal"]
    df_pre = df.loc[df.preproc != "minimal"]
    unq = df.feature.unique()
    ordering = get_feature_ordering(unq)
    palette = make_palette(unq)

    for grouper, aggregate in zip(groupers, aggregates):
        min_summarized_per_feature = np.unique(df.groupby(grouper + ["feature"]).count())[
            0
        ]
        if sorter == "median":
            summary_pre = df_pre.groupby(grouper + ["feature"]).median(numeric_only=True)
            summary_nopre = df_nopre.groupby(grouper + ["feature"]).median(
                numeric_only=True
            )

        else:
            summary_pre = df_pre.groupby(grouper + ["feature"]).max(numeric_only=True)
            summary_nopre = df_nopre.groupby(grouper + ["feature"]).max(numeric_only=True)
        bests_pre = (
            summary_pre.reset_index()
            .groupby(grouper)
            .apply(lambda g: g.nlargest(k, columns="auroc"))
        )
        bests_nopre = (
            summary_nopre.reset_index()
            .groupby(grouper)
            .apply(lambda g: g.nlargest(k, columns="auroc"))
        )

        best_feats_pre = bests_pre["feature"]
        best_feats_nopre = bests_nopre["feature"]

        sbn.set_style("darkgrid")
        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
        ax_nopre: Axes = axes.flat[0]  # type: ignore
        ax_pre: Axes = axes.flat[1]  # type: ignore
        sbn.countplot(
            y=best_feats_pre,
            order=ordering,
            color="black",
            palette=palette,
            ax=ax_pre,
        )
        sbn.countplot(
            y=best_feats_nopre,
            order=ordering,
            color="black",
            palette=palette,
            ax=ax_nopre,
        )
        Sorter = sorter.capitalize() if sorter == "median" else "Max"
        suptitle = f"Total Number of Instances where Feature Yields One of the Top-{k} {Sorter} AUROCs"
        ylabel_pre = "More Preprocessing"
        ylabel_nopre = "Brain-extraction only"
        title = f"Summarizing / Grouping at level of: {grouper}"

        if len(aggregate) > 0:
            title += f"\n(i.e. expected {sorter} performance across all variations of choices of {aggregate})"

        title += (
            f"\n[Number of 5-fold runs summarized by {Sorter} "
            f"per feature grouping: {min_summarized_per_feature}+]"
        )

        ax_pre.set_title(title, fontsize=10)
        ax_nopre.set_title(title, fontsize=10)
        ax_pre.set_ylabel(ylabel_pre, fontsize=14)
        ax_nopre.set_ylabel(ylabel_nopre, fontsize=14)
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
    df = load_all_renamed()

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
    df = load_all_renamed()

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
    K = 3
    plot_topk_features_by_aggregation(sorter="median", k=K)
    plot_topk_features_by_aggregation(sorter="best", k=K)
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="data",
        position=(0.86, 0.15),
    )
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="classifier",
        position=(0.85, 0.28),
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="data",
        position=(0.64, 0.45),
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="classifier",
        position=(0.87, 0.26),
    )

    K = 5
    plot_topk_features_by_aggregation(sorter="median", k=K)
    plot_topk_features_by_aggregation(sorter="best", k=K)
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="data",
        position=(0.86, 0.15),
    )
    plot_topk_features_by_grouping(
        sorter="median",
        k=K,
        by="classifier",
        position=(0.85, 0.28),
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="data",
        position=(0.64, 0.45),
    )
    plot_topk_features_by_grouping(
        sorter="best",
        k=K,
        by="classifier",
        position=(0.56, 0.55),
    )

    plot_topk_features_by_preproc(sorter="median", k=3)
    plot_topk_features_by_preproc(sorter="median", k=5)
    plot_topk_features_by_preproc(sorter="best", k=3)
    plot_topk_features_by_preproc(sorter="best", k=5)


if __name__ == "__main__":
    simplefilter(action="ignore", category=PerformanceWarning)
    print("\n" * 50)
    # df = load_all_renamed()
    summarize_performance_by_aggregation(metric="auroc", summarizer="median")

    # generate_all_topk_plots()

    # df = load_all_renamed()
    # df = df.loc[df.preproc != "minimal"]
    # df = df.loc[df.preproc == "minimal"]
    # naive_describe(df)

    # naive_describe(df)
    # print(f"Summarized {len(df)} 5-fold runs")
