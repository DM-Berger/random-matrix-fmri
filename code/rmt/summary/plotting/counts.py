# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from matplotlib.axes import Axes
from pandas import DataFrame

from rmt.summary.constants import AGGREGATES, SUBGROUPERS, get_aggregates
from rmt.summary.groupings import get_feature_ordering, is_rmt, make_palette
from rmt.summary.loading import load_combined
from rmt.summary.plotting.utils import make_legend
from rmt.visualize import UPDATED_PLOT_OUTDIR as PLOT_OUTDIR
from rmt.visualize import best_rect


def topk_outdir(k: int) -> Path:
    outdir = PLOT_OUTDIR / f"top-{k}_plots"
    outdir.mkdir(exist_ok=True, parents=True)
    return Path(outdir)


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
        suptitle = (
            "Total Number of Instances where Feature Yields "
            f"One of the Top-{k} {Sorter} AUROCs"
        )
        title = f"Summarizing / Grouping at level of: {grouper}"
        if len(aggregates) > 0:
            title += (
                f"\n(i.e. expected {sorter} performance across "
                f"all variations of choice of {aggregates})"
            )
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

        else:
            summaries = [
                df_.groupby(grouper + ["feature"]).max(numeric_only=True)
                for df_ in preproc_dfs
            ]
        bests = [
            (
                summary.reset_index()
                .groupby(grouper, group_keys=False)
                .apply(lambda g: g.nlargest(k, columns="auroc"))
            )
            for summary in summaries
        ]

        best_feats = [best["feature"] for best in bests]

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
        suptitle = (
            "Total Number of Instances where Feature Yields "
            f"One of the Top-{k} {Sorter} AUROCs"
        )
        title = f"Summarizing / Grouping at level of: {grouper}"

        if len(aggregate) > 0:
            title += (
                f"\n(i.e. expected {sorter} performance across "
                f"all variations of choices of {aggregate})"
            )

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
    suptitle = (
        f"Overall Frequency of Feature Producing one of Top {k} "
        f"{sorter.capitalize()} AUROCs"
    )

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
    rmt_special: list[str] = sorted(filter(is_rmt, order))
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
        f"Frequency of Feature Producing one of Top 3 {sorter.capitalize()} "
        f"AUROCs by {by.capitalize()}"
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
