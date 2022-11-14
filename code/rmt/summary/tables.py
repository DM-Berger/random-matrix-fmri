# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

import pandas as pd
import numpy as np
from numpy import ndarray
from numba import njit
from tqdm import tqdm
from pandas import DataFrame
from typing_extensions import Literal

from rmt.summary.loading import get_described, get_described_w_classifier, load_combined
from rmt.summary.groupings import fine_feature_grouping, slice_grouping
from rmt.summary.constants import DROPS


HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)


def print_correlations(df: DataFrame, sorter: str) -> None:
    print(
        pd.get_dummies(df.reset_index())
        .corr(method="spearman")
        .round(3)
        .loc[sorter]
        .sort_values(ascending=False)
    )


def auroc_correlations(
    df: DataFrame, subset: Literal["all", "features"], predictive_only: bool
) -> DataFrame:
    def corr_renamer(s: str) -> str:
        if "feature_" in s:
            return s.replace("feature_", "")
        return s

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
    # plot_topk_features_by_aggregation(sorter=sorter)

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
    # plot_feature_counts_grouped(bests3, sorter, by="data")
    # plot_feature_counts_grouped(bests3, sorter, by="classifier")

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


def get_overfit_scores() -> None:
    # df = load_all_renamed()
    df = load_combined()
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["fine_feature"] = df["feature"].apply(fine_feature_grouping)
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
        ["subgroup", "fine_feature", "classifier", "preproc", "trim", "slice_group"]
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

    # sbn.catplot(
    #     data=corrs,
    #     y="correlation",
    #     x="classifier",
    #     order=CLASSIFIER_ORDER,
    #     col="subgroup",
    #     col_order=SUBGROUP_ORDER,
    #     col_wrap=3,
    #     kind="bar",
    # )
    # fig = plt.gcf()
    # fig.suptitle("Correlation with Overfitting (based on AUROC)")
    # plt.show()

    print(
        df.groupby(["subgroup", "classifier"])
        .describe()["is_overfit"]
        .drop(columns=["std", "min", "max", "25%", "75%", "count"])
    )


def summary_stats_and_tables() -> None:
    df = load_combined()
    df_ses = load_combined(drop_ses=False)
    df["mega_feature"] = df["coarse_feature"].apply(
        lambda s: "eigs_all" if s != "tseries" else "tseries"
    )
    df_ses["mega_feature"] = df_ses["coarse_feature"].apply(
        lambda s: "eigs_all" if s != "tseries" else "tseries"
    )
    meds = df.groupby(["subgroup", "mega_feature"])["auroc"].median().unstack()
    meds_ses = df_ses.groupby(["subgroup", "mega_feature"])["auroc"].median().unstack()

    print("Medians excluding Ses- subgroups:")
    print(meds.round(3))
    print("Medians including Ses- subgroups:")
    print(meds_ses.round(3))

    print("Medians > 0.5 for Ses- subgroups:")
    print((meds > 0.5).mean(axis=0).round(3))
    print("Range:")
    print(meds.quantile([0, 1]).round(2).T)
    print("Medians > 0.5 including Ses- subgroups:")
    print((meds_ses > 0.5).mean(axis=0).round(3))
    print("Range:")
    print(meds_ses.quantile([0, 1]).round(2).T)

    print("Proportion of AUROCs greater than 0.5 (Excluding Ses)")
    g05 = (
        df.groupby(["subgroup", "mega_feature"])["auroc"]
        .apply(lambda grp: (grp > 0.5).mean())
        .unstack()
    )
    print(g05)
    print("Proportion of AUROCs greater than 0.5 (Incuding Ses)")
    g05_ses = (
        df_ses.groupby(["subgroup", "mega_feature"])["auroc"]
        .apply(lambda grp: (grp > 0.5).mean())
        .unstack()
    )
    print(g05_ses)

    print("Proportion of AUROCs greater than 0.5 (Excluding Ses)")
    print(g05.describe().T)
    print("Proportion of AUROCs greater than 0.5 (Including Ses)")
    print(g05_ses.describe().T)


def make_feature_table() -> None:
    df = load_combined()
    feats = (
        df.loc[:, ["coarse_feature", "fine_feature", "feature"]].value_counts().to_frame()
    )
    feats.sort_values(by=["coarse_feature", "fine_feature", "feature"], inplace=True)
    print(feats.to_latex())


@njit(fastmath=True, cache=True)
def paired_difference(arr1: ndarray, arr2: ndarray) -> tuple[float, float, float, float]:
    distance = 0
    count = 0
    pos, neg, eq = 0, 0, 0
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            d = arr1[i] - arr2[j]
            if d > 0:
                pos += 1
            elif d < 0:
                neg += 1
            else:
                eq += 1
            distance += (1 / (count + 1)) * (d - distance)
    return distance, pos, neg, eq


def feature_superiority() -> None:
    df = load_combined()
    unf = np.asarray(df[df.feature == "unfolded"]["auroc"])
    results = []
    for feature in tqdm(df[df.fine_feature == "rmt + eigs"].feature.unique()):
        compare = np.asarray(df[df.feature == feature]["auroc"])
        diff, pos, neg, eq = paired_difference(unf, compare)
        total = pos + neg + eq
        result = DataFrame(
            {
                "feature": feature,
                "avg_diff": diff,
                "unf_greater": pos / total,
                "unf_lower": neg / total,
            },
            index=[0],
        )
        results.append(result)
        print(result)

    result = pd.concat(results, axis=0, ignore_index=True)
    print(result)
    """
                              feature  avg_diff  unf_greater  unf_lower

    0                        rigidity -0.070536     0.551810   0.448034
    1                        levelvar  0.074107     0.545576   0.454281
    2             rigidity + levelvar  0.038343     0.551635   0.448217
    3                        unfolded  0.000000     0.499420   0.499420
    4             unfolded + levelvar  0.000000     0.499180   0.499667
    5             unfolded + rigidity  0.000000     0.498636   0.500209
    6  unfolded + rigidity + levelvar  0.000000     0.498541   0.500299
    0  eigs + rigidity + levelvar     -0.128571     0.514880   0.484008
    1             eigs + levelvar     -0.135714     0.516269   0.482620
    2             eigs + rigidity     -0.135714     0.516425   0.482465
    3             eigs + unfolded     -0.135714     0.514644   0.484246
    4  eigs + unfolded + levelvar     -0.135714     0.515045   0.483844
    5  eigs + unfolded + rigidity     -0.128571     0.514770   0.484120
    """
