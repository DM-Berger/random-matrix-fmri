# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from typing import Literal

import pandas as pd
from pandas import DataFrame

from rmt.features import FEATURE_OUTFILES as PATHS

PROJECT = ROOT.parent

HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)
DROPS = [
    "acc+",
    "auroc",
    "classifier_GradientBoostingClassifier",
]


def corr_renamer(s: str) -> str:
    if "feature_" in s:
        return s.replace("feature_", "")
    return s


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
# subset by ids, and group again (lol)
def feature_dataset_aurocs(df: DataFrame, sorter: str = "best") -> DataFrame:
    if sorter == "best":
        print(f"{HEADER}Best AUROCs by feature and dataset:{FOOTER}")
    else:
        print(
            f"{HEADER}Mean/Median AUROCs by feature and dataset (median-sorted):{FOOTER}"
        )

    if sorter == "best":
        ordering = ["best", "median", "mean", "std", "5%", "95%"]
    elif sorter == "median":
        ordering = ["median", "best", "mean", "std", "5%", "95%"]
    else:
        raise NotImplementedError()
    processed = (
        df.groupby(["feature", "data", "comparison", "slice"])
        .describe(percentiles=[0.05, 0.95])
        .loc[:, "auroc"]
        .drop(columns=["count"])
        .reset_index()
        .rename(
            columns={
                "max": "best",
                "feature": "Feature",
                "data": "Dataset",
                "50%": "median",
            }
        )
        .sort_values(
            by=["Dataset", "Feature", "comparison", "slice", sorter], ascending=False
        )
        .groupby(["Dataset", "Feature", "comparison", "slice"])
        # .sort_values(by=["Dataset", sorter], ascending=False)  # type: ignore
        # .sort(by=["Dataset", sorter], ascending=False)  # type: ignore
        .max()
        .loc[:, ordering]
        .sort_values(by=["Dataset", "Feature", "comparison", sorter], ascending=False)  # type: ignore
    )
    print(processed.columns)
    print(processed.loc[:, :, :3])
    return processed


def naive_describe(df: DataFrame) -> None:
    # pd.options.display. = True
    pd.options.display.max_info_rows = None
    pd.options.display.max_rows = None
    pd.options.display.expand_frame_repr = True

    # df = df.loc[df.classifier != "SVC"].copy()
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

    # auroc_correlations(df, subset="all", predictive_only=False)
    # auroc_correlations(df, subset="features", predictive_only=False)
    # auroc_correlations(df, subset="all", predictive_only=True)
    # auroc_correlations(df, subset="features", predictive_only=True)

    # feature_aurocs(df, sorter="best")
    # feature_aurocs(df, sorter="median")

    feature_dataset_aurocs(df, sorter="best")
    feature_dataset_aurocs(df, sorter="median")

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


if __name__ == "__main__":
    print("\n" * 50)
    df = pd.concat(
        [pd.read_json(path) for path in PATHS.values()], axis=0, ignore_index=True
    )
    # df = df.loc[df.preproc != "minimal"]
    df = df.loc[df.preproc == "minimal"]
    # naive_describe(df)

    print("\n\nRemoving meaningless 'REFLECT' data:\n")
    non_reflects = df.data.apply(lambda s: "Reflect" not in s)
    df = df.loc[non_reflects]
    naive_describe(df)
