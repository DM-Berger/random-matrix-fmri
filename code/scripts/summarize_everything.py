# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import pandas as pd
from pandas import DataFrame

from rmt.features import FEATURE_OUTFILES as PATHS

PROJECT = ROOT.parent

HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)
DROPS = [
    "acc+",
    "auroc",
    "mean_acc",
    "min_acc",
    "min_auroc",
    "max_auroc",
    "classifier_GradientBoostingClassifier",
]


def corr_renamer(s: str) -> str:
    if "feature_" in s:
        return s.replace("feature_", "")
    return s


def describe(df: DataFrame) -> None:
    # pd.options.display. = True
    pd.options.display.max_info_rows = None
    pd.options.display.max_rows = None
    pd.options.display.expand_frame_repr = True

    df = df.loc[df.classifier != "SVC"].copy()
    df.loc[:, "feature"] = df.feature.str.replace("plus", " + ").copy()
    df.loc[:, "feature"] = df.feature.str.replace("eig ", "eigs ").copy()
    df.loc[:, "feature"] = df.feature.str.replace("eigenvalues", "eigs").copy()
    df.loc[:, "feature"] = df.feature.str.replace("eigssmoothed", "eigs_smoothed").copy()
    df.loc[:, "feature"] = df.feature.str.replace("eigssavgol", "eigs_savgol").copy()
    df.loc[:, "feature"] = df.feature.str.replace("smoothed", "smooth").copy()
    df.loc[:, "feature"] = df.feature.str.replace(
        "allfeatures", "eigs + rigidity + levelvar"
    ).copy()
    df.loc[:, "feature"] = df.feature.str.replace("rigidities", "rigidity").copy()
    df.loc[:, "feature"] = df.feature.str.replace("levelvars", "levelvar").copy()
    corrs = pd.get_dummies(df)
    corrs_pred = corrs.loc[corrs["acc+"] > 0.0]

    print(f"{HEADER}Spearman correlations with AUROC{FOOTER}")
    print(
        corrs.corr(method="spearman")
        .loc["auroc"]
        .drop(index=DROPS)
        .rename(index=corr_renamer)
        .sort_values(ascending=False)
    )
    print(f"{HEADER}Correlation (Spearman) between feature inclusion and AUROC{FOOTER}")
    print(
        corrs.corr(method="spearman")
        .loc["auroc"]
        .drop(index=DROPS)
        .filter(like="feature_")
        .rename(index=corr_renamer)
        .sort_values(ascending=False)
    )

    print(f"{HEADER}Spearman correlations with AUROC for predictive pairs{FOOTER}")
    print(
        corrs_pred.corr(method="spearman")
        .loc["auroc"]
        .drop(index=DROPS)
        .sort_values(ascending=False)
    )
    print(
        f"{HEADER}Correlation (Spearman) between feature inclusion and AUROC for predictive pairs{FOOTER}"
    )
    print(
        corrs_pred.corr(method="spearman")
        .loc["auroc"]
        .drop(index=DROPS)
        .filter(like="feature_")
        .rename(index=corr_renamer)
        .sort_values(ascending=False)
    )

    print(f"{HEADER}Feature best AUROCs overall:{FOOTER}")
    print(
        df.groupby("feature")
        .describe()
        .rename(columns={"max": "best"})
        .loc[:, "auroc"]
        .round(4)
        .drop(columns="count")
        .loc[:, ["best"]]
        .sort_values(by="best", ascending=False)
    )

    print(f"{HEADER}Feature mean and median AUROCs overall (median-sorted):{FOOTER}")
    print(
        df.groupby("feature")
        .describe()
        .rename(columns={"50%": "median"})
        .loc[:, "auroc"]
        .round(4)
        .drop(columns="count")
        .loc[:, ["mean", "median"]]
        .sort_values(by="median", ascending=False)
    )

    print(f"{HEADER}Best AUROCs by feature and dataset:{FOOTER}")
    bests = (
        df.groupby(["feature", "data"])
        .describe()
        .loc[:, "auroc"]
        .round(3)
        .drop(columns=["count", "min", "25%", "50%", "75%"])
        .reset_index()
        .sort_values(by=["data", "max"], ascending=False)
        .rename(columns={"max": "best_AUROC", "feature": "Feature", "data": "Dataset"})
        .loc[:, ["Feature", "Dataset", "best_AUROC"]]
        .groupby(["Dataset", "Feature"])
        .max()
    )
    print(bests.sort_values(by=["Dataset", "best_AUROC"], ascending=False))

    print(f"{HEADER}Mean/Median AUROCs by feature and dataset (median-sorted):{FOOTER}")
    descriptives = (
        df.groupby(["feature", "data"])
        .describe()
        .rename(columns={"50%": "median"})
        .loc[:, "auroc"]
        .round(3)
        .drop(columns=["count", "25%", "75%"])
        .reset_index()
        .sort_values(by=["data", "mean", "median"], ascending=False)
        .rename(columns={"feature": "Feature", "data": "Dataset"})
        .loc[:, ["Feature", "Dataset", "mean", "median", "std", "min", "max"]]
        .groupby(["Dataset", "Feature"])
    )
    print(descriptives.max().sort_values(by=["Dataset", "median"], ascending=False))
    # print(df.groupby(["feature", "data", "comparison"]).count())

    osteo = (
        df.loc[df.data == "Osteo"]
        .loc[df.comparison == "nopain v duloxetine"]
        .drop(columns=["data", "comparison"])
    )
    print(
        f"{HEADER}Summary stats of AUROCs for 'nopain v duloxetine' comparison (max-sorted):{FOOTER}"
    )
    print(
        osteo.groupby(["feature"])
        .describe()
        .loc[:, "auroc"]
        .round(3)
        .rename(columns={"50%": "median"})
        .loc[:, ["mean", "median", "min", "max", "std"]]
        .sort_values(by=["max", "median"], ascending=False)
        # .reset_index()
        # .sort_values(by=["max"], ascending=False)
        # .rename(columns={"max": "best_AUROC", "feature": "Feature"})
        # .loc[:, ["Feature", "best_AUROC"]]
        # .groupby(["Feature"])
        # .describe()
        # .max()
    )
    # print(
    #     df.groupby(["feature", "data"])
    #     .describe()
    #     .loc[:, "max_auroc"]
    #     # .loc[["mean", "max", "std"]]
    #     .corr(method="spearman", numeric_only=False)
    # )


if __name__ == "__main__":
    df = pd.concat(
        [pd.read_json(path) for path in PATHS.values()], axis=0, ignore_index=True
    )
    df = df.loc[df.preproc != "minimal"]
    describe(df)

    non_reflects = df.data.apply(lambda s: "Reflect" not in s)
    df = df.loc[non_reflects]
    print("\n\nRemoving meaningless 'REFLECT' data:\n")
    describe(df)
