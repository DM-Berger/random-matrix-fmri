# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import pandas as pd
from pandas import DataFrame

PROJECT = ROOT.parent

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
    header = "=" * 80 + "\n"
    footer = "\n" + ("=" * 80)
    drops = [
        "acc+",
        "auroc",
        "mean_acc",
        "min_acc",
        "min_auroc",
        "max_auroc",
        "classifier_GradientBoostingClassifier",
    ]

    print(f"{header}Spearman correlations with AUROC{footer}")
    print(
        corrs.corr(method="spearman")
        .loc["auroc"]
        .drop(index=drops)
        .sort_values(ascending=False)
    )
    print(f"{header}Correlation (Spearman) between feature inclusion and AUROC{footer}")
    print(
        corrs.corr(method="spearman")
        .loc["auroc"]
        .drop(index=drops)
        .filter(like="feature_")
        .rename(index=corr_renamer)
        .sort_values(ascending=False)
    )

    print(f"{header}Spearman correlations with AUROC for predictive pairs{footer}")
    print(
        corrs_pred.corr(method="spearman")
        .loc["auroc"]
        .drop(index=drops)
        .sort_values(ascending=False)
    )
    print(
        f"{header}Correlation (Spearman) between feature inclusion and AUROC for predictive pairs{footer}"
    )
    print(
        corrs_pred.corr(method="spearman")
        .loc["auroc"]
        .drop(index=drops)
        .filter(like="feature_")
        .rename(index=corr_renamer)
        .sort_values(ascending=False)
    )

    print(f"{header}Feature best AUROCs overall:{footer}")
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

    print(f"{header}Feature mean and median AUROCs overall (median-sorted):{footer}")
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

    print(f"{header}Best AUROCs by feature and dataset:{footer}")
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

    print(f"{header}Mean/Median AUROCs by feature and dataset (median-sorted):{footer}")
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
        f"{header}Summary stats of AUROCs for 'nopain v duloxetine' comparison (max-sorted):{footer}"
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
    paths = [
        PROJECT / "all_combined_predictions.json",
        PROJECT / "eigenvalue_predictions.json",
        PROJECT / "eigs-minmax-5_predictions.json",
        PROJECT / "eigs-minmax-10_predictions.json",
        PROJECT / "eigs-minmax-20_predictions.json",
        PROJECT / "eigs-middle-10_predictions.json",
        PROJECT / "eigs-middle-20_predictions.json",
        PROJECT / "eigs-middle-40_predictions.json",
        PROJECT / "eig_smoothed_predictions.json",
        PROJECT / "eigenvalues+eig_smoothed_predictions.json",
        PROJECT / "eigenvalues+eig_savgol_predictions.json",
        PROJECT / "eig_savgol_predictions.json",
        PROJECT / "rigidity_predictions.json",
        PROJECT / "levelvar_predictions.json",
        PROJECT / "eig+levelvar_predictions.json",
        PROJECT / "eig+rigidity_predictions.json",
        PROJECT / "eig+unfolded_predictions.json",
        PROJECT / "eig+unfolded+levelvar_predictions.json",
        PROJECT / "eig+unfolded+rigidity_predictions.json",
        PROJECT / "rigidity+levelvar_predictions.json",
        PROJECT / "unfolded_predictions.json",
        PROJECT / "unfolded+levelvar_predictions.json",
        PROJECT / "unfolded+rigidity_predictions.json",
        PROJECT / "unfolded+rigidity+levelvar_predictions.json",
    ]
    df = pd.concat([pd.read_json(path) for path in paths], axis=0, ignore_index=True)
    df = df.loc[df.preproc == "minimal"]
    # describe(df)

    non_reflects = df.data.apply(lambda s: "Reflect" not in s)
    df = df.loc[non_reflects]
    print("\n\nRemoving meaningless 'REFLECT' data:\n")
    describe(df)
