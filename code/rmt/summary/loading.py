# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from pathlib import Path

import pandas as pd
from joblib import Memory
from pandas import DataFrame
from tqdm import tqdm
from typing_extensions import Literal

from rmt.summary.groupings import (
    coarse_feature_grouping,
    fine_feature_grouping,
    slice_grouping,
)
from rmt.updated_features import FEATURE_OUTFILES as PATHS

PROJECT = ROOT.parent
MEMORY = Memory(PROJECT / "__JOBLIB_CACHE__")


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
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["fine_feature"] = df["feature"].apply(fine_feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)
    df["coarse_feature"] = df["feature"].apply(coarse_feature_grouping)
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
    df["subgroup"] = df["data"] + " - " + df["comparison"]
    df["fine_feature"] = df["feature"].apply(fine_feature_grouping)
    df["slice_group"] = df["slice"].apply(slice_grouping)
    df["coarse_feature"] = df["feature"].apply(coarse_feature_grouping)
    return df


def load_combined(drop_ses: bool = True) -> DataFrame:
    df = load_all_renamed()
    ts = load_tseries()
    df = pd.concat([df, ts], axis=0)
    if drop_ses:
        df = df.loc[~df.data.str.contains("Ses")]
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
