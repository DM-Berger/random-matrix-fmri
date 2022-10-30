import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
UPDATED = DATA_ROOT / "updated"
VIGILANCE = UPDATED / "Rest_w_VigilanceAttention/ds001168-download"
COMBINED_DF = VIGILANCE / "all_participants.json"
TSVS = sorted(VIGILANCE.rglob("*_sessions.tsv"))
PANAS_ATTENTION_COLS = [
    "panas_attentive",
    "panas_alert",
    "panas_sluggish",
    "panas_tired",
    "panas_sleepy",
    "panas_drowsy",
    "panas_interested",
    "panas_concentrating",
]
NEG_SCORE = ["panas_sluggish", "panas_tired", "panas_sleepy", "panas_drowsy"]
VIGILANCE_COLS = ["vigilance", "vigilance_nyc-q"]
TASK_ATTENTION_COLS = [
    "CCPT_avg_succ_RT",
    # "CCPT_avg_FP_RT",  # This column is almost entirely NaNs
    # "CCPT_avg_FN_RT",  # This column contains multiple NaNs
    "CCPT_succ_count",
    "CCPT_FP_count",
    "CCPT_FN_count",
]


CSVS = [
    VIGILANCE / "task_attention.csv",
    VIGILANCE / "task_attention_ses-1.csv",
    VIGILANCE / "task_attention_ses-2.csv",
    VIGILANCE / "weekly_attentions.csv",
    VIGILANCE / "weekly_attentions_ses-1.csv",
    VIGILANCE / "weekly_attentions_ses-2.csv",
    VIGILANCE / "selfreported_vigilance.csv",
    VIGILANCE / "selfreported_vigilance_ses-1.csv",
    VIGILANCE / "selfreported_vigilance_ses-2.csv",
]


def get_combined_df() -> DataFrame:
    dfs = [pd.read_csv(tsv, sep="\t") for tsv in TSVS]
    for i in range(len(dfs)):
        df = dfs[i]
        tsv = TSVS[i]
        df.insert(loc=0, column="sid", value=0)
        df["sid"] = re.search(r"sub-(\d\d).*", str(tsv))[1]  # type: ignore
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def get_comparison_df(
    df: DataFrame,
    attention: Literal["weekly", "task", "vigilance"],
    session: Literal[None, 1, 2],
) -> DataFrame:
    base_cols = ["sid", "session"]
    if attention == "weekly":
        df = df.loc[:, base_cols + PANAS_ATTENTION_COLS]
        for col in NEG_SCORE:
            df.loc[:, col] = 9 - df[col].copy()
    elif attention == "vigilance":
        df = df.loc[:, base_cols + VIGILANCE_COLS]
    elif attention == "task":
        df = df.loc[:, base_cols + TASK_ATTENTION_COLS]
    else:
        raise ValueError(f"Invalid attention kind: {attention}")
    if session is not None:
        df = df.loc[df["session"] == f"ses-{int(session)}"]
    return df


def kmeans_label(df: DataFrame) -> DataFrame:
    x = df.drop(columns=["sid", "session"])
    x = StandardScaler().fit_transform(x)
    km = KMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(x)
    labeled = df.copy()
    labeled["label"] = labels

    # ensure label "1" is "high attention"
    g0 = x[labeled["label"] == 0]
    g1 = x[labeled["label"] == 1]
    mean0 = g0.mean()
    mean1 = g1.mean()
    if mean1 < mean0:
        labeled["label"] = 1 - labeled["label"]
    labeled.loc[labeled["label"] == 1, "label"] = "high"
    labeled.loc[labeled["label"] == 0, "label"] = "low"
    return labeled


if __name__ == "__main__":
    df = get_combined_df()
    df.to_json(COMBINED_DF, indent=2)
    print(f"Saved combined info data to {COMBINED_DF}")

    for attention in ["vigilance", "weekly", "task"]:
        for session in [None, 1, 2]:
            df_vig = get_comparison_df(df, attention="vigilance", session=session)
            labeled = kmeans_label(df_vig)
            print(f"{attention} session={session}\n{labeled.label.value_counts()}")
