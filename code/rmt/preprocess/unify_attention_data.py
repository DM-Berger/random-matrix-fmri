import re
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
CACHE_DIR = DATA_ROOT.parent / "__OBSERVABLES_CACHE__"
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


def median_split_labels(
    attention: Literal["weekly", "task", "vigilance"],
    session: Literal[None, 1, 2, "ses-1", "ses-2"],
) -> DataFrame:
    if session == "ses-1":
        session = 1
    if session == "ses-2":
        session = 2
    to_hash = (attention, str(session))
    hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
    outfile = CACHE_DIR / f"med-split_{hsh}.json"
    if outfile.exists():
        labeled = pd.read_json(outfile)
        labeled.loc[:, "sid"] = labeled["sid"].apply(lambda s: f"{int(s):02d}")
        return labeled

    table = pd.read_json(COMBINED_DF)
    df = get_comparison_df(table, attention=attention, session=session)
    x = df.drop(columns=["sid", "session"])
    x = StandardScaler().fit_transform(x)
    x = x.mean(axis=1)
    labels = (x >= np.median(x)).astype(int)
    labeled = df.copy()
    labeled["label"] = labels

    high_labels = {
        "vigilance": "vigilant",
        "weekly": "trait_attend",
        "task": "task_attend",
    }
    low_labels = {
        "vigilance": "nonvigilant",
        "weekly": "trait_nonattend",
        "task": "task_nonattend",
    }
    labeled.loc[labeled["label"] == 1, "label"] = high_labels[attention]
    labeled.loc[labeled["label"] == 0, "label"] = low_labels[attention]
    labeled.loc[:, "sid"] = labeled["sid"].apply(lambda s: f"{int(s):02d}")
    labeled.to_json(outfile)
    print(f"Saved Median-split groupings to {outfile}")
    return labeled


if __name__ == "__main__":
    df = get_combined_df()
    df.to_json(COMBINED_DF, indent=2)
    print(f"Saved combined info data to {COMBINED_DF}")

    for attention in ["vigilance", "weekly", "task"]:
        for session in [None, 1, 2]:
            labeled = median_split_labels(attention, session)
            print(f"{attention} session={session}\n{labeled.label.value_counts()}")
            # print(labeled["sid"])
