import numpy as np
import pandas as pd

from pathlib import Path

# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
# I know what I'm doing here, shut up Pandas
pd.options.mode.chained_assignment = None

DATA_ROOT = Path(__file__).resolve().parent / "rmt"

TSVS = [
    DATA_ROOT / "sub-01_sessions.tsv",
    DATA_ROOT / "sub-02_sessions.tsv",
    DATA_ROOT / "sub-03_sessions.tsv",
    DATA_ROOT / "sub-04_sessions.tsv",
    DATA_ROOT / "sub-05_sessions.tsv",
    DATA_ROOT / "sub-06_sessions.tsv",
    DATA_ROOT / "sub-07_sessions.tsv",
    DATA_ROOT / "sub-08_sessions.tsv",
    DATA_ROOT / "sub-09_sessions.tsv",
    DATA_ROOT / "sub-10_sessions.tsv",
    DATA_ROOT / "sub-11_sessions.tsv",
    DATA_ROOT / "sub-12_sessions.tsv",
    DATA_ROOT / "sub-13_sessions.tsv",
    DATA_ROOT / "sub-14_sessions.tsv",
    DATA_ROOT / "sub-15_sessions.tsv",
    DATA_ROOT / "sub-16_sessions.tsv",
    DATA_ROOT / "sub-17_sessions.tsv",
    DATA_ROOT / "sub-18_sessions.tsv",
    DATA_ROOT / "sub-19_sessions.tsv",
    DATA_ROOT / "sub-20_sessions.tsv",
    DATA_ROOT / "sub-21_sessions.tsv",
    DATA_ROOT / "sub-22_sessions.tsv",
]

dfs = [pd.read_csv(tsv, delimiter="\t") for tsv in TSVS]
table = pd.DataFrame()
for df in dfs:
    df.set_index("subject_id", inplace=True)
    df.index = df.index.map(lambda idx: "{:02d}".format(idx))
    table = table.append(df, sort=False)

# PANAS ask how much items (adjectives) from last week apply,
# from 1 ('very slightly or not at all') to 9 ('extremely')
# https://www.nature.com/articles/sdata201454.pdf
weekly_attentions = list(
    map(
        lambda s: "panas_" + s,
        [
            "attentive",
            "alert",
            "sluggish",
            "tired",
            "sleepy",
            "drowsy",
            "interested",
            "concentrating",
        ],
    )
)
vigilance = ["vigilance", "vigilance_nyc-q"]
task_attention = [
    "CCPT_avg_succ_RT",
    # "CCPT_avg_FP_RT",  # This column is almost entirely NaNs
    # "CCPT_avg_FN_RT",  # This column contains multiple NaNs
    "CCPT_succ_count",
    "CCPT_FP_count",
    "CCPT_FN_count",
]

# session 1
ses1 = table["session"] == "ses-1"
df_weekly = table.loc[ses1, :][weekly_attentions]
df_vigilance = table.loc[ses1, :][vigilance]
df_task_attention = table.loc[ses1, :][task_attention]

# reverse-score plausible negative-attention items
for colname in ["panas_sluggish", "panas_tired", "panas_sleepy", "panas_drowsy"]:
    df_weekly[colname] = 9 - df_weekly[colname]
# get some overall weekly attention measures
df_weekly["overall"] = df_weekly.sum(axis=1)
df_weekly["overall_z"] = (df_weekly["overall"] - np.mean(df_weekly["overall"])) / np.std(
    df_weekly["overall"], ddof=1
)
df_weekly["high_attender"] = df_weekly["overall_z"] > 0

# nyc-q vigilance scores are from 0-100, scale to 1-9 to match PANAS current
# e.g. PANAS = 0.08*Vigilance + 1
df_vigilance["vigilance_nyc-q"] = np.round(df_vigilance["vigilance_nyc-q"] * 0.08 + 1, 1)
df_vigilance["overall"] = df_vigilance.sum(axis=1)
df_vigilance["overall_z"] = (df_vigilance["overall"] - np.mean(df_vigilance["overall"])) / np.std(
    df_vigilance["overall"], ddof=1
)
df_vigilance["high_attender"] = df_vigilance["overall_z"] > 0

# make "score" be success_count - FP_count - FN_count (penalize false +, -)
df_task_attention["score"] = (
    df_task_attention["CCPT_succ_count"]
    - df_task_attention["CCPT_FP_count"]
    - df_task_attention["CCPT_FN_count"]
)
df_task_attention["score_z"] = (
    df_task_attention["score"] - np.mean(df_task_attention["score"])
) / np.std(df_task_attention["score"], ddof=1)
df_task_attention["high_attender"] = df_task_attention["score_z"] > 0

df_weekly.to_csv("rmt/weekly_attentions_ses-1.csv")
df_vigilance.to_csv("rmt/selfreported_vigilance_ses-1.csv")
df_task_attention.to_csv("rmt/task_attention_ses-1.csv")

# session 2
ses2 = table["session"] == "ses-2"
df_weekly = table.loc[ses2, :][weekly_attentions]
df_vigilance = table.loc[ses2, :][vigilance]
df_task_attention = table.loc[ses2, :][task_attention]

# reverse-score plausible negative-attention items
for colname in ["panas_sluggish", "panas_tired", "panas_sleepy", "panas_drowsy"]:
    df_weekly[colname] = 9 - df_weekly[colname]
# get some overall weekly attention measures
df_weekly["overall"] = df_weekly.sum(axis=1)
df_weekly["overall_z"] = (df_weekly["overall"] - np.mean(df_weekly["overall"])) / np.std(
    df_weekly["overall"], ddof=1
)
df_weekly["high_attender"] = df_weekly["overall_z"] > 0

# nyc-q vigilance scores are from 0-100, scale to 1-9 to match PANAS current
# e.g. PANAS = 0.08*Vigilance + 1
df_vigilance["vigilance_nyc-q"] = np.round(df_vigilance["vigilance_nyc-q"] * 0.08 + 1, 1)
df_vigilance["overall"] = df_vigilance.sum(axis=1)
df_vigilance["overall_z"] = (df_vigilance["overall"] - np.mean(df_vigilance["overall"])) / np.std(
    df_vigilance["overall"], ddof=1
)
df_vigilance["high_attender"] = df_vigilance["overall_z"] > 0

# make "score" be success_count - FP_count - FN_count (penalize false +, -)
df_task_attention["score"] = (
    df_task_attention["CCPT_succ_count"]
    - df_task_attention["CCPT_FP_count"]
    - df_task_attention["CCPT_FN_count"]
)
df_task_attention["score_z"] = (
    df_task_attention["score"] - np.mean(df_task_attention["score"])
) / np.std(df_task_attention["score"], ddof=1)
df_task_attention["high_attender"] = df_task_attention["score_z"] > 0

df_weekly.to_csv("rmt/weekly_attentions_ses-2.csv")
df_vigilance.to_csv("rmt/selfreported_vigilance_ses-2.csv")
df_task_attention.to_csv("rmt/task_attention_ses-2.csv")
