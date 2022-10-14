from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DFS = [
    ROOT / "eigenvalue_predictions.json",
    ROOT / "rigidity_predictions.json",
    ROOT / "levelvar_predictions.json",
]

if __name__ == "__main__":
    dfs = [pd.read_json(df) for df in DFS]
    for i, df in enumerate(dfs):
        out = ROOT / f"{DFS[i].stem}.csv"
        df.to_csv(out)
        print(f"Converted {DFS[i]} to {out}")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.sort_values(by="auroc", ascending=False, inplace=True)
    idx = df["acc+"] > 0.01
    idx1 = df["acc+"] > 0.0
    idx2 = df["acc+"] > 0.001
    idx3 = df["acc+"] > 0.01
    print(
        df.sort_values(by="acc+", ascending=False)
        .loc[idx]
        .to_markdown(tablefmt="simple", floatfmt="0.3f")
    )
    idx = np.array(df["data"].apply(lambda s: "Reflect" not in s), dtype=np.bool_)
    print(
        df.sort_values(by="auroc", ascending=False)
        .loc[idx]
        .to_markdown(tablefmt="simple", floatfmt="0.3f")
    )
    print(
        f"Proportion of pairings predicted better than guessing: {np.sum(idx1) / len(df):0.3f}"
    )
    print(
        f"Proportion of pairings predicted better than guessing by 0.1% or more: {np.sum(idx2) / len(df):0.3f}"
    )
    print(
        f"Proportion of pairings predicted better than guessing by 1.0% or more: {np.sum(idx3) / len(df):0.3f}"
    )

    idx = np.array(df["data"].apply(lambda s: "Reflect" not in s), dtype=np.bool_)
    hard = df.loc[idx]
    idx1 = hard["acc+"] > 0.0
    idx2 = hard["acc+"] > 0.001
    idx3 = hard["acc+"] > 0.01
    print(
        f"Proportion of non-trivial pairings predicted better than guessing: {np.sum(idx1) / len(hard):0.3f}"
    )
    print(
        f"Proportion of non-trivial pairings predicted better than guessing by 0.1% or more: {np.sum(idx2) / len(hard):0.3f}"
    )
    print(
        f"Proportion of non-trivial pairings predicted better than guessing by 1.0% or more: {np.sum(idx3) / len(hard):0.3f}"
    )

    print(
        pd.get_dummies(df)
        .corr("spearman", numeric_only=False)
        .loc["acc+"]
        .to_frame()
        .sort_values(by="acc+", ascending=False)
    )
    print(
        pd.get_dummies(df[df["acc+"] > 0.01])
        .corr("spearman", numeric_only=False)
        .loc["acc+"]
        .to_frame()
        .sort_values(by="acc+", ascending=False)
    )
    print("AUROC correlations")
    print(
        pd.get_dummies(df[df["acc+"] > 0.01])
        .corr("spearman", numeric_only=False)
        .loc["auroc"]
        .to_frame()
        .sort_values(by="auroc", ascending=False)
    )

    csv = ROOT / "all_predictions.csv"
    json = ROOT / "all_predictions.json"
    df.to_csv(csv)
    df.to_json(json)
    print(f"Saved all predictions to {csv}")
    print(f"Saved all predictions to {json}")
