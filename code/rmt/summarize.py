import pandas as pd

from pathlib import Path
from typing import Any, Tuple

from rmt._data_constants import DATA_ROOT, DATASETS, DATASETS_FULLPRE
from rmt._filenames import relpath, stats_fnames
from rmt._precompute import precompute_dataset
from rmt.comparisons import Pairings


def compute_all_diffs_dfs(args: Any, silent: bool = False) -> Path:
    _, diffs_out = stats_fnames(args, args.normalize, extension="csv")
    all_diffs_out = DATA_ROOT / diffs_out
    DUDS = ["control_pre_v_parkinsons", "park_pre_v_parkinsons", "control_v_park_pre", "control_v_control_pre"]

    all_diffs = []
    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS
    for dataset_name, dataset in datasets.items():
        if dataset_name == "SINGLE_SUBJECT":
            continue
        precompute_dataset(dataset_name, args=args, force_all=False, silent=silent)
        pairings = Pairings(args, dataset_name)
        for pair in pairings.pairs:
            if pair.label in DUDS:
                continue
            diffs = pair.paired_differences(args)
            all_diffs.append(diffs)

    diffs_df = pd.concat(all_diffs)
    diffs_df.to_csv(all_diffs_out)
    if not silent:
        print(diffs_df)
    print(f"Saved all differences to {all_diffs_out}")
    return all_diffs_out


def compute_all_preds_df(args: Any, silent: bool = False, force: bool = False) -> Path:
    preds_out, _ = stats_fnames(args, args.normalize, extension="csv")
    all_preds_out = DATA_ROOT / preds_out
    if not force and all_preds_out.exists():
        return all_preds_out
    DUDS = ["control_pre_v_parkinsons", "park_pre_v_parkinsons", "control_v_park_pre", "control_v_control_pre"]
    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS

    all_preds = []
    for dataset_name, dataset in datasets.items():
        if dataset_name == "SINGLE_SUBJECT":
            continue
        precompute_dataset(dataset_name, args=args, force_all=False, silent=silent)
        pairings = Pairings(args, dataset_name)
        for pair in pairings.pairs:
            if pair.label in DUDS:
                continue
            preds = pair.paired_predicts(args=args, logistic=True, normalize=args.normalize, silent=silent)
            all_preds.append(preds)

    preds_df = pd.concat(all_preds)
    preds_df.to_csv(all_preds_out)
    if not silent:
        print(preds_df)
    print(f"Saved all predictions to {all_preds_out}")
    return all_preds_out


def supplement_stat_dfs(diffs: Path = None, preds: Path = None, force: bool = False) -> Tuple[Path, Path]:
    preds_out = preds.parent / f"[SUMS]{preds.name}" if preds else None
    diffs_out = diffs.parent / f"[SORT]{diffs.name}" if diffs else None
    if preds:
        if force or not preds_out.exists():  # type: ignore
            spec = ["Brody", "Rigidity", "Levelvar"]
            obs = ["Noise", "Noise (shift)"] + spec
            full = ["Raw Eigs", "Unfolded", "Largest", "Largest20"] + obs
            g = "Guess"
            df_pred = pd.read_csv(preds).set_index(["Algorithm", "Dataset", "Comparison"])

            df_pred["Mean (all)"] = df_pred[full].mean(axis=1) - df_pred[g]
            df_pred["Mean (obs)"] = df_pred[obs].mean(axis=1) - df_pred[g]
            df_pred["Mean (spec)"] = df_pred[spec].mean(axis=1) - df_pred[g]
            df_pred["Max (all)"] = df_pred[full].max(axis=1) - df_pred[g]
            df_pred["Max (obs)"] = df_pred[obs].max(axis=1) - df_pred[g]
            df_pred["Max (spec)"] = df_pred[spec].max(axis=1) - df_pred[g]
            df_pred.to_csv(preds_out)
            print(f"Saved supplemented predictions to {relpath(preds_out)}")  # type: ignore

    if diffs:
        if force or not diffs_out.exists():  # type: ignore
            df_diff = pd.read_csv(diffs)
            df_diff.rename({"Unnamed: 0": "RMT Statistic"}, axis="columns", inplace=True)
            df_diff.set_index(["RMT Statistic", "DATASET", "Comparison"], inplace=True)
            good = ["p (Welch's)", "p (Mann-Whitney)", "d"]
            df_diff.sort_values(by=good, ascending=[True, True, False], inplace=True)
            diffs_out = diffs.parent / f"[SORT]{diffs.name}"
            df_diff.to_csv(diffs_out)
            print(f"Saved sorted diffs to {relpath(diffs_out)}")
    return diffs_out, preds_out  # type: ignore
