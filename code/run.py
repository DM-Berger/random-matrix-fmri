import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import sys

from typing import Any
from warnings import filterwarnings

from rmt._data_constants import DATA_ROOT, DATASETS, DATASETS_FULLPRE
from rmt._filenames import preview_precompute_outpaths
from rmt._precompute import precompute_dataset
from rmt.args import ARGS
from rmt.plot_datasets import plot_largest, plot_pred_rigidity, plot_pred_levelvar, plot_pred_nnsd
from rmt.summarize import compute_all_diffs_dfs, compute_all_preds_df, supplement_stat_dfs


def plot_learning_fails(args: Any):
    pass


def all_pred_means_by_algo(
    dataset_name: str = None,
    comparison: str = None,
    subtract_guess: bool = True,
    fullpre: bool = True,
    normalize: bool = False,
    silent: bool = False,
    force: bool = False,
) -> None:
    FULLPRE = ARGS.fullpre

    UNFOLD = [5, 7, 9, 11, 13]

    dfs = []
    for trim in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim
        for norm in [normalize]:
            ARGS.normalize = norm
            for degree in UNFOLD:
                ARGS.unfold["degree"] = degree
                supplemented = supplement_stat_dfs(diffs=None, preds=compute_all_preds_df(ARGS, silent=silent))[1]
                df = pd.read_csv(supplemented)
                df["Degree"] = ARGS.unfold["degree"]
                df["Trim"] = ARGS.trim
                dfs.append(df)

    compare: pd.DataFrame = pd.concat(dfs)
    features = ["Raw Eigs", "Largest20", "Largest", "Noise", "Noise (shift)", "Rigidity", "Levelvar"]
    compare = compare[
        [
            "Dataset",
            "Algorithm",
            "Comparison",
            "Degree",
            "Trim",
            "Raw Eigs",
            "Largest20",
            "Largest",
            "Noise",
            "Noise (shift)",
            "Rigidity",
            "Levelvar",
            "Guess",
        ]
    ]
    if dataset_name is not None:
        compare = compare[compare["Dataset"] == dataset_name]
    if comparison is not None:
        compare = compare[compare["Comparison"] == comparison]

    algo_eigs, algo_20, algo_rig, algo_var, algo_largest, algo_noise, algo_noise_shift = [
        pd.DataFrame(dtype=float) for _ in range(7)
    ]
    for algo in compare["Algorithm"].unique():
        # print(algo)
        alg_compare = compare[compare["Algorithm"] == algo]
        if subtract_guess:
            alg_compare = alg_compare[features].apply(lambda col: col - alg_compare["Guess"])
        desc = alg_compare.describe().drop(labels="count", axis=0)
        if algo == "Logistic Regression":
            algo = "LR"
        # guess = desc["Guess"] if subtract_guess else desc["Guess"] - desc["Guess"]
        algo_eigs[algo] = desc["Raw Eigs"].rename(columns={"Raw Eigs": algo})
        algo_20[algo] = desc["Largest20"].rename(columns={"Largest20": algo})
        algo_rig[algo] = desc["Rigidity"].rename(columns={"Rigidity": algo})
        algo_var[algo] = desc["Levelvar"].rename(columns={"Levelvar": algo})
        algo_largest[algo] = desc["Largest"].rename(columns={"Largest": algo})
        algo_noise[algo] = desc["Noise"].rename(columns={"Noise": algo})
        algo_noise_shift[algo] = desc["Noise (shift)"].rename(columns={"Noise (shift)": algo})
        # print(compare[compare["Algorithm"] == algo].describe())
        # print("\n\n")
        # algo_rig["Guess"] =
    d = f"{dataset_name}_" if dataset_name is not None else ""
    c = f"{comparison}_" if comparison is not None else ""
    f = "_fullpre" if FULLPRE else ""
    n = "_normed" if normalize else ""
    g = "-guess" if subtract_guess else ""
    algo_eigs.to_csv(DATA_ROOT / f"{d}{c}all_preds_eigs_by_algo{f}{n}{g}.csv")
    algo_20.to_csv(DATA_ROOT / f"{d}{c}all_preds_largest20_by_algo{f}{n}{g}.csv")
    algo_rig.to_csv(DATA_ROOT / f"{d}{c}all_preds_rigidity_by_algo{f}{n}{g}.csv")
    algo_var.to_csv(DATA_ROOT / f"{d}{c}all_preds_levelvar_by_algo{f}{n}{g}.csv")
    algo_largest.to_csv(DATA_ROOT / f"{d}{c}all_preds_largest_by_algo{f}{n}{g}.csv")
    algo_noise.to_csv(DATA_ROOT / f"{d}{c}all_preds_noise_by_algo{f}{n}{g}.csv")
    algo_noise_shift.to_csv(DATA_ROOT / f"{d}{c}all_preds_noise_shift_by_algo{f}{n}{g}.csv")


def make_pred_means(
    dataset_name: str, comparison: str, subtract_guess: bool = False, silent: bool = False, force: bool = False
) -> None:
    FULLPRE = ARGS.fullpre

    UNFOLD = [5, 7, 9, 11, 13]
    normalize = False

    dfs = []
    for trim in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim
        for norm in [normalize]:
            ARGS.normalize = norm
            for degree in UNFOLD:
                ARGS.unfold["degree"] = degree
                supplemented = supplement_stat_dfs(diffs=None, preds=compute_all_preds_df(ARGS, silent=True))[1]
                df = pd.read_csv(supplemented)
                df["Degree"] = ARGS.unfold["degree"]
                df["Trim"] = ARGS.trim
                dfs.append(df)

    orig: pd.DataFrame = pd.concat(dfs)
    data = orig[orig["Dataset"] == dataset_name]
    compare = data[data["Comparison"] == comparison]
    compare = compare[["Algorithm", "Degree", "Trim", "Rigidity", "Levelvar", "Guess"]]
    # print(compare[["Algorithm", "Degree", "Trim", "Rigidity", "Levelvar", "Guess"]])
    # print("Rigidity:")
    # print(compare["Rigidity"].describe())
    # print("Levelvar:")
    # print(compare["Levelvar"].describe())
    algo_rig, algo_var = pd.DataFrame(), pd.DataFrame()
    for algo in compare["Algorithm"].unique():
        # print(algo)
        desc = compare[compare["Algorithm"] == algo].describe()
        if subtract_guess:
            algo_rig[algo] = desc["Rigidity"].rename(columns={"Rigidity": algo}) - desc["Guess"]
            algo_rig[algo] = desc["Rigidity"].rename(columns={"Rigidity": algo}) - desc["Guess"]
        else:
            algo_rig[algo] = desc["Rigidity"].rename(columns={"Rigidity": algo})
            algo_var[algo] = desc["Levelvar"].rename(columns={"Levelvar": algo})
        # print(compare[compare["Algorithm"] == algo].describe())
        # print("\n\n")
    guess_label = "-guess" if subtract_guess else ""
    rig_out = DATA_ROOT / f"{dataset_name}_{comparison}_rigidity_by_algo{guess_label}.csv"
    var_out = DATA_ROOT / f"{dataset_name}_{comparison}_levelvar_by_algo{guess_label}.csv"
    algo_rig.to_csv(rig_out)
    algo_var.to_csv(var_out)
    print(f"Saved {dataset_name} {comparison} rigidity predictions by algorithm to {rig_out.relative_to(DATA_ROOT)}")
    print(f"Saved {dataset_name} {comparison} levelvar predictions by algorithm to {var_out.relative_to(DATA_ROOT)}")


def make_marchenko_plots(shifted: bool = False):
    for fullpre in [True, False]:
        ARGS.fullpre = fullpre
        for normalize in [False]:
            ARGS.normalize = normalize
            datasets_all = DATASETS_FULLPRE if ARGS.fullpre else DATASETS
            datasets = {}
            for dataset_name, dataset in datasets_all.items():
                if dataset_name == "SINGLE_SUBJECT":
                    continue
                datasets[dataset_name] = dataset

            # print(len(datasets))  # 12
            # sys.exit(1)

            fig: plt.Figure
            fig, axes = plt.subplots(nrows=4, ncols=3)
            suptitle = (
                f"{'Shifted ' if shifted else ''}Marchenko Noise Ratio {'(preprocessed)' if ARGS.fullpre else ''}"
            )
            for i, dataset_name in enumerate(datasets):
                dfs = []
                for groupname, observables in precompute_dataset(dataset_name, ARGS, silent=True).items():
                    df_full = pd.read_pickle(observables["marchenko"])
                    df = pd.DataFrame(df_full.loc["noise_ratio_shifted" if shifted else "noise_ratio", :])
                    df["subgroup"] = [groupname for _ in range(len(df))]
                    dfs.append(df)
                df = pd.concat(dfs)

                sbn.set_context("paper")
                sbn.set_style("ticks")
                # args_prefix = argstrings_from_args(args)[0]
                dname = dataset_name.upper()
                # prefix = f"{dname}_{args_prefix}_{'fullpre_' if args.fullpre else ''}"
                title = f"{dname}"

                with sbn.axes_style("ticks"):
                    subfontsize = 10
                    fontsize = 12
                    ax: plt.Axes = axes.flat[i]
                    sbn.violinplot(x="subgroup", y=f"noise_ratio{'_shifted' if shifted else ''}", data=df, ax=ax)
                    sbn.despine(offset=10, trim=True, ax=ax)
                    ax.set_title(title, fontdict={"fontsize": fontsize})
                    ax.set_xlabel("", fontdict={"fontsize": subfontsize})
                    ax.set_ylabel("", fontdict={"fontsize": subfontsize})
            fig.suptitle(suptitle)
            fig.text(x=0.5, y=0.04, s="Subgroup", ha="center", va="center")  # xlabel
            fig.text(x=0.05, y=0.5, s="Noise Proportion", ha="center", va="center", rotation="vertical")  # ylabel
            fig.subplots_adjust(hspace=0.48, wspace=0.3)
            # sbn.despine(ax=axes.flat[-1], trim=True)
            fig.delaxes(ax=axes.flat[-1])
            plt.show(block=False)

    plt.show()

    # plt.show()
    # plt.close()
    # else:
    # out = outdir / f"{prefix}marchenko.png"
    # plt.gcf().set_size_inches(w=8, h=8)
    # plt.savefig(out)
    # plt.close()
    # print(f"Marchenko plot saved to {relpath(out)}")


def make_largest_plots():
    for fullpre in [True, False]:
        ARGS.fullpre = fullpre
        for NORMALIZE in [False]:
            datasets_all = DATASETS_FULLPRE if ARGS.fullpre else DATASETS
            datasets = {}
            for dataset_name, dataset in datasets_all.items():
                if dataset_name == "SINGLE_SUBJECT":
                    continue
                datasets[dataset_name] = dataset

            # print(len(datasets))  # 12
            # sys.exit(1)

            fig: plt.Figure
            fig, axes = plt.subplots(nrows=4, ncols=3)
            suptitle = f"Largest Eigenvalue{' (preprocessed)' if ARGS.fullpre else ''}"
            for i, dataset_name in enumerate(datasets):
                dfs = []
                for groupname, observables in precompute_dataset(dataset_name, ARGS, silent=True).items():
                    df_full = pd.read_pickle(observables["largest"])
                    df = pd.DataFrame(df_full.loc["largest", :], dtype=float)
                    df["subgroup"] = [groupname for _ in range(len(df))]
                    dfs.append(df)
                df = pd.concat(dfs)

                sbn.set_context("paper")
                sbn.set_style("ticks")
                dname = dataset_name.upper()
                title = f"{dname}"

                with sbn.axes_style("ticks"):
                    subfontsize = 10
                    fontsize = 12
                    ax: plt.Axes = axes.flat[i]
                    sbn.violinplot(x="subgroup", y="largest", data=df, ax=ax)
                    sbn.despine(offset=10, trim=True, ax=ax)
                    ax.set_title(title, fontdict={"fontsize": fontsize})
                    ax.set_xlabel("", fontdict={"fontsize": subfontsize})
                    ax.set_ylabel("", fontdict={"fontsize": subfontsize})
            fig.suptitle(suptitle)
            fig.text(x=0.5, y=0.04, s="Subgroup", ha="center", va="center")  # xlabel
            fig.text(x=0.05, y=0.5, s="Magnitude", ha="center", va="center", rotation="vertical")  # ylabel
            fig.subplots_adjust(hspace=0.48, wspace=0.3)
            # sbn.despine(ax=axes.flat[-1], trim=True)
            fig.delaxes(ax=axes.flat[-1])
            plt.show(block=False)

    plt.show()

    # plt.show()
    # plt.close()
    # else:
    # out = outdir / f"{prefix}marchenko.png"
    # plt.gcf().set_size_inches(w=8, h=8)
    # plt.savefig(out)
    # plt.close()
    # print(f"Marchenko plot saved to {relpath(out)}")


def get_cmds():
    global ARGS
    ARGS.fullpre = True
    for trim_idx in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim_idx
        for normalize in [False]:
            ARGS.normalize = normalize
            for degree in [5, 7, 9, 11, 13]:
                ARGS.unfold["degree"] = degree
                ARGS.print()
                ARGS.cmd()
                # plot_marchenko(ARGS)
                # plot_brody(ARGS)
                # plot_largest(ARGS)
                # compute_all_diffs_dfs(ARGS, NORMALIZE, silent=True)
                # diffs = compute_all_diffs_dfs(ARGS, silent=True)
                # supplement_stat_dfs(diffs=diffs)


filterwarnings("ignore", category=RuntimeWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=np.RankWarning)
# plot_pred_rigidity(ARGS, "OSTEO", "duloxetine_v_nopain", ensembles=True, silent=True, force=False)
# plot_pred_levelvar(ARGS, "OSTEO", "duloxetine_v_nopain", ensembles=True,
# plot_pred_rigidity(ARGS, "PARKINSONS", "control_v_parkinsons", ensembles=True, silent=True, force=False)
# plot_pred_levelvar(ARGS, "PARKINSONS", "control_v_parkinsons", ensembles=True, silent=True, force=False)
plot_pred_nnsd(ARGS, "OSTEO", "duloxetine_v_nopain", trim=4.0, ensembles=True, silent=True, force=False)
plt.show()
# get_cmds()
# make_largest_plots()
# preview_precompute_outpaths(ARGS)
sys.exit(0)

# preds = compute_all_preds_df(ARGS, silent=True)
# supplement_stat_dfs(preds=preds)
# get_cmds()
# all_pred_means_by_algo(ARGS, "PSYCH_VIGILANCE_SES-1", "high_v_low", subtract_guess=False, fullpre=True, normalize=True, silent=True)
# all_pred_means_by_algo(ARGS, "PSYCH_VIGILANCE_SES-2", "high_v_low", subtract_guess=False, fullpre=True,  normalize=True, silent=True)
# all_pred_means_by_algo(ARGS, None, fullpre=False, silent=True)
# all_pred_means_by_algo(
#     ARGS, "OSTEO", "duloxetine_v_nopain", subtract_guess=True, fullpre=True, silent=True
# )
# all_pred_means_by_algo(
#     ARGS, "OSTEO", "duloxetine_v_nopain", subtract_guess=False, fullpre=True, silent=True
# )
# all_pred_means_by_algo(ARGS, "OSTEO", "duloxetine_v_nopain", subtract_guess=True, fullpre=True, normalize=False, silent=True)
# all_pred_means_by_algo(
#     ARGS, "REFLECT_INTERLEAVED", subtract_guess=False, fullpre=True, normalize=True, silent=False
# )
# all_pred_means_by_algo(
#     ARGS,
#     "PARKINSONS",
#     "control_v_parkinsons",
#     subtract_guess=False,
#     fullpre=True,
#     normalize=True,
#     silent=True,
# )
# all_pred_means_by_algo(
#     ARGS,
#     "PARKINSONS",
#     "control_v_parkinsons",
#     subtract_guess=False,
#     fullpre=True,
#     normalize=False,
#     silent=True,
# )
# all_pred_means_by_algo(ARGS, subtract_guess=True, normalize=True, fullpre=True, silent=True)
# all_pred_means_by_algo(ARGS, subtract_guess=True, normalize=False, fullpre=True, silent=True)
# make_pred_means(ARGS, dataset_name="OSTEO", comparison="duloxetine_v_nopain", silent=True)
# make_pred_hists(ARGS, density=True, silent=True)
# make_pred_hists(ARGS, normalize=True, density=True, silent=True)
# make_pred_hists(ARGS, normalize=False, density=True, silent=True)
# make_marchenko_plots(ARGS, shifted=True)
# make_largest_plots(ARGS)
# plot_pred_rigidity(ARGS, "PSYCH_TASK_ATTENTION_SES-1", "high_v_low", silent=True, force=False)
# plot_pred_rigidity(ARGS, "PARKINSONS", "control_v_parkinsons", silent=True, force=False)
# plot_pred_levelvar(ARGS, "PARKINSONS", "control_v_parkinsons", silent=True, force=False)
# plot_pred_levelvar(ARGS, "OSTEO", "duloxetine_v_nopain", silent=True, force=False)
# make_pred_means(ARGS, "OSTEO", "duloxetine_v_nopain", subtract_guess=True, silent=True, force=False)
# make_pred_means(
#     ARGS, "OSTEO", "duloxetine_v_nopain", subtract_guess=False, silent=True, force=False
# )

plt.show()
sys.exit(0)
# preview_precompute_outpaths(ARGS)
# compute_all_preds_df(ARGS, silent=True)

# preds = compute_all_preds_df(ARGS, silent=False)
# supplement_stat_dfs(preds=preds)

# plot_marchenko(ARGS)
# plot_brody(pathdict, title=f"{dataset_name.upper()} - Brody β", outdir=PLOT_OUTDIRS[dataset_name])
# plot_largest(
#     pathdict,
#     title=f"{dataset_name.upper()} - Largest Eigenvalue",
#     outdir=PLOT_OUTDIRS[dataset_name],
# )


# for pairing in pairings:
#     print(pairing.paired_differences(trim_args=TRIM_IDX, unfold_args=UNFOLD_ARGS))
#     pairing.plot_nnsd(
#         trim_args=TRIM_IDX,
#         unfold_args=UNFOLD_ARGS,
#         title=f"{dataset_name} - NNSD",
#         outdir=PLOT_OUTDIRS[dataset_name],
#     )
#     pairing.plot_rigidity(
#         title=f"{dataset_name}: {pairing.subgroup1} v. {pairing.subgroup2} - Rigidity",
#         outdir=PLOT_OUTDIRS[dataset_name],
#     )
#     pairing.plot_levelvar(
#         title=f"{dataset_name}: {pairing.subgroup1} v. {pairing.subgroup2} - Levelvar",
#         outdir=PLOT_OUTDIRS[dataset_name],
#     )
