import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sbn

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.ensemble import GOE, GDE

from typing import Any, List

from rmt._data_constants import DATA_ROOT, DATASETS, DATASETS_FULLPRE
from rmt._filenames import argstrings_from_args, relpath
from rmt._precompute import precompute_dataset
from rmt._utilities import _percentile_boot
from rmt.args import ARGS
from rmt.comparisons import Pairings
from rmt.summarize import supplement_stat_dfs, compute_all_preds_df

# def plot_marchenko(datapaths: DataSummaryPaths, title: str = None, outdir: Path = None) -> None:
def plot_marchenko(args: Any, show: bool = False) -> None:
    """Create violin plots of the distributions of marchenko noise percents

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    show: bool
        If False (default) just save the plot. Otherwise, call plt.show() and do
        NOT save
    """
    # label is always g1_v_g2, we want "attention" to be orange, "nonattend"
    # to be black
    if not show:
        outdir = DATA_ROOT / "plots"
        os.makedirs(outdir, exist_ok=True)

    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS

    total = 0
    fig, axes = plt.subplots(nrows=3, nocols=8)
    for dataset_name in datasets:
        for groupname, observables in precompute_dataset(dataset_name, args, silent=True).items():
            total += 1  # count how much plots we need

    fig, axes = plt.subplots(nrows=3, nocols=8)
    for dataset_name in datasets:
        dfs, dfs_noise = [], []
        for groupname, observables in precompute_dataset(dataset_name, args, silent=True).items():
            df_full = pd.read_pickle(observables["marchenko"])
            df = pd.DataFrame(df_full.loc["noise_ratio", :])
            df_noise = pd.DataFrame(df_full.loc["noise_ratio_shifted", :])
            n = len(df)
            df["subgroup"] = [groupname for _ in range(n)]
            df_noise["subgroup"] = [groupname for _ in range(n)]
            dfs.append(df)
            dfs_noise.append(df_noise)
        df = pd.concat(dfs)
        df_noise = pd.concat(dfs_noise)

        sbn.set_context("paper")
        sbn.set_style("ticks")
        args_prefix = argstrings_from_args(args)[0]
        dname = dataset_name.upper()
        prefix = f"{dname}_{args_prefix}_{'fullpre_' if args.fullpre else ''}"
        title = f"{dname} - Marchenko Noise Ratio"
        title_shifted = f"{title} (shifted)"
        subtitle = args_prefix.replace("_", " ")

        with sbn.axes_style("ticks"):
            subfontsize = 12
            fontsize = 16
            ax: plt.Axes = sbn.violinplot(x="subgroup", y="noise_ratio", data=df)
            sbn.despine(offset=10, trim=True)
            plt.gcf().suptitle(title, fontsize=fontsize)
            ax.set_title(subtitle, fontdict={"fontsize": subfontsize})
            ax.set_xlabel("Noise Proportion", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Subgroup", fontdict={"fontsize": fontsize})
            if show:
                plt.show()
                plt.close()
            else:
                out = outdir / f"{prefix}marchenko.png"
                plt.gcf().set_size_inches(w=8, h=8)
                plt.savefig(out)
                plt.close()
                print(f"Marchenko plot saved to {relpath(out)}")

            ax = sbn.violinplot(x="subgroup", y="noise_ratio_shifted", data=df_noise)
            sbn.despine(offset=10, trim=True)
            plt.gcf().suptitle(title_shifted, fontsize=fontsize)
            ax.set_title(subtitle, fontdict={"fontsize": subfontsize})
            ax.set_xlabel("Noise Proportion", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Subgroup", fontdict={"fontsize": fontsize})
            if show:
                plt.show()
                plt.close()
            else:
                out = outdir / f"{prefix}marchenko_shifted.png"
                plt.gcf().set_size_inches(w=8, h=8)
                plt.savefig(out)
                plt.close()
                print(f"Marchenko shifted plot saved to {relpath(out)}")


# def plot_brody(datapaths: DataSummaryPaths, title: str = None, outdir: Path = None) -> None:
def plot_brody(args: Any, show: bool = False) -> None:
    """Create violin plots of the distributions of Brody fit parameters

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    show: bool
        If False (default) just save the plot. Otherwise, call plt.show() and do
        NOT save
    """
    if not show:
        outdir = DATA_ROOT / "plots"
        os.makedirs(outdir, exist_ok=True)

    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS
    for dataset_name in datasets:
        dfs = []
        for groupname, observables in precompute_dataset(dataset_name, args, silent=True).items():
            path = observables["brody"]
            df_full = pd.read_pickle(path)
            df = pd.DataFrame(df_full.loc["beta", :])
            n = len(df)
            df["subgroup"] = [groupname for _ in range(n)]
            dfs.append(df)
        df = pd.concat(dfs)

        sbn.set_context("paper")
        sbn.set_style("ticks")
        args_prefix = argstrings_from_args(args)[0]
        dname = dataset_name.upper()
        prefix = f"{dname}_{args_prefix}_{'fullpre_' if args.fullpre else ''}"
        title = f"{dname} - Brody Parameter β"
        subtitle = args_prefix.replace("_", " ")
        with sbn.axes_style("ticks"):
            subfontsize = 12
            fontsize = 16
            ax = sbn.violinplot(x="subgroup", y="beta", data=df)
            sbn.despine(offset=10, trim=True)
            plt.gcf().suptitle(title, fontsize=fontsize)
            ax.set_title(subtitle, fontdict={"fontsize": subfontsize})
            ax.set_xlabel("β", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Subgroup", fontdict={"fontsize": fontsize})
            if show:
                plt.show()
                plt.close()
            else:
                out = outdir / f"{prefix}brody.png"
                plt.gcf().set_size_inches(w=8, h=8)
                plt.savefig(out)
                plt.close()
                print(f"Brody violin plot saved to {relpath(out)}")


# def plot_largest(datapaths: DataSummaryPaths, title: str = None, outdir: Path = None) -> None:
def plot_largest(args: Any, show: bool = False) -> None:
    """Create violin plots of the distributions of the largest eigenvalues

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    show: bool
        If False (default) just save the plot. Otherwise, call plt.show() and do
        NOT save
    """
    if not show:
        outdir = DATA_ROOT / "plots"
        os.makedirs(outdir, exist_ok=True)

    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS
    for dataset_name in datasets:
        dfs = []
        for groupname, observables in precompute_dataset(dataset_name, args, silent=True).items():
            path = observables["largest"]
            df_full = pd.read_pickle(path)
            df = pd.DataFrame(df_full.loc["largest", :], dtype=float)
            n = len(df)
            df["subgroup"] = [groupname for _ in range(n)]
            dfs.append(df)
        df = pd.concat(dfs)

        sbn.set_context("paper")
        sbn.set_style("ticks")
        args_prefix = argstrings_from_args(args)[0]
        dname = dataset_name.upper()
        prefix = f"{dname}_{args_prefix}_{'fullpre_' if args.fullpre else ''}"
        title = f"{dname} - Largest λ"
        subtitle = args_prefix.replace("_", " ")
        with sbn.axes_style("ticks"):
            subfontsize = 12
            fontsize = 16
            ax = sbn.violinplot(x="subgroup", y="largest", data=df)
            sbn.despine(offset=10, trim=True)
            plt.gcf().suptitle(title, fontsize=fontsize)
            ax.set_title(subtitle, fontdict={"fontsize": subfontsize})
            ax.set_title(title)
            ax.set_xlabel("λ_max", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Subgroup", fontdict={"fontsize": fontsize})
            if show:
                plt.show()
                plt.close()
            else:
                out = outdir / f"{prefix}largest.png"
                plt.gcf().set_size_inches(w=8, h=8)
                plt.savefig(out, dpi=300)
                plt.close()
                print(f"Largest λ violin plot saved to {relpath(out)}")


def plot_raw_eigs(args: Any, show: bool = False) -> None:
    """Create violin plots of the distributions of the largest eigenvalues

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    show: bool
        If False (default) just save the plot. Otherwise, call plt.show() and do
        NOT save
    """
    if not show:
        outdir = DATA_ROOT / "plots"
        os.makedirs(outdir, exist_ok=True)

    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS
    for dataset_name, subgroups in datasets.items():
        for subgroup_name, eigpaths in subgroups.items():
            pass

    # for subgroupname, eigpaths in dataset.items():
    for dataset_name in datasets:
        dfs = []
        for groupname, observables in precompute_dataset(dataset_name, args, silent=True).items():
            path = observables["largest"]
            df_full = pd.read_pickle(path)
            df = pd.DataFrame(df_full.loc["largest", :], dtype=float)
            n = len(df)
            df["subgroup"] = [groupname for _ in range(n)]
            dfs.append(df)
        df = pd.concat(dfs)

        sbn.set_context("paper")
        sbn.set_style("ticks")
        args_prefix = argstrings_from_args(args)[0]
        dname = dataset_name.upper()
        prefix = f"{dname}_{args_prefix}_{'fullpre_' if args.fullpre else ''}"
        title = f"{dname} - Largest λ"
        subtitle = args_prefix.replace("_", " ")
        with sbn.axes_style("ticks"):
            subfontsize = 12
            fontsize = 16
            ax = sbn.violinplot(x="subgroup", y="largest", data=df)
            sbn.despine(offset=10, trim=True)
            plt.gcf().suptitle(title, fontsize=fontsize)
            ax.set_title(subtitle, fontdict={"fontsize": subfontsize})
            ax.set_title(title)
            ax.set_xlabel("λ_max", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Subgroup", fontdict={"fontsize": fontsize})
            if show:
                plt.show()
                plt.close()
            else:
                out = outdir / f"{prefix}largest.png"
                plt.gcf().set_size_inches(w=8, h=8)
                plt.savefig(out, dpi=300)
                plt.close()
                print(f"Largest λ violin plot saved to {relpath(out)}")


def plot_pred_nnsd(
    args: Any,
    dataset_name: str,
    comparison: str,
    unfold: List[int] = [5, 7, 9, 11, 13],
    ensembles: bool = True,
    trim: float = 3.0,
    silent: bool = False,
    force: bool = False,
) -> None:
    global ARGS
    # ARGS.fullpre = True
    BINS = np.linspace(0, trim, 20)
    for trim_idx in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim_idx
        all_pairs = []
        for normalize in [False]:
            args.normalize = normalize
            for degree in unfold:
                ARGS.unfold["degree"] = degree
                pairings = Pairings(args, dataset_name)
                pairing = list(filter(lambda p: p.label == comparison, pairings.pairs))
                if len(pairing) != 1:
                    raise ValueError("Too many pairings, something is wrong.")
                all_pairs.append(pairing[0])
        g1, _, g2 = all_pairs[0].label.split("_")  # groupnames
        fig: plt.Figure
        fig, axes = plt.subplots(nrows=1, ncols=len(all_pairs), sharex=True)
        for i, pair in enumerate(all_pairs):
            ax: plt.Axes = axes.flat[i]
            eigs1, eigs2 = pair.eigs1, pair.eigs2
            unf1 = [Eigenvalues(np.load(e)).unfold(**ARGS.unfold) for e in eigs1]
            unf2 = [Eigenvalues(np.load(e)).unfold(**ARGS.unfold) for e in eigs2]
            alpha1, alpha2 = 1 / len(unf1), 1 / len(unf2)
            alpha_adj = 0.01
            alpha1 += alpha_adj
            alpha2 += alpha_adj
            for j, unf in enumerate(unf1):
                spacings = unf.spacings
                if trim > 0.0:
                    spacings = spacings[spacings <= trim]
                # Generate expected distributions for classical ensembles
                sbn.distplot(
                    spacings,
                    norm_hist=True,
                    bins=BINS,
                    kde=False,
                    label=g1 if j == 0 else None,
                    axlabel="spacing (s)",
                    color="#FD8208",
                    hist_kws={"alpha": alpha1},
                    ax=ax,
                )
            for j, unf in enumerate(unf2):
                spacings = unf.spacings
                if trim > 0.0:
                    spacings = spacings[spacings <= trim]
                sbn.distplot(
                    spacings,
                    norm_hist=True,
                    bins=BINS,  # doane
                    kde=False,
                    label=g2 if j == 0 else None,
                    axlabel="spacing (s)",
                    color="#000000",
                    hist_kws={"alpha": alpha2},
                    ax=ax,
                )

            if ensembles:
                s = np.linspace(0, trim, 10000)
                poisson = GDE.nnsd(spacings=s)
                goe = GOE.nnsd(spacings=s)
                sbn.lineplot(x=s, y=poisson, color="#08FD4F", label="Poisson", ax=ax)
                sbn.lineplot(x=s, y=goe, color="#0066FF", label="GOE", ax=ax)
            ax.legend().set_visible(False)
            ax.set_title(f"Unfolding Degree {unfold[i]}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes.flat[-1].legend().set_visible(True)
        fig.text(0.5, 0.04, "L", ha="center", va="center")  # xlabel
        fig.text(
            0.03, 0.5, "∆₃(L)", ha="center", va="center", rotation="vertical", fontdict={"fontname": "DejaVu Sans"}
        )  # ylabel
        fig.set_size_inches(w=12, h=3)
        fig.subplots_adjust(top=0.83, bottom=0.14, left=0.085, right=0.955, hspace=0.2, wspace=0.2)
        plt.suptitle(f"{dataset_name} {ARGS.trim} - Spectral Rigidity")
        plt.show(block=False)


def plot_pred_rigidity(
    args: Any,
    dataset_name: str,
    comparison: str,
    unfold: List[int] = [5, 7, 9, 11, 13],
    ensembles: bool = True,
    silent: bool = False,
    force: bool = False,
) -> None:
    global ARGS
    # ARGS.fullpre = True
    for trim_idx in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim_idx
        all_pairs = []
        for normalize in [False]:
            args.normalize = normalize
            for degree in unfold:
                ARGS.unfold["degree"] = degree
                pairings = Pairings(args, dataset_name)
                pairing = list(filter(lambda p: p.label == comparison, pairings.pairs))
                if len(pairing) != 1:
                    raise ValueError("Too many pairings, something is wrong.")
                all_pairs.append(pairing[0])
        g1, _, g2 = all_pairs[0].label.split("_")  # groupnames
        fig: plt.Figure
        fig, axes = plt.subplots(nrows=1, ncols=len(all_pairs), sharex=True)
        for i, pair in enumerate(all_pairs):
            ax: plt.Axes = axes.flat[i]
            df1 = pd.read_pickle(pair.rigidity[0]).set_index("L")
            df2 = pd.read_pickle(pair.rigidity[1]).set_index("L")
            boots1 = _percentile_boot(df1)
            boots2 = _percentile_boot(df2)
            sbn.lineplot(x=df1.index, y=boots1["mean"], color="#FD8208", label=g1, ax=ax)
            ax.fill_between(x=df1.index, y1=boots1["low"], y2=boots1["high"], color="#FD8208", alpha=0.3)
            sbn.lineplot(x=df2.index, y=boots2["mean"], color="#000000", label=g2, ax=ax)
            ax.fill_between(x=df2.index, y1=boots2["low"], y2=boots2["high"], color="#000000", alpha=0.3)
            if ensembles:
                L = df1.index
                poisson = GDE.spectral_rigidity(L=L)
                goe = GOE.spectral_rigidity(L=L)
                sbn.lineplot(x=L, y=poisson, color="#08FD4F", label="Poisson", ax=ax)
                sbn.lineplot(x=L, y=goe, color="#0066FF", label="GOE", ax=ax)
            ax.legend().set_visible(False)
            ax.set_title(f"Unfolding Degree {unfold[i]}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes.flat[-1].legend().set_visible(True)
        fig.text(0.5, 0.04, "L", ha="center", va="center")  # xlabel
        fig.text(
            0.03, 0.5, "∆₃(L)", ha="center", va="center", rotation="vertical", fontdict={"fontname": "DejaVu Sans"}
        )  # ylabel
        fig.set_size_inches(w=12, h=3)
        fig.subplots_adjust(top=0.83, bottom=0.14, left=0.085, right=0.955, hspace=0.2, wspace=0.2)
        plt.suptitle(f"{dataset_name} {ARGS.trim} - Spectral Rigidity")
        plt.show(block=False)


def plot_pred_levelvar(
    args: Any,
    dataset_name: str,
    comparison: str,
    unfold: List[int] = [5, 7, 9, 11, 13],
    ensembles: bool = True,
    silent: bool = False,
    force: bool = False,
) -> None:
    global ARGS
    # ARGS.fullpre = True
    for trim_idx in ["(1,-1)", "(1,-20)"]:
        ARGS.trim = trim_idx
        all_pairs = []
        for normalize in [False]:
            ARGS.normalize = normalize
            for degree in unfold:
                ARGS.unfold["degree"] = degree
                pairings = Pairings(args, dataset_name)
                pairing = list(filter(lambda p: p.label == comparison, pairings.pairs))
                if len(pairing) != 1:
                    raise ValueError("Too many pairings, something is wrong.")
                all_pairs.append(pairing[0])
        g1, _, g2 = all_pairs[0].label.split("_")  # groupnames
        fig: plt.Figure
        fig, axes = plt.subplots(nrows=1, ncols=len(all_pairs), sharex=True)
        for i, pair in enumerate(all_pairs):
            ax: plt.Axes = axes.flat[i]
            df1 = pd.read_pickle(pair.levelvar[0]).set_index("L")
            df2 = pd.read_pickle(pair.levelvar[1]).set_index("L")
            boots1 = _percentile_boot(df1)
            boots2 = _percentile_boot(df2)
            sbn.lineplot(x=df1.index, y=boots1["mean"], color="#FD8208", label=g1, ax=ax)
            ax.fill_between(x=df1.index, y1=boots1["low"], y2=boots1["high"], color="#FD8208", alpha=0.3)
            sbn.lineplot(x=df2.index, y=boots2["mean"], color="#000000", label=g2, ax=ax)
            ax.fill_between(x=df2.index, y1=boots2["low"], y2=boots2["high"], color="#000000", alpha=0.3)
            if ensembles:
                L = df1.index
                poisson = GDE.spectral_rigidity(L=L)
                goe = GOE.spectral_rigidity(L=L)
                sbn.lineplot(x=L, y=poisson, color="#08FD4F", label="Poisson", ax=ax)
                sbn.lineplot(x=L, y=goe, color="#0066FF", label="GOE", ax=ax)
            ax.legend().set_visible(False)
            ax.set_title(f"Unfolding Degree {unfold[i]}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes.flat[-1].legend().set_visible(True)
        fig.text(0.5, 0.04, "L", ha="center", va="center")  # xlabel
        fig.text(
            0.03,
            0.5,
            r"$\Sigma ^2 \left( L \right)$",
            ha="center",
            va="center",
            rotation="vertical",
            fontdict={"fontname": "DejaVu Sans"},
        )  # ylabel
        fig.set_size_inches(w=12, h=3)
        fig.subplots_adjust(top=0.83, bottom=0.14, left=0.085, right=0.955, hspace=0.2, wspace=0.24)
        plt.suptitle(f"{dataset_name} {ARGS.trim} - Level Number Variance")
        plt.show(block=False)


def make_pred_hists(
    args: Any, density: bool = True, normalize: bool = False, silent: bool = False, force: bool = False
) -> None:
    global ARGS
    # ARGS.fullpre = True
    dfs = []

    def hist_over_trim(trim: str, unfold=[5, 7, 9, 11, 13], normalize=normalize):
        global ARGS

        for trim_idx in [trim]:
            ARGS.trim = trim_idx
            for normalize in [normalize]:
                ARGS.normalize = normalize
                for degree in unfold:
                    ARGS.unfold["degree"] = degree
                    supplemented = supplement_stat_dfs(diffs=None, preds=compute_all_preds_df(args, silent=True))[1]
                    dfs.append(pd.read_csv(supplemented))
        FEATURES = [
            "Raw Eigs",
            # "Largest",
            # "Largest20",
            # "Noise",
            # "Noise (shift)",
            # "Brody",
            # "Rigidity",
            # "Levelvar",
        ]
        HCOLORS = [
            "#777777",  # Raw Eigs
            # "#000000",  # Raw Eigs
            # "#c10000",  # Largest
            # "#a80000",  # Largest20
            # "#06B93A",  # Noise
            # "#058C2C",  # Noise (shift)
            # "#EA00FF",  # brody
            # "#FD8208",  # rigidity
            # "#D97007",  # levelvar
        ]

        orig = pd.concat(dfs)
        hist_info, bins_all, guesses, titles = [], [], [], []
        for dataset in orig["Dataset"].unique():
            by_dataset = orig[orig["Dataset"] == dataset]
            for comparison in by_dataset["Comparison"].unique():
                all_algo_compare = by_dataset[by_dataset["Comparison"] == comparison]
                hist_data = all_algo_compare[FEATURES]
                if len(hist_data) == 0:
                    continue
                hmin, hmax = hist_data.min().min(), hist_data.max().max()
                bins = np.linspace(hmin, hmax, 8)
                hist_info.append(hist_data)
                bins_all.append(bins)
                guesses.append(all_algo_compare["Guess"].iloc[0])
                titles.append(f"{dataset}--{comparison}")
        fig, axes = plt.subplots(nrows=3, ncols=6, sharex=False)
        for i, (hist_data, bins, guess, title) in enumerate(zip(hist_info, bins_all, guesses, titles)):
            sbn.set_style("ticks")
            sbn.set_palette("Accent")
            ax: plt.Axes = axes.flat[i]
            ax.hist(
                hist_data.T,
                # hist_data,
                bins=bins,
                # bins=20,
                stacked=True,
                density=density,
                histtype="bar",
                label=hist_data.columns,
                color=HCOLORS,
            )
            ax.axvline(x=guess, color="black", label="Guess")
            if i == 0:
                ax.legend()
            ax.set_title(title, fontdict={"fontsize": 8})
        fig.text(0.5, 0.04, "Feature Prediction Accuracy", ha="center", va="center")  # xlabel
        fig.text(
            0.1, 0.5, "Density" if density else "Frequency", ha="center", va="center", rotation="vertical"
        )  # ylabel
        f = " (fullpre)" if ARGS.fullpre else ""
        n = " (normed)" if ARGS.normalize else ""
        fig.suptitle(f"Trim {trim} unfolds={unfold}{f}{n}")
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.show(block=False)

    for FULLPRE in [True]:
        # for unfold in [[5, 7], [11, 13]]:
        for unfold in [[5, 7, 9, 11, 13]]:
            hist_over_trim(trim="(1,-1)", normalize=normalize, unfold=unfold)
            hist_over_trim(trim="(1,-20)", normalize=normalize, unfold=unfold)
