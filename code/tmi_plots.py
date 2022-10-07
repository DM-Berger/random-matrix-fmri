import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sbn


from warnings import filterwarnings
from pathlib import Path
from typing import Any, List, Union
from typing_extensions import Literal

from rmt.args import ARGS
from rmt.comparisons import Pairings
from rmt.plot_datasets import plot_largest, plot_pred_levelvar
from rmt.summarize import supplement_stat_dfs, compute_all_preds_df
from rmt.utilities import _percentile_boot

from empyricalRMT.brody import brody_dist, fit_brody_mle
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.ensemble import GOE, GDE

Fmt = Literal["svg", "png"]

TMI_FOLDER = Path(__file__).resolve().parent.parent / "paper/paper/TMI/figures"
SUPPLEMENTARY = TMI_FOLDER / "supplementary/figures"
if not TMI_FOLDER.exists():
    os.makedirs(TMI_FOLDER)
if not SUPPLEMENTARY.exists():
    os.makedirs(SUPPLEMENTARY)

# fmt: off
FEATURES = [
    "Raw Eigs",
    "Largest",
    "Largest20",
    "Noise",
    "Noise (shift)",
    "Brody",
    "Rigidity",
    "Levelvar",
]
HCOLORS = [
    "#000000",  # Raw Eigs
    "#c10000",  # Largest
    "#a80000",  # Largest20
    "#06B93A",  # Noise
    "#058C2C",  # Noise (shift)
    "#EA00FF",  # brody
    "#FD8208",  # rigidity
    "#D97007",  # levelvar
]
COLORDICT = dict(zip(FEATURES, HCOLORS))
RAWEIGS_COLOR = ["#777777"]
# fmt: on

UNFOLDS = [5, 7, 9, 11, 13]
N_SUBPLOTS = 17

# see bottom of https://matplotlib.org/tutorials/introductory/customizing.html for all options
FONT_RC = {
    "family": "sans-serif",
    "style": "normal",
    "variant": "normal",  # "small-caps" is other
    "weight": "normal",
    "size": 8.0,
    "serif": "Times New Roman",
    "sans-serif": "Arial",
}
LINES_RC = {"linewidth": 1.0, "markeredgewidth": 0.5}
AXES_RC = {"linewidth": 0.5, "titlesize": "medium"}
PATCHES_RC = {"linewidth": 0.5}
TEXT_RC = {}
FIGWIDTH = 7.16  # inches


def set_tmi_style():
    mpl.rc("lines", **LINES_RC)
    mpl.rc("patch", **PATCHES_RC)
    mpl.rc("font", **FONT_RC)
    mpl.rc("axes", **AXES_RC)


def shorten_title(title: str) -> str:
    REPLACEMENTS = {
        "LEARNING": "LEARN",
        "duloxetine": "dulox",
        "PSYCH_": "PSY-",
        "VIGILANCE": "VIG",
        "TASK_ATTENTION": "TASK_ATT",
        "WEEKLY_ATTENTION": "WEEK_ATT",
        "SES-": "s",
        "PARKINSONS": "PARK",
        "parkinsons": "park",
        "control": "ctrl",
        "SUMMED": "SUM",
        "INTERLEAVED": "INTER",
        "--": "\n",
    }
    for long, short in REPLACEMENTS.items():
        title = title.replace(long, short)
    return title


def hist_suptitle(trim: str, unfold: List[str], fullpre: bool, normalize: bool) -> str:
    trim1 = "trimming largest eigenvalue"
    trim20 = "trimming 20 largest eigenvalues"
    t = trim1 if trim == "(1,-1)" else trim20
    u = str(unfold).replace("[", "{").replace("]", "}")
    f = " (fullpre)" if fullpre else ""
    n = " (normed)" if normalize else ""
    return f"Accuracies across all classifiers and unfolding degrees {u}, {t}{f}{n}"


def rigidity_suptitle(
    dataset_name: str, trim: str, fullpre: bool, normalize: bool
) -> str:
    trim1 = "trimming largest eigenvalue"
    trim20 = "trimming 20 largest eigenvalues"
    t = trim1 if trim == "(1,-1)" else trim20
    f = " (fullpre)" if fullpre else ""
    n = " (normed)" if normalize else ""
    return f"{shorten_title(dataset_name)} Spectral Rigidity - {t}{f}{n}"


def make_plot(
    fig: plt.Figure, show: bool, fmt: Union[Fmt, List[Fmt]], fignum: str
) -> None:
    if show:
        plt.show(block=False)
    else:
        if not isinstance(fmt, list):
            fmt = [fmt]
        for f in fmt:
            savefolder = SUPPLEMENTARY if fignum.lower()[0] == "s" else TMI_FOLDER
            outfile = savefolder / f"levma{fignum.replace('s', '')}.{f}"
            fig.savefig(outfile, dpi=600, pad_inches=0.0)
            print(f"Plot saved to {outfile}")
        plt.close()


def make_stacked_accuracy_histograms(
    args: Any,
    features: List[str] = FEATURES,
    fullpre: bool = True,
    trim: Literal["1", "20"] = "1",
    unfolds: List[int] = UNFOLDS,
    density: bool = True,
    normalize: bool = False,
    silent: bool = True,
    force: bool = False,
    nrows: int = 3,
    ncols: int = 6,
    fignum: str = "0",
    fmt: Union[Fmt, List[Fmt]] = "png",
    show: bool = False,
) -> None:
    """Generate the multi-part figure where each subplot is a dataset, and those subplots contain
    the stacked histograms of mean LOOCV accuracies across classifiers, unfoldings, and trimmings.

    Parameters
    ----------
    args: Args
       Ppass in ARGS here. (Yes, this is really ugly).

    features: List[str]
        Which features to include in the histograms

    fullpre: bool
        Whether or not to generate histograms for the fully preprocessed data.

    trim: int
        If "1", trim largest only. If "20", trim largest 20 eigenvalues.

    density: bool
        Whether the final histogram should be frequency based. If False, will be count based.

    normalize: bool
        If True, normalize features prior to prediction. Only relevant for raw eigenvalues.

    silent: bool
        If True, don't display progress bars.

    force: bool
        If True, recompute all LOOCV accuracies.

    fignum: str
        Figure number as appearing in paper. If prepended with "s", is saved to supplementary.
    """
    global ARGS
    if nrows * ncols < N_SUBPLOTS:
        raise ValueError(f"Requires `nrows` * `ncols` >= {N_SUBPLOTS}.")
    ARGS.fullpre = fullpre
    dfs = []

    def hist_over_trim(
        trim: str, unfold=UNFOLDS, fullpre=fullpre, normalize=normalize, fmt=fmt
    ):
        global ARGS

        # collect relevant accuracy data
        # we need to use and modify the global ARGS because I did this poorly
        for trim_idx in [trim]:
            ARGS.trim = trim_idx
            ARGS.normalize = normalize
            for degree in unfold:
                ARGS.unfold["degree"] = degree
                supplemented = supplement_stat_dfs(
                    diffs=None, preds=compute_all_preds_df(args, silent=True)
                )[1]
                dfs.append(pd.read_csv(supplemented))

        # Pre-assemble some plotting labels, bin sizes, "Guess" line info
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
                titles.append(shorten_title(f"{dataset}--{comparison}"))

        if len(features) == 1 and features[0] == "Raw Eigs":
            hcolors = RAWEIGS_COLOR
        else:
            hcolors = [COLORDICT[f] for f in features]
        fig: plt.Figure
        axes: plt.Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False)
        fig.set_size_inches(w=FIGWIDTH, h=5)
        for i, (hist_data, bins, guess, title) in enumerate(
            zip(hist_info, bins_all, guesses, titles)
        ):
            sbn.set_style("ticks")
            sbn.set_palette("Accent")
            ax: plt.Axes = axes.flat[i]
            ax.hist(
                hist_data.T,
                bins=bins,
                stacked=True,
                density=density,
                histtype="bar",
                label=hist_data.columns,
                color=hcolors,
                linewidth=0.5,
            )
            ax.axvline(x=guess, color="black", label="Guess")
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
            ax.set_title(title, fontdict={"fontsize": 8})
        # tidy up, adjust, add legend
        for i in range(N_SUBPLOTS, nrows * ncols):
            fig.delaxes(axes.flat[i])  # remove last plots, regardless of nrows, ncols
        text = dict(ha="center", va="center", fontsize=8)
        fig.legend(handles, labels, loc=[0.84, 0.08], fontsize=8, labelspacing=0.4)
        fig.text(0.5, 0.04, "Feature Prediction Accuracy", **text)  # xlabel
        fig.text(
            0.025,
            0.5,
            "Total Density" if density else "Frequency",
            rotation="vertical",
            **text,
        )  # ylabel
        fig.text(0.5, 0.99, hist_suptitle(trim, unfold, fullpre, normalize), **text)
        fig.subplots_adjust(
            top=0.905, bottom=0.105, left=0.065, right=0.975, hspace=0.6, wspace=0.35
        )
        make_plot(fig, show, fmt, fignum)

    if trim not in ["1", "20"]:
        raise ValueError("Invalid trim.")
    hist_over_trim(
        trim=f"(1,-{trim})", fullpre=fullpre, normalize=normalize, unfold=unfolds
    )


def plot_pred_rigidity(
    args: Any,
    dataset_name: str,
    comparison: str,
    fullpre: bool = True,
    unfold: List[int] = [5, 7, 9, 11, 13],
    ensembles: bool = True,
    silent: bool = True,
    force: bool = False,
    fignum: str = "0",
    fmt: Union[Fmt, List[Fmt]] = "png",
    show: bool = False,
) -> None:
    global ARGS
    ARGS.fullpre = fullpre
    # for trim_idx in ["(1,-1)", "(1,-20)"]:
    for trim_idx in ["(1,-1)"]:
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
            sbn.lineplot(x=df2.index, y=boots2["mean"], color="#000000", label=g2, ax=ax)
            ax.fill_between(
                x=df2.index,
                y1=boots2["low"],
                y2=boots2["high"],
                color="#000000",
                alpha=0.3,
            )
            sbn.lineplot(x=df1.index, y=boots1["mean"], color="#FD8208", label=g1, ax=ax)
            ax.fill_between(
                x=df1.index,
                y1=boots1["low"],
                y2=boots1["high"],
                color="#FD8208",
                alpha=0.3,
            )
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
        axes.flat[0].legend(fontsize=8, labelspacing=0.2, framealpha=0.0).set_visible(
            True
        )
        text = dict(ha="center", va="center", fontsize=8)
        fig.text(0.5, 0.04, "L", **text)  # xlabel
        fig.text(
            0.03,
            0.5,
            "Spectral Rigidity, ∆₃(L)",
            rotation="vertical",
            fontdict={"fontname": "DejaVu Sans"},
            **text,
        )  # ylabel
        fig.set_size_inches(w=7, h=1.5)
        fig.subplots_adjust(
            top=0.88, bottom=0.215, left=0.09, right=0.95, hspace=0.2, wspace=0.32
        )
        # plt.suptitle(f"{dataset_name} {ARGS.trim} - Spectral Rigidity")
        # fig.text(0.5, 0.97, rigidity_suptitle(dataset_name, ARGS.trim, fullpre, normalize), **text)  # suptitle
        make_plot(fig, show, fmt, fignum)


def plot_pred_levelvar(
    args: Any,
    dataset_name: str,
    comparison: str,
    unfold: List[int] = [5, 7, 9, 11, 13],
    ensembles: bool = True,
    show: bool = True,
    fmt: str = "png",
    fignum: str = "0",
    silent: bool = False,
    force: bool = False,
) -> None:
    global ARGS
    # ARGS.fullpre = True
    # for trim_idx in ["(1,-1)", "(1,-20)"]:
    for trim_idx in ["(1,-1)"]:
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
            ax.fill_between(
                x=df1.index,
                y1=boots1["low"],
                y2=boots1["high"],
                color="#FD8208",
                alpha=0.3,
            )
            sbn.lineplot(x=df2.index, y=boots2["mean"], color="#000000", label=g2, ax=ax)
            ax.fill_between(
                x=df2.index,
                y1=boots2["low"],
                y2=boots2["high"],
                color="#000000",
                alpha=0.3,
            )
            if ensembles:
                L = df1.index
                poisson = GDE.spectral_rigidity(L=L)
                goe = GOE.spectral_rigidity(L=L)
                sbn.lineplot(x=L, y=poisson, color="#08FD4F", label="Poisson", ax=ax)
                # sbn.lineplot(x=L, y=goe, color="#0066FF", label="GOE", ax=ax, fmt="--")
                ax.plot(L, goe, "--", color="#0066FF", label="GOE")
            ax.legend().set_visible(False)
            ax.set_title(f"Unfolding Degree {unfold[i]}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes.flat[0].legend(fontsize=8, labelspacing=0.2, framealpha=0.0).set_visible(
            True
        )
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
        fig.set_size_inches(w=7, h=1.5)
        fig.subplots_adjust(
            top=0.88, bottom=0.215, left=0.09, right=0.95, hspace=0.2, wspace=0.32
        )
        make_plot(fig, show, fmt, fignum)
        # plt.suptitle(f"{dataset_name} {ARGS.trim} - Level Number Variance")


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
    # for trim_idx in ["(1,-1)", "(1,-20)"]:
    for trim_idx in ["(1,-1)"]:
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
        fig, axes = plt.subplots(
            nrows=1, ncols=len(all_pairs), sharex=True, squeeze=False
        )
        for i, (pair, unfold_degree) in enumerate(zip(all_pairs, unfold)):
            ax: plt.Axes = axes.flat[i]
            eigs1, eigs2 = pair.eigs1, pair.eigs2
            unfold_args = {**ARGS.unfold, **dict(degree=unfold_degree)}
            unf1 = [Eigenvalues(np.load(e)).unfold(**unfold_args) for e in eigs1]
            unf2 = [Eigenvalues(np.load(e)).unfold(**unfold_args) for e in eigs2]
            alpha1, alpha2 = 1 / len(unf1), 1 / len(unf2)
            # alpha_adj = 0.02  # good for just plotting hists, no brody
            alpha_adj = 0.00
            alpha1 += alpha_adj
            alpha2 += alpha_adj

            for j, unf in enumerate(unf1):
                spacings = unf.spacings
                if trim > 0.0:
                    spacings = spacings[spacings <= trim]
                beta = fit_brody_mle(spacings)
                brody = brody_dist(spacings, beta)
                # Generate expected distributions for classical ensembles
                sbn.distplot(
                    spacings,
                    norm_hist=True,
                    bins=BINS,
                    kde=False,
                    # label=g1 if j == 0 else None,
                    axlabel="spacing (s)",
                    color="#FD8208",
                    # hist_kws={"alpha": alpha1, "histtype": "step", "linewidth": 0.5},
                    hist_kws={"alpha": alpha1},
                    # kde_kws={"alpha": alpha1, "color":"#FD8208"},
                    ax=ax,
                )
                sbn.lineplot(
                    x=spacings,
                    y=brody,
                    color="#FD8208",
                    ax=ax,
                    alpha=0.9,
                    label=g1 if j == 0 else None,
                    linewidth=0.5,
                )

            for j, unf in enumerate(unf2):
                spacings = unf.spacings
                if trim > 0.0:
                    spacings = spacings[spacings <= trim]
                beta = fit_brody_mle(spacings)
                brody = brody_dist(spacings, beta)
                sbn.distplot(
                    spacings,
                    norm_hist=True,
                    bins=BINS,  # doane
                    kde=False,
                    # label=g2 if j == 0 else None,
                    axlabel="spacing (s)",
                    color="#000000",
                    # hist_kws={"alpha": alpha2, "histtype": "step", "linewidth": 0.5},
                    hist_kws={"alpha": alpha2},
                    # kde_kws={"alpha": alpha2, "color":"#000000"},
                    ax=ax,
                )
                sbn.lineplot(
                    x=spacings,
                    y=brody,
                    color="#000000",
                    ax=ax,
                    alpha=0.9,
                    label=g2 if j == 0 else None,
                    linewidth=0.5,
                )

            if ensembles:
                s = np.linspace(0, trim, 10000)
                poisson = GDE.nnsd(spacings=s)
                goe = GOE.nnsd(spacings=s)
                sbn.lineplot(
                    x=s, y=poisson, color="#08FD4F", label="Poisson", ax=ax, alpha=0.5
                )
                sbn.lineplot(x=s, y=goe, color="#0066FF", label="GOE", ax=ax, alpha=0.5)
            ax.legend().set_visible(False)
            ax.set_title(f"Unfolding Degree {unfold[i]}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        axes.flat[0].legend().set_visible(True)
        fig.text(0.5, 0.04, "spacing (s)", ha="center", va="center")  # xlabel
        fig.text(
            0.03, 0.5, "p(s)", ha="center", va="center", rotation="vertical"
        )  # ylabel
        fig.set_size_inches(w=7, h=1.5)  # TMI full-page max width is 7 inches
        # fig.set_size_inches(w=3.5, h=3.5)  # TMI half-page max width is 3.5 inches
        fig.subplots_adjust(
            top=0.83, bottom=0.2, left=0.075, right=0.955, hspace=0.2, wspace=0.23
        )
        # fontdic = {"fontname": "Arial", "fontsize": 10.0}
        # fig.suptitle(f"{dataset_name} {ARGS.trim} - NNSD", fontdict=fontdic)
        make_plot(fig, show=False, fmt="png", fignum="9")
        # plt.show(block=False)


if __name__ == "__main__":
    set_tmi_style()
    filterwarnings("ignore", category=RuntimeWarning)
    filterwarnings("ignore", category=FutureWarning)
    # make_stacked_accuracy_histograms(ARGS, trim="1", fignum="1", fmt=["png", "svg"])
    # make_stacked_accuracy_histograms(ARGS, trim="20", fignum="s1", fmt=["png", "svg"])
    # plot_pred_rigidity(ARGS, "OSTEO", "duloxetine_v_nopain", show=False, fignum="2")
    plot_pred_levelvar(
        ARGS, "OSTEO", "duloxetine_v_nopain", ensembles=True, show=False, fignum="2"
    )
    # plot_pred_rigidity(ARGS, "PARKINSONS", "control_v_parkinsons", ensembles=True, silent=True, force=False)
    # plot_pred_levelvar(ARGS, "PARKINSONS", "control_v_parkinsons", ensembles=True, silent=True, force=False)
    # plot_pred_nnsd(ARGS, "OSTEO", "duloxetine_v_nopain", trim=4.0, ensembles=True, silent=True, force=False)
    plt.show()  # This should be after all plotting calls
