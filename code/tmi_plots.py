import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from pathlib import Path
from typing import Any, List
from typing_extensions import Literal

from rmt.args import ARGS
from rmt.summarize import supplement_stat_dfs, compute_all_preds_df


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
UNFOLDS = [5, 7, 9, 11, 13]
N_SUBPLOTS = 17
# fmt: on

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


def make_stacked_accuracy_histograms(
    args: Any,
    features: List[str] = FEATURES,
    fullpre: bool = True,
    density: bool = True,
    normalize: bool = False,
    silent: bool = True,
    force: bool = False,
    nrows: int = 3,
    ncols: int = 6,
    fignum: int = 0,
    savefolder: Path = Path.home() / "Desktop",
    fmt: Literal["svg", "png"] = "png",
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

    density: bool
        Whether the final histogram should be frequency based. If False, will be count based.

    normalize: bool
        If True, normalize features prior to prediction. Only relevant for raw eigenvalues.

    silent: bool
        If True, don't display progress bars.

    force: bool
        If True, recompute all LOOCV accuracies.
    """
    global ARGS
    if nrows * ncols < N_SUBPLOTS:
        raise ValueError(f"Requires `nrows` * `ncols` >= {N_SUBPLOTS}.")
    # ARGS.fullpre = True
    dfs = []

    def hist_over_trim(trim: str, unfold=UNFOLDS, fullpre=fullpre, normalize=normalize):
        global ARGS

        # collect relevant accuracy data
        for trim_idx in [trim]:
            ARGS.trim = trim_idx
            ARGS.normalize = normalize
            for degree in unfold:
                ARGS.unfold["degree"] = degree
                supplemented = supplement_stat_dfs(diffs=None, preds=compute_all_preds_df(args, silent=True))[1]
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
        fig.set_size_inches(w=7.16, h=5)
        for i, (hist_data, bins, guess, title) in enumerate(zip(hist_info, bins_all, guesses, titles)):
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
        fig.text(0.025, 0.5, "Total Density" if density else "Frequency", rotation="vertical", **text)  # ylabel
        fig.text(0.5, 0.99, hist_suptitle(trim, unfold, fullpre, normalize), **text)
        fig.subplots_adjust(top=0.905, bottom=0.105, left=0.065, right=0.975, hspace=0.6, wspace=0.35)
        # plt.show(block=False)
        outfile = savefolder / f"levma{fignum}.{fmt}"
        fig.savefig(outfile, dpi=600, pad_inches=0.0)

    for FULLPRE in [True]:
        # for unfold in [[5, 7], [11, 13]]:
        for unfold in [[5, 7, 9, 11, 13]]:
            hist_over_trim(trim="(1,-1)", normalize=normalize, unfold=unfold)
            # hist_over_trim(trim="(1,-20)", normalize=normalize, unfold=unfold)


if __name__ == "__main__":
    set_tmi_style()
    make_stacked_accuracy_histograms(ARGS)
    plt.show()
    # plt.show()  # This should be after all plotting calls
