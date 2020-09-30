import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from typing import Any, List

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
# fmt: on


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
    # ARGS.fullpre = True
    dfs = []

    def hist_over_trim(trim: str, unfold=UNFOLDS, normalize=normalize):
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
                titles.append(f"{dataset}--{comparison}")

        if len(features) == 1 and features[0] == "Raw Eigs":
            hcolors = RAWEIGS_COLOR
        else:
            hcolors = [COLORDICT[f] for f in features]
        fig: plt.Figure
        axes: plt.Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False)
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
            )
            ax.axvline(x=guess, color="black", label="Guess")
            if i == 0:
                # ax.legend()
                handles, labels = ax.get_legend_handles_labels()
            ax.set_title(title, fontdict={"fontsize": 8})

        # tidy up, adjust, add legend
        fig.delaxes(axes[nrows - 1][ncols - 1])  # remove last plot
        fig.legend(handles, labels, loc=[0.8, 0.11], fontsize=8, labelspacing=0.4)
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
            # hist_over_trim(trim="(1,-20)", normalize=normalize, unfold=unfold)


if __name__ == "__main__":
    make_stacked_accuracy_histograms(ARGS)
    plt.show()  # This should be after all plotting calls
