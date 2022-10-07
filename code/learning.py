import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sbn

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.ensemble import GOE, Poisson
from glob import glob
from matplotlib.pyplot import Axes
from numpy import ndarray
from pathlib import Path
from tqdm import tqdm

from typing import Any, Dict, List, Union
from typing_extensions import Literal

from rmt.filenames import precomputed_subgroup_paths_from_args
from rmt.precompute import (
    precompute_brody,
    precompute_largest,
    precompute_levelvar,
    precompute_marchenko,
    precompute_rigidity,
)
from rmt._types import Observable
from rmt.utilities import _kde, _percentile_boot

SubjField = Literal["group", "runs"]
Subject = Dict[SubjField, Union[str, ndarray]]
Subjects = Dict[str, Subject]

repro = Path(__file__).resolve().parent
DATA_ROOT = repro / "data"

LEARNING_DATA = DATA_ROOT / "Task_Learning" / "rmt"
os.makedirs(LEARNING_DATA, exist_ok=True)

OUTDIRS = {"SLEPT": "", "NOSLEEP": ""}

PLOT_OUTDIR = LEARNING_DATA / "plots"
SUMMARY_OUTDIR = LEARNING_DATA / "summary"

# fmt: off
TRIM_ARGS     = r"(1,-1)"  # must be indices of trims, tuple, no spaces
UNFOLD_ARGS   = {"smoother": "poly", "degree": 7, "detrend": False}
LEVELVAR_ARGS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.001, "max_L_iters": 50000}
RIGIDITY_ARGS = {"L": np.arange(2, 20, 0.5)}
BRODY_ARGS    = {"method": "mle"}
# fmt: on


def get_subjects_dict() -> Dict[str, Subject]:
    eig_paths: List[Path] = np.sort(
        [
            Path(p).resolve()
            for p in glob(str(LEARNING_DATA.parent) + "/**/*eigs*", recursive=True)
        ]
    )
    subj_ids = np.unique(list(map(lambda p: p.stem.replace("eigs-", "")[:2], eig_paths)))
    subjects: Subjects = {}
    for subj_id in subj_ids:
        subjects[subj_id] = {"group": "", "runs": np.empty([8], dtype=Path)}

    for path in eig_paths:
        group = path.parent.stem
        fname = path.stem.replace("eigs-", "")
        subj_id = fname[:2]
        run_id = int(fname[-1])
        subjects[subj_id]["group"] = group
        subjects[subj_id]["runs"][run_id - 1] = path  # type: ignore
    return subjects


def precompute_subjects(
    subjects: Dict[str, Subject],
    args: Any,
    force_all: bool = False,
    force_largest: bool = False,
    force_marchenko: bool = False,
    force_brody: bool = False,
    force_rigidity: bool = False,
    force_levelvar: bool = False,
) -> Dict[str, Dict[Observable, Path]]:
    raise NotImplementedError("This might not work anymore with the Args refactor!")
    labels = precomputed_subgroup_paths_from_args("LEARNING", "", args)
    summaries: Dict[str, Dict[Observable, Path]] = {}
    for subj_id, subj in subjects.items():
        # the below are checked and good
        largest = SUMMARY_OUTDIR / f"subj-{subj_id}_largest.zip"
        marchenko = SUMMARY_OUTDIR / f"subj-{subj_id}_marchenko.zip"
        brody = SUMMARY_OUTDIR / (f"subj-{subj_id}_" + labels["brody"] + ".zip")
        rigidity = SUMMARY_OUTDIR / (f"subj-{subj_id}_" + labels["rigidity"] + ".zip")
        levelvar = SUMMARY_OUTDIR / (f"subj-{subj_id}_" + labels["levelvar"] + ".zip")
        if force_all:
            force_largest = (
                force_marchenko
            ) = force_brody = force_rigidity = force_levelvar = True

        eigpaths = list(subj["runs"])

        precompute_largest(eigpaths=eigpaths, out=largest, force=force_largest)
        precompute_marchenko(eigpaths=eigpaths, out=marchenko, force=force_marchenko)
        precompute_brody(eigpaths=eigpaths, args=args, out=brody, force=force_brody)
        precompute_rigidity(
            eigpaths=eigpaths, args=args, out=rigidity, force=force_rigidity
        )
        precompute_levelvar(
            eigpaths=eigpaths, args=args, out=levelvar, force=force_levelvar
        )
        summaries[subj_id] = {
            "largest": largest,
            "marchenko": marchenko,
            "brody": brody,
            "rigidity": rigidity,
            "levelvar": levelvar,
        }
    return summaries


def rename_columns(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    renamed = list(map(lambda col: str(re.sub(r"eigs-\d\d_", "", col)), cols))
    # map(lambda col: col.replace("eigs", "subj").replace("_", "_run-"), cols)  # type: ignore
    df.rename(columns=dict(zip(cols, renamed)), inplace=True)


def plot_subject_largest(
    summaries: Dict[str, Dict[Observable, Path]], outfile: Path = None
) -> None:
    sbn.set_context("paper")
    sbn.set_style("ticks", {"ytick.left": False})
    fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
    for i, (subj_id, summary) in tqdm(
        enumerate(summaries.items()), desc="Largest", total=len(summaries)
    ):
        ax: Axes = axes.flat[i]
        df = pd.read_pickle(summary["largest"])
        largest = df.to_numpy(dtype=float).ravel()
        sbn.violinplot(x=largest, ax=ax, color="#919191")
        # sbn.despine(offset=10, trim=True, ax=axes.flat[i])
        ax.set_title(f"subj-{subj_id}")
        ticks = [0, 20000, 40000]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
    fig.suptitle("Per-Subject Distribution of Largest Eigenvalues")
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.04, "Eigenvalue Magnitude", ha="center", va="center")  # xlabel
    fig.text(0.1, 0.5, "Density", ha="center", va="center", rotation="vertical")  # ylabel
    if outfile is None:
        plt.show()
    else:
        fig.savefig(str(outfile.resolve()), dpi=300)
        print(f"Saved largest eigenvalues plot to {str(outfile.relative_to(DATA_ROOT))}")
    plt.close()


def plot_subject_marchenko(
    summaries: Dict[str, Dict[Observable, Path]], outfiles: Dict[str, Path] = None
) -> None:
    for ratio in ["noise_ratio", "noise_ratio_shifted"]:
        sbn.set_context("paper")
        sbn.set_style("ticks", {"ytick.left": False})
        fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
        for i, (subj_id, summary) in tqdm(
            enumerate(summaries.items()), desc="Marchenko", total=len(summaries)
        ):
            ax: Axes = axes.flat[i]
            df = pd.read_pickle(summary["marchenko"])

            noise = df.loc[ratio, :].to_numpy(dtype=float).ravel()
            # sbn.violinplot(x=noise, ax=ax, color="#919191")
            sbn.distplot(
                noise,
                hist=False,
                norm_hist=True,
                kde=True,
                rug=True,
                ax=ax,
                color="#000000",
            )
            # sbn.despine(offset=10, trim=True, ax=axes.flat[i])
            ax.set_title(f"subj-{subj_id}")
            ticks = [0.0, 0.05, 0.10] if ratio == "noise_ratio" else [0.05, 0.10, 0.15]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
        if ratio == "noise_ratio_shifted":
            fig.suptitle("Per-Subject Eigenvalue Noise Ratio (shifted)")
        else:
            fig.suptitle("Per-Subject Eigenvalue Noise Ratios")
        fig.set_size_inches(8, 6)
        fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
        if ratio == "noise_ratio":
            plt.setp(axes, xlim=(-0.02, 0.12), ylim=(0.0, 100))  # better use of space
        else:
            plt.setp(axes, xlim=(0.0, 0.2), ylim=(0.0, 100))  # better use of space
        fig.text(
            0.5, 0.04, "Proportion of Eigenvalues due to Noise", ha="center", va="center"
        )  # xlabel
        fig.text(
            0.05, 0.5, "Density", ha="center", va="center", rotation="vertical"
        )  # ylabel
        if outfiles is None:
            plt.show()
        else:
            fig.savefig(str(outfiles[ratio].resolve()), dpi=300)
            print(
                f"Saved largest eigenvalues plot to {str(outfiles[ratio].relative_to(DATA_ROOT))}"
            )
        plt.close()


def plot_subject_brody(
    summaries: Dict[str, Dict[Observable, Path]], args: Any, outfile: Path = None
) -> None:
    sbn.set_context("paper")
    sbn.set_style("ticks", {"ytick.left": False})
    fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
    for i, (subj_id, summary) in tqdm(
        enumerate(summaries.items()), desc="Brody", total=len(summaries)
    ):
        ax: Axes = axes.flat[i]
        df = pd.read_pickle(summary["brody"])

        beta = df.loc["beta"].to_numpy(dtype=float).ravel()
        sbn.violinplot(x=beta, ax=ax, color="#919191")
        # sbn.despine(offset=10, trim=True, ax=axes.flat[i])
        ax.set_title(f"subj-{subj_id}")
        ticks = [0.0, 0.5, 1.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
    fig.suptitle("Per-Subject Brody Parameters (β)")
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.04, "β", ha="center", va="center")  # xlabel
    fig.text(0.1, 0.5, "Density", ha="center", va="center", rotation="vertical")  # ylabel
    if outfile is None:
        plt.show()
    else:
        fig.savefig(str(outfile.resolve()), dpi=300)
        print(f"Saved Brody plot to {str(outfile.relative_to(DATA_ROOT))}")
    plt.close()


def plot_subject_nnsd(args: Any, n_bins: int = 20, outfile: Path = None) -> None:
    subjects = get_subjects_dict()
    sbn.set_context("paper")
    sbn.set_style("ticks", {"ytick.left": False})
    c1 = "#000000"
    fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
    pbar = tqdm(desc="NNSD", total=8 * len(subjects))
    for i, (subj_id, subject) in enumerate(subjects.items()):
        ax: Axes = axes.flat[i]
        eigpaths = subject["runs"]
        unfoldeds = []
        for path in eigpaths:
            vals = np.load(path)
            if args.trim in ["(1,:)", "", "(0,:)"]:
                vals = vals[1:]  # smallest eigenvalue is always spurious here
            else:
                low, high = eval(args.trim)
                vals = vals[low:high]
            unfoldeds.append(np.sort(Eigenvalues(vals).unfold(**args.unfold).vals))
        all_spacings = [np.diff(unfolded) for unfolded in unfoldeds]
        kde_gridsize = 1000
        kdes = np.empty([len(all_spacings), kde_gridsize], dtype=float)
        s = np.linspace(0, 3, kde_gridsize)
        bins = np.linspace(0, 3, n_bins + 1)
        for j, spacings in enumerate(all_spacings):
            sbn.distplot(
                spacings[(spacings > 0) & (spacings <= 3)],
                norm_hist=True,
                bins=bins,
                kde=False,
                color=c1,
                hist_kws={"alpha": 1.0 / 8, "range": (0.0, 3.0)},
                ax=ax,
            )
            kde = _kde(spacings, grid=s)
            kdes[j, :] = kde
            pbar.update()
            # sbn.lineplot(x=s, y=kde, color=c1, alpha=1.0 / 8, ax=ax)
        # sbn.lineplot(x=s, y=kdes.mean(axis=0), color="#9d0000", label="Mean KDE", ax=ax)
        sbn.lineplot(x=s, y=kdes.mean(axis=0), color=c1, label="Mean KDE", ax=ax)
        sbn.lineplot(
            x=s, y=Poisson.nnsd(spacings=s), color="#08FD4F", label="Poisson", ax=ax
        )
        sbn.lineplot(x=s, y=GOE.nnsd(spacings=s), color="#0066FF", label="GOE", ax=ax)

        ax.set_title(f"subj-{subj_id}")
        ax.set_ylabel("")
        ax.legend(frameon=False, framealpha=0)
        plt.setp(ax.get_legend().get_texts(), fontsize="6")
        # if i != 0:
        handle, labels = ax.get_legend_handles_labels()
        ax._remove_legend(handle)
        ticks = [0, 1, 2, 3]
        ax.set_xticks(ticks)
        # ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
        ax.set_xticklabels(ticks)
    pbar.close()

    fig.suptitle("Per-Subject NNSD")
    fig.set_size_inches(10, 6)
    fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.04, "spacing (s)", ha="center", va="center")  # xlabel
    fig.text(
        0.05, 0.5, "density p(s)", ha="center", va="center", rotation="vertical"
    )  # ylabel
    plt.setp(axes, xlim=(0.0, 3.0), ylim=(0.0, 1.2))  # better use of space
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", frameon=False, framealpha=0, fontsize="8"
    )
    if outfile is None:
        plt.show()
    else:
        fig.savefig(str(outfile.resolve()), dpi=300)
        print(f"Saved NNSD plot to {str(outfile.relative_to(DATA_ROOT))}")
    plt.close()


def plot_subject_rigidity(
    summaries: Dict[str, Dict[Observable, Path]], args: Any, outfile: Path = None
) -> None:
    sbn.set_context("paper")
    sbn.set_style("ticks", {"ytick.left": False})
    c1 = "#000000"
    fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
    for i, (subj_id, summary) in enumerate(summaries.items()):
        ax: Axes = axes.flat[i]
        label = f"subj-{subj_id}"
        rigidities = pd.read_pickle(summary["rigidity"]).set_index("L")
        rename_columns(rigidities)
        L = rigidities.index.to_numpy(dtype=float).ravel()
        for col in rigidities:
            sbn.lineplot(x=L, y=rigidities[col], color=c1, alpha=1.0 / 8.0, ax=ax)
            # sbn.lineplot(x=L, y=rigidities[col], label=f"run-{col}", color=c1, alpha=1.0/8.0, ax=ax)

        boots = _percentile_boot(rigidities, B=2000)
        # sbn.lineplot(x=L, y=boots["mean"], color=c1, label=label)
        sbn.lineplot(x=L, y=boots["mean"], color=c1, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c1, alpha=0.3)
        # plot theoretically-expected curves
        sbn.lineplot(
            x=L, y=Poisson.spectral_rigidity(L=L), color="#08FD4F", label="Poisson", ax=ax
        )
        sbn.lineplot(
            x=L, y=GOE.spectral_rigidity(L=L), color="#0066FF", label="GOE", ax=ax
        )

        ax.set_ylabel("")
        ax.set_title(label)
        ax.legend(frameon=False, framealpha=0)
        plt.setp(ax.get_legend().get_texts(), fontsize="6")
        if i != 0:
            handle, labels = ax.get_legend_handles_labels()
            ax._remove_legend(handle)
        ticks = [0, 10, 20]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
    plt.setp(axes, ylim=(0.0, 1.2))  # better use of space
    fig.suptitle("Per-Subject Task Spectral Rigidities")
    fig.set_size_inches(10, 6)
    fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.04, "L", ha="center", va="center")  # xlabel
    fig.text(
        0.05,
        0.5,
        "∆₃(L)",
        ha="center",
        va="center",
        rotation="vertical",
        fontname="DejaVu Sans",
    )  # ylabel
    if outfile is None:
        plt.show()
    else:
        fig.savefig(str(outfile.resolve()), dpi=300)
        print(f"Saved rigidities plot to {str(outfile.relative_to(DATA_ROOT))}")
    plt.close()


def plot_subject_levelvar(
    summaries: Dict[str, Dict[Observable, Path]], args: Any, outfile: Path = None
) -> None:
    sbn.set_context("paper")
    sbn.set_style("ticks", {"ytick.left": False})
    c1 = "#000000"
    fig, axes = plt.subplots(ncols=6, nrows=4, sharex=True, sharey=True)
    top = []
    for i, (subj_id, summary) in enumerate(summaries.items()):
        ax: Axes = axes.flat[i]
        label = f"subj-{subj_id}"
        levelvars = pd.read_pickle(summary["levelvar"]).set_index("L")
        rename_columns(levelvars)
        L = levelvars.index.to_numpy(dtype=float).ravel()
        for col in levelvars:
            sbn.lineplot(x=L, y=levelvars[col], color=c1, alpha=1.0 / 8.0, ax=ax)
            # sbn.lineplot(x=L, y=levelvars[col], label=f"run-{col}", color=c1, alpha=1.0/8.0, ax=ax)

        boots = _percentile_boot(levelvars, B=2000)
        top.append(np.max(boots["mean"]))
        # sbn.lineplot(x=L, y=boots["mean"], color=c1, label=label)
        sbn.lineplot(x=L, y=boots["mean"], color=c1, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c1, alpha=0.3)
        # plot theoretically-expected curves
        sbn.lineplot(
            x=L, y=Poisson.level_variance(L=L), color="#08FD4F", label="Poisson", ax=ax
        )
        sbn.lineplot(x=L, y=GOE.level_variance(L=L), color="#0066FF", label="GOE", ax=ax)

        ax.set_ylabel("")
        ax.set_title(label)
        ax.legend(frameon=False, framealpha=0)
        plt.setp(ax.get_legend().get_texts(), fontsize="6")
        if i != 0:
            handle, labels = ax.get_legend_handles_labels()
            ax._remove_legend(handle)
        ticks = [1, 5, 10]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, horizontalalignment="right")
    plt.setp(axes, ylim=(0.0, np.percentile(top, 90)))  # better use of space
    fig.suptitle("Per-Subject Task Level Number Variances")
    fig.set_size_inches(10, 6)
    fig.subplots_adjust(left=0.13, bottom=0.15, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.04, "L", ha="center", va="center")  # xlabel
    fig.text(
        0.05,
        0.5,
        "Σ²(L)",
        ha="center",
        va="center",
        rotation="vertical",
        fontname="DejaVu Sans",
    )  # ylabel
    if outfile is None:
        plt.show()
    else:
        fig.savefig(str(outfile.resolve()), dpi=300)
        print(f"Saved levelvar plot to {str(outfile.relative_to(DATA_ROOT))}")
    plt.close()


def plot_subjects(summaries: Dict[str, Dict[Observable, Path]], args: Any) -> None:
    # MARCHENKO_OUTS = {
    #     "noise_ratio": PLOT_OUTDIR / "marchenko.png",
    #     "noise_ratio_shifted": PLOT_OUTDIR / "marchenko_shifted.png",
    # }
    outs = precomputed_subgroup_paths_from_args("LEARNING", "", args)
    # plot_subject_largest(
    #     summaries,,
    #     outfile = PLOT_OUTDIR / "largest.png",
    # )
    # plot_subject_marchenko(summaries, MARCHENKO_OUTS)
    plot_subject_nnsd(args, n_bins=20, outfile=PLOT_OUTDIR / f"{outs['nnsd']}.png")
    plot_subject_rigidity(
        summaries, args, outfile=PLOT_OUTDIR / f"{outs['rigidity']}.png"
    )
    plot_subject_levelvar(
        summaries, args, outfile=PLOT_OUTDIR / f"{outs['levelvar']}.png"
    )


SUBJECTS: Subjects = get_subjects_dict()
# pprint(SUBJECTS)

TRIM_IDX = "(1,-1)"  # must be indices of trims, tuple, no spaces
UNFOLD_ARGS = {"smoother": "poly", "degree": 13, "detrend": False}
LEVELVAR_ARGS = {"L": np.arange(0.5, 10, 0.1), "tol": 0.001, "max_L_iters": 50000}
RIGIDITY_ARGS = {"L": np.arange(2, 20, 0.5)}
BRODY_ARGS = {"method": "mle"}

# yes, this is grotesque, but sometimes you need some damn singletons
# fmt: off
class Args:
    exists = False
    def __init__(self): # noqa
        if _Args.exists: raise RuntimeError("Args object already exists.") # noqa

    @property
    def trim(self): return TRIM_IDX # noqa
    @property
    def unfold(self): return UNFOLD_ARGS # noqa
    @property
    def levelvar(self): return LEVELVAR_ARGS # noqa
    @property
    def rigidity(self): return RIGIDITY_ARGS # noqa
    @property
    def brody(self): return BRODY_ARGS # noqa

ARGS = Args() # noqa
# fmt: on
for unfold_args in [
    {"smoother": "poly", "degree": 3, "detrend": False},
    # {"smoother": "poly", "degree": 5, "detrend": False},
    # {"smoother": "poly", "degree": 9, "detrend": False},
    # {"smoother": "poly", "degree": 11, "detrend": False},
    # {"smoother": "poly", "degree": 13, "detrend": False},
]:
    summaries = precompute_subjects(subjects=SUBJECTS, args=ARGS, force_all=False)

    plot_subjects(summaries, ARGS)
