import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sbn

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.ensemble import GOE, Poisson
from numpy import ndarray
from pathlib import Path
from pandas import DataFrame
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Tuple

from rmt._data_constants import (
    DATASETS,
    DATASETS_FULLPRE,
    DUDS,
    PAIRED_DATA,
    STAT_OUTDIRS,
    STAT_OUTDIRS_FULLPRE,
)
from rmt._types import Observable
from rmt._filenames import _prefix, precomputed_subgroup_paths_from_args, relpath, stats_fnames
from rmt._utilities import _percentile_boot, _configure_sbn_style, _cohen_d


def _trimmed_from_args(vals: ndarray, trim_args: str) -> ndarray:
    """Parse `trim_args` and return the appropriate values from `vals`"""
    if trim_args in ["(1,:)", "", "(0,:)"]:
        vals = vals[1:]  # smallest eigenvalue is always spurious here
    else:
        low, high = eval(trim_args)
        vals = vals[low:high]
    return vals


def group_differences_summary(g1: ndarray, g2: ndarray, rowname: str, dataset_name: str, comparison: str) -> DataFrame:
    """Given ndarrays of scalar RMT features `g1` and `g2`, compute various
    descriptive statistics and return them as a DataFrame.

    Parameters
    ----------
    rowname: str
        The str to use for the data row label (index).

    dataset_name: str
        The Dataset key name (just used here as a label).

    comparison: str
        The label for the comparison.

    Returns
    -------
    df: DataFrame
        The summary DataFrame.
    """
    is_paired = dataset_name.lower() in PAIRED_DATA  # e.g. for paired t-tests outputs = None
    g1, g2 = np.ravel(g1), np.ravel(g2)

    nans = (np.nan, np.nan)
    t_paired, p_paired = ttest_rel(g1, g2) if is_paired else nans
    t_ind, p_ind = ttest_ind(g1, g2, equal_var=False) if not is_paired else nans
    W, pw = wilcoxon(g1, g2) if is_paired else nans
    try:
        U, pu = mannwhitneyu(g1, g2, alternative="two-sided")
    except ValueError as e:
        if str(e.args).find("All numbers are identical"):
            U, pu = np.nan, np.nan
        else:
            raise e
    auc = U / (len(g1) * len(g2))

    outputs = pd.DataFrame(
        data={
            "DATASET": dataset_name.upper(),
            "Comparison": comparison,
            "μ₁": np.mean(g1),
            "μ₂": np.mean(g2),
            "d": _cohen_d(g1, g2),
            "t-test (paired)": t_paired,
            "p (paired)": p_paired,
            "t-test (Welch's)": t_ind,
            "p (Welch's)": p_ind,
            "Wilcoxon": W,
            "p (Wilcoxon)": pw,
            "Mann-Whitney U": U,
            "p (Mann-Whitney)": pu,
            "AUC": auc,
        },
        index=[rowname],
    )
    return outputs


class Pair:
    """A class for bundling together data for each comparison.

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    dataset_name: str
        The Dataset name (e.g. the key value for indexing into _data_constants.DATASETS).

    label: str
        The label string for the comparison (e.g. 'task_v_rest').

    eigs1: List[Path]
        The list of paths to the raw eigenvalues of subgroup 1.

    eigs2: List[Path]
        The list of paths to the raw eigenvalues of subgroup 2.

    subgroup1: str
        The name of the first SUBGROUP being analyzed (not dataset name). E.g. the
        top-level key values defined in _data_constants.get_all_filepath_groupings().

    subgroup2: str
        The name of the second SUBGROUP being analyzed (not dataset name). E.g. the
        top-level key values defined in _data_constants.get_all_filepath_groupings().

    rigidity: Pair = Tuple[Path, Path]
        Tuple of paths to the precomputed unfolded feature values. (Will need an
        `args` later.)

    levelvar: Pair = Tuple[Path, Path]
        Tuple of paths to the precomputed unfolded feature values. (Will need an
        `args` later.)

    brody: Pair = Tuple[Path, Path]
        Tuple of paths to the precomputed unfolded feature values. (Will need an
        `args` later.)

    marchenko: Pair = Tuple[Path, Path]
        Tuple of paths to the precomputed unfolded feature values. (Will need an
        `args` later.)
    """

    def __init__(
        self,
        args: Any,
        dataset_name: str,
        label: str,
        eigs1: List[Path],
        eigs2: List[Path],
        subgroup1: str,
        subgroup2: str,
        rigidity: Tuple[Path, Path],
        levelvar: Tuple[Path, Path],
        brody: Tuple[Path, Path],
        marchenko: Tuple[Path, Path],
    ):
        if len(eigs1) == 0:
            raise ValueError(f"eigs1 argument for dataset {dataset_name} and pairing {label} has no eigenvalues.")
        if len(eigs2) == 0:
            raise ValueError(f"eigs2 argument for dataset {dataset_name} and pairing {label} has no eigenvalues.")
        self.args = args
        self.dataset_name = dataset_name
        self.label = label
        self.eigs1 = eigs1
        self.eigs2 = eigs2
        self.subgroup1 = subgroup1
        self.subgroup2 = subgroup2
        self.rigidity = rigidity
        self.levelvar = levelvar
        self.brody = brody
        self.marchenko = marchenko

    def __getitem__(self, key: Observable) -> Tuple[Path, Path]:
        if key == "rigidity":
            return self.rigidity
        elif key == "levelvar":
            return self.levelvar
        elif key == "brody":
            return self.brody
        elif key == "marchenko":
            return self.marchenko
        else:
            raise ValueError("Can only index into Pairing via observable name.")

    def _predict_file(self, normalize: bool) -> Path:
        """Get the path of the predictions based on args and normalize."""
        comparison = self.label
        dataset_name = self.dataset_name
        fname, _ = stats_fnames(self.args, normalize, "pickle")
        stat_outdirs = STAT_OUTDIRS_FULLPRE if self.args.fullpre else STAT_OUTDIRS
        parent = stat_outdirs[dataset_name]
        outfile = parent / f"{comparison}__{fname}"
        return outfile

    def _diff_file(self) -> Path:
        """Get the path of the scalar feature differences based on args and normalize."""
        comparison = self.label
        dataset_name = self.dataset_name
        _, fname = stats_fnames(self.args, False, "pickle")
        stat_outdirs = STAT_OUTDIRS_FULLPRE if self.args.fullpre else STAT_OUTDIRS
        parent = stat_outdirs[dataset_name]
        outfile = parent / f"{comparison}__{fname}"
        return outfile

    def _get_formatted_data(self) -> Dict[str, Tuple[DataFrame, DataFrame]]:
        """Load all data based on self.args, and reformat for ML classifiers."""
        trim_args = self.args.trim
        unf_args = self.args.unfold
        # see if the raw eigs alone are more useful than RMT stats
        eigs1 = [np.load(p) for p in self.eigs1]
        eigs2 = [np.load(p) for p in self.eigs2]

        unfolded1 = DataFrame(
            data=[Eigenvalues(_trimmed_from_args(eigs, trim_args)).unfold(**unf_args).vals for eigs in eigs1]
        )
        unfolded2 = DataFrame(
            data=[Eigenvalues(_trimmed_from_args(eigs, trim_args)).unfold(**unf_args).vals for eigs in eigs2]
        )
        l1, l2 = np.min([len(eigs) for eigs in eigs1]), np.min([len(eigs) for eigs in eigs2])
        l_shared = np.min([l1, l2])
        eigs1 = [eigs[-l_shared:] for eigs in eigs1]  # use largest eigenvalues only
        eigs2 = [eigs[-l_shared:] for eigs in eigs2]
        eigs1, eigs2 = DataFrame(data=np.array(eigs1)), DataFrame(data=np.array(eigs2))

        largest1 = DataFrame([np.load(p).max() for p in self.eigs1])
        largest2 = DataFrame([np.load(p).max() for p in self.eigs2])
        largest20_1 = DataFrame([np.load(p)[-20:] for p in self.eigs1])
        largest20_2 = DataFrame([np.load(p)[-20:] for p in self.eigs2])
        noise1 = pd.read_pickle(self.marchenko[0]).loc["noise_ratio", :].T
        noise2 = pd.read_pickle(self.marchenko[1]).loc["noise_ratio", :].T
        noise_shifted1 = pd.read_pickle(self.marchenko[0]).loc["noise_ratio_shifted", :].T
        noise_shifted2 = pd.read_pickle(self.marchenko[1]).loc["noise_ratio_shifted", :].T
        brody1 = pd.read_pickle(self.brody[0]).loc["beta"].T
        brody2 = pd.read_pickle(self.brody[1]).loc["beta"].T
        rig1 = pd.read_pickle(self.rigidity[0]).set_index("L").T  # must be (n_samples, n_features)
        rig2 = pd.read_pickle(self.rigidity[1]).set_index("L").T
        var1 = pd.read_pickle(self.levelvar[0]).set_index("L").T
        var2 = pd.read_pickle(self.levelvar[1]).set_index("L").T

        return {
            "Raw Eigs": (eigs1, eigs2),
            "Unfolded": (unfolded1, unfolded2),
            "Largest": (largest1, largest2),
            "Largest20": (largest20_1, largest20_2),
            "Noise": (noise1, noise2),
            "Noise (shift)": (noise_shifted1, noise_shifted2),
            "Brody": (brody1, brody2),
            "Rigidity": (rig1, rig2),
            "Levelvar": (var1, var2),
        }

    def _to_X_y(self, g1: Any, g2: Any, normalize: bool = False) -> Tuple[ndarray, ndarray]:
        X = pd.concat([g1, g2])
        if len(X.shape) == 1:
            X = X.to_numpy().ravel()
            X = X.reshape(-1, 1)
        if normalize:
            X = MinMaxScaler().fit_transform(X)
        # g1_size, g2_size = len(self.eigs1), len(self.eigs2)
        g1_size, g2_size = g1.shape[0], g2.shape[0]
        labels = [self.subgroup1 for _ in range(g1_size)] + [self.subgroup2 for _ in range(g2_size)]
        y = np.array(labels).ravel()
        return X, y

    def _lda_classify(self, g1: Any, g2: Any, normalize: bool = False) -> np.float64:
        X, y = self._to_X_y(g1, g2, normalize)
        try:
            lda = LDA(solver="lsqr").fit(X, y)
            # return np.mean(cross_val_score(LR, X, y, cv=X.shape[0] - 1))
            return np.mean(cross_val_score(lda, X, y, cv=LeaveOneOut()))
        except ValueError:
            return np.nan

    def _logistic_regression_classify(self, g1: Any, g2: Any, normalize: bool = False) -> np.float64:
        X, y = self._to_X_y(g1, g2, normalize)
        try:
            LR = LogisticRegression(solver="liblinear", max_iter=500).fit(X, y)
            # return np.mean(cross_val_score(LR, X, y, cv=X.shape[0] - 1))
            return np.mean(cross_val_score(LR, X, y, cv=LeaveOneOut()))
        except ValueError:
            return np.nan

    def _svm_classify(self, g1: Any, g2: Any, normalize: bool = False) -> np.float64:
        X, y = self._to_X_y(g1, g2, normalize)
        try:
            SVM = SVC(gamma="scale")
            # return np.mean(cross_val_score(SVM, X, y, cv=X.shape[0] - 1))
            return np.mean(cross_val_score(SVM, X, y, cv=LeaveOneOut()))
        except ValueError:
            return np.nan

    def _knn_classify(self, g1: Any, g2: Any, n_neighbours: int, normalize: bool = True) -> np.float64:
        X, y = self._to_X_y(g1, g2, normalize)
        try:
            knn = KNN(n_neighbors=n_neighbours, n_jobs=-1)
            # return np.mean(cross_val_score(knn, X, y, cv=X.shape[0] - 1))
            return np.mean(cross_val_score(knn, X, y, cv=LeaveOneOut()))
        except ValueError:
            return np.nan

    def paired_all_observable_predicts(
        self, dataset_name: str, knns=[2, 3, 4, 5, 6, 7, 8, 9, 10], use_eigs=False
    ) -> Tuple[Optional[DataFrame], DataFrame]:
        # see if the raw eigs alone are more useful than RMT stats
        # eigs = None
        if use_eigs:
            eigs1 = [np.load(p) for p in self.eigs1]
            eigs2 = [np.load(p) for p in self.eigs2]
            l1, l2 = np.min([len(eigs) for eigs in eigs1]), np.min([len(eigs) for eigs in eigs2])
            l_shared = np.min([l1, l2])
            eigs1 = [eigs[-l_shared:] for eigs in eigs1]  # use largest eigenvalues only
            eigs2 = [eigs[-l_shared:] for eigs in eigs2]
            eigs1, eigs2 = DataFrame(data=np.array(eigs1)), DataFrame(data=np.array(eigs2))
            # eigs = pd.concat([eigs1, eigs2])

        largest1 = np.array([np.load(p).max() for p in self.eigs1]).ravel()
        largest2 = np.array([np.load(p).max() for p in self.eigs2]).ravel()
        if len(largest1) > len(largest2):
            largest1 = largest1.reshape(len(largest2), -1)
            largest2 = largest2.reshape(-1, 1)
        else:
            largest2 = largest2.reshape(len(largest1), -1)
            largest1 = largest1.reshape(-1, 1)
        noise1 = pd.read_pickle(self.marchenko[0]).loc["noise_ratio", :].T
        noise2 = pd.read_pickle(self.marchenko[1]).loc["noise_ratio", :].T
        noise_shifted1 = pd.read_pickle(self.marchenko[0]).loc["noise_ratio_shifted", :].T
        noise_shifted2 = pd.read_pickle(self.marchenko[1]).loc["noise_ratio_shifted", :].T
        brody1 = pd.read_pickle(self.brody[0]).loc["beta"].T
        brody2 = pd.read_pickle(self.brody[1]).loc["beta"].T
        rig1 = pd.read_pickle(self.rigidity[0]).set_index("L").T  # must be (n_samples, n_features)
        rig2 = pd.read_pickle(self.rigidity[1]).set_index("L").T
        var1 = pd.read_pickle(self.levelvar[0]).set_index("L").T
        var2 = pd.read_pickle(self.levelvar[1]).set_index("L").T

        largest = np.hstack([largest1, largest2])
        noise = pd.concat([noise1, noise2]).to_numpy(dtype=np.float64)
        noise_shifted = pd.concat([noise_shifted1, noise_shifted2]).to_numpy(dtype=np.float64)
        brody = pd.concat([brody1, brody2]).to_numpy(dtype=np.float64)
        rig, var = pd.concat([rig1, rig2]), pd.concat([var1, var2]).to_numpy(dtype=np.float64)

        for arr in [largest, noise, noise_shifted, brody, rig, var]:
            print(arr.shape)
        X = np.vstack([largest, noise, noise_shifted, brody, rig, var])
        labels = [self.subgroup1 for _ in self.eigs1] + [self.subgroup2 for _ in self.eigs2]
        y = np.array(labels).ravel()

        LR = LogisticRegression(n_jobs=-1).fit(X, y)
        SVM = SVC(gamma="scale")
        print("Computing Logistic Regression")
        log_score = np.mean(cross_val_score(LR, X, y, cv=LeaveOneOut()))
        print("Computing SVM")
        svm_score = np.mean(cross_val_score(SVM, X, y, cv=LeaveOneOut()))
        knn_scores = np.zeros([len(knns)])
        for i, n_neighbors in tqdm(enumerate(knns), desc="KNN Predicts", total=len(knns)):
            knn = KNN(n_neighbors=n_neighbors, n_jobs=-1)
            knn_scores[i] = np.mean(cross_val_score(knn, X, y, cv=LeaveOneOut()))

        knns = [f"KNN-{k}" for k in knns]
        df = DataFrame(index=["Logistic Regression", "SVM"] + knns, dtype=float)
        rnd = 2
        guess = np.max([len(self.eigs1), len(self.eigs2)]) / (len(self.eigs1) + len(self.eigs2))

        df.loc["Logistic Regression", "Observables"] = np.round(log_score, rnd)
        df.loc["Logistic Regression", "Guess"] = np.round(guess, rnd)
        df.loc["SVM", "Observables"] = np.round(svm_score, rnd)
        df.loc["SVM", "Guess"] = np.round(guess, rnd)
        for i, knn in enumerate(knns):
            df.loc[knn, "Observables"] = np.round(knn_scores[i], rnd)
        return df

    def paired_predicts(
        self,
        args: Any,
        logistic: bool = True,
        knns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        normalize: bool = False,
        force: bool = False,
        silent: bool = False,
    ) -> Optional[DataFrame]:
        outfile = self._predict_file(normalize)
        if not force and outfile.exists():
            return pd.read_pickle(outfile)

        # compute descriptive statistics
        data = self._get_formatted_data()

        knn_keys = [f"KNN-{k}" for k in knns]
        non_knns = ["Logistic Regression", "LDA", "SVM"] if logistic else ["LDA", "SVM"]
        index = non_knns + knn_keys
        preds = DataFrame(index=pd.Index(index, name="Algorithm"), dtype=float)
        # rnd = 3
        # add useful metadata in first columns
        for key in index:
            preds.loc[key, "Dataset"] = self.dataset_name.upper()
            preds.loc[key, "Comparison"] = self.label

        if logistic:
            label = "Logistic Regression"
            for predictor, (g1, g2) in tqdm(data.items(), total=len(data), desc=label, disable=silent):
                preds.loc[label, predictor] = self._logistic_regression_classify(g1, g2, normalize)

        for predictor, (g1, g2) in tqdm(data.items(), total=len(data), desc="LDA", disable=silent):
            preds.loc["LDA", predictor] = self._lda_classify(g1, g2, normalize)

        for predictor, (g1, g2) in tqdm(data.items(), total=len(data), desc="SVM", disable=silent):
            preds.loc["SVM", predictor] = self._svm_classify(g1, g2, normalize)

        for predictor, (g1, g2) in data.items():
            for k in tqdm(knns, total=len(knns), desc=f"{predictor} KNN", disable=silent):
                preds.loc[f"KNN-{k}", predictor] = self._knn_classify(g1, g2, k, normalize)

        # show guess to beat in last column
        guess = np.max([len(self.eigs1), len(self.eigs2)]) / (len(self.eigs1) + len(self.eigs2))
        for key in index:
            preds.loc[key, "Guess"] = guess

        if not outfile.parent.exists():
            os.makedirs(outfile.parent, exist_ok=True)
        preds.to_pickle(outfile)
        print(f"Saved predictions to {relpath(outfile)}")

        return preds

    def paired_differences(self, force: bool = False) -> DataFrame:
        outfile = self._diff_file()
        if not force and outfile.exists():
            return pd.read_pickle(outfile)

        dataset_name = self.dataset_name
        data = self._get_formatted_data()
        largest_diffs = group_differences_summary(*data["Largest"], "Largest Eigenvalue", dataset_name, self.label)
        noise_diffs = group_differences_summary(*data["Noise"], "Noise Ratio", dataset_name, self.label)
        noise_shifted_diffs = group_differences_summary(
            *data["Noise (shift)"], "Shifted Noise Ratio", dataset_name, self.label
        )
        brody_diffs = group_differences_summary(*data["Brody"], "β", dataset_name, self.label)
        diffs = pd.concat([largest_diffs, noise_diffs, noise_shifted_diffs, brody_diffs], sort=False)

        if not outfile.parent.exists():
            os.makedirs(outfile.parent, exist_ok=True)

        diffs.to_pickle(outfile)
        print(f"Saved diffs to {relpath(outfile)}")
        return diffs

    def plot_nnsd(
        self,
        trim_args: str,
        unfold_args: dict,
        n_bins: int = 20,
        max_spacing: float = 5.0,
        title: str = None,
        outdir: Path = None,
    ) -> None:
        # label is always g1_v_g2, we want "attention" to be orange, "nonattend"
        # to be black
        if self.label in ["rest_v_task", "nopain_v_pain", "control_v_control_pre", "park_pre_v_parkinsons"]:
            c1, c2 = "#000000", "#FD8208"
        elif self.label in [
            "allpain_v_nopain",
            "allpain_v_duloxetine",
            "high_v_low",
            "control_pre_v_park_pre",
            "control_v_parkinsons",
        ]:
            c1, c2 = "#FD8208", "#000000"
        elif self.label in ["control_pre_v_parkinsons", "control_v_park_pre"]:
            return  # meaningless comparison
        else:
            c1, c2 = "#EA00FF", "#FD8208"

        unfolded1 = []
        for path in self.eigs1:
            vals = np.load(path)
            if trim_args in ["(1,:)", "", "(0,:)"]:
                vals = vals[1:]  # smallest eigenvalue is always spurious here
            else:
                low, high = eval(trim_args)
                vals = vals[low:high]
            unfolded1.append(np.sort(Eigenvalues(vals).unfold(**unfold_args).vals))

        unfolded2 = []
        for path in self.eigs2:
            vals = np.load(path)
            if trim_args in ["(1,:)", "", "(0,:)"]:
                vals = vals[1:]  # smallest eigenvalue is always spurious here
            else:
                low, high = eval(trim_args)
                vals = vals[low:high]
            unfolded2.append(np.sort(Eigenvalues(vals).unfold(**unfold_args).vals))

        spacings1 = [np.diff(unfolded) for unfolded in unfolded1]
        spacings2 = [np.diff(unfolded) for unfolded in unfolded2]
        # trim largest histogram skewing spacing
        spacings1 = [spacings[spacings < max_spacing] for spacings in spacings1]
        spacings2 = [spacings[spacings < max_spacing] for spacings in spacings2]
        mean_brody1 = np.round(float(pd.read_pickle(self.brody[0]).mean(axis=1)), 2)
        mean_brody2 = np.round(float(pd.read_pickle(self.brody[1]).mean(axis=1)), 2)

        _configure_sbn_style()
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

        # plot curves for each group
        for spacings in spacings1:
            sbn.distplot(
                spacings,
                norm_hist=True,
                bins=n_bins,
                kde=True,
                axlabel="spacing (s)",
                color=c1,
                kde_kws={"alpha": np.max([0.1, 1 / len(spacings1)])},
                hist_kws={"alpha": np.max([0.1, 1 / len(spacings1)])},
                ax=axes[0],
            )
        for spacings in spacings2:
            sbn.distplot(
                spacings,
                norm_hist=True,
                bins=n_bins,
                kde=True,
                axlabel="spacing (s)",
                color=c2,
                kde_kws={"alpha": np.max([0.1, 1 / len(spacings1)])},
                hist_kws={"alpha": np.max([0.1, 1 / len(spacings2)])},
                ax=axes[1],
            )

        # plot bootstrapped means and CIs for each group
        # boots = _percentile_boot(df1)
        # sbn.lineplot(x=L, y=boots["mean"], color=c1, label=self.subgroup1, ax=ax)
        # ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c1, alpha=0.3)

        # boots = _percentile_boot(df2)
        # sbn.lineplot(x=L, y=boots["mean"], color=c2, label=self.subgroup2, ax=ax)
        # ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c2, alpha=0.3)

        # plot theoretically-expected curves
        s = np.linspace(0, max_spacing, 5000)
        sbn.lineplot(x=s, y=Poisson.nnsd(spacings=s), color="#08FD4F", label="Poisson", ax=axes[0])
        sbn.lineplot(x=s, y=Poisson.nnsd(spacings=s), color="#08FD4F", label="Poisson", ax=axes[1])
        sbn.lineplot(x=s, y=GOE.nnsd(spacings=s), color="#0066FF", label="GOE", ax=axes[0])
        sbn.lineplot(x=s, y=GOE.nnsd(spacings=s), color="#0066FF", label="GOE", ax=axes[1])

        # ensure all plots have identical axes
        axes[0].set_ylim(top=2.0)
        axes[1].set_ylim(top=2.0)

        # ax.legend().set_visible(True)
        fig.suptitle(f"{self.label}: NNSD" if title is None else title)
        for i, ax in enumerate(axes):
            ax.set_xlabel("spacing (s)")
            ax.set_ylabel("density p(s)", fontname="DejaVu Sans")
            ax.set_title(
                f"{self.subgroup1 if i == 0 else self.subgroup2} (<β> = {mean_brody1 if i == 0 else mean_brody2})"
            )
        if outdir is None:
            plt.show()
            plt.close()
        else:
            os.makedirs(outdir, exist_ok=True)
            prefix = _prefix(trim_args, unfold_args)
            outfile = outdir / f"{prefix}_{self.label}_nnsd.png"
            fig.set_size_inches(10, 5)
            plt.savefig(outfile, dpi=300)
            plt.close()
            print(f"Pooled nnsd plot saved to {relpath(outfile)}")

    def plot_rigidity(self, title: str = None, outdir: Path = None) -> None:
        # label is always g1_v_g2, we want "attention" to be orange, "nonattend"
        # to be black
        if self.label in ["rest_v_task", "nopain_v_pain", "control_v_control_pre", "park_pre_v_parkinsons"]:
            c1, c2 = "#000000", "#FD8208"
        elif self.label in [
            "allpain_v_nopain",
            "allpain_v_duloxetine",
            "high_v_low",
            "control_pre_v_park_pre",
            "control_v_parkinsons",
        ]:
            c1, c2 = "#FD8208", "#000000"
        else:
            c1, c2 = "#EA00FF", "#FD8208"
        df1 = pd.read_pickle(self.rigidity[0]).set_index("L")
        df2 = pd.read_pickle(self.rigidity[1]).set_index("L")
        L = np.array(df1.index)
        _configure_sbn_style()
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()

        # plot curves for each group
        for col in df1:
            sbn.lineplot(x=L, y=df1[col], color=c1, alpha=0.05, ax=ax)
        for col in df2:
            sbn.lineplot(x=L, y=df2[col], color=c2, alpha=0.05, ax=ax)

        # plot bootstrapped means and CIs for each group
        boots = _percentile_boot(df1)
        sbn.lineplot(x=L, y=boots["mean"], color=c1, label=self.subgroup1, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c1, alpha=0.3)

        boots = _percentile_boot(df2)
        sbn.lineplot(x=L, y=boots["mean"], color=c2, label=self.subgroup2, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c2, alpha=0.3)

        # plot theoretically-expected curves
        sbn.lineplot(x=L, y=Poisson.spectral_rigidity(L=L), color="#08FD4F", label="Poisson", ax=ax)
        sbn.lineplot(x=L, y=GOE.spectral_rigidity(L=L), color="#0066FF", label="GOE", ax=ax)

        ax.legend().set_visible(True)
        ax.set_title(f"{self.label}: Rigidity" if title is None else title)
        ax.set_xlabel("L")
        ax.set_ylabel("∆₃(L)", fontname="DejaVu Sans")
        fig.set_size_inches(w=8, h=8)
        if outdir is None:
            plt.show()
            plt.close()
        else:
            os.makedirs(outdir, exist_ok=True)
            outfile = outdir / f"{self.rigidity[0].stem}_{self.label}.png"
            fig.savefig(outfile, dpi=300)
            plt.close()
            print(f"Pooled rigidity plots saved to {relpath(outfile)}")

    def plot_levelvar(self, title: str = None, outdir: Path = None) -> None:
        # label is always g1_v_g2, we want "attention" to be orange, "nonattend"
        # to be black
        if self.label in ["rest_v_task", "nopain_v_pain", "control_v_control_pre", "park_pre_v_parkinsons"]:
            c1, c2 = "#000000", "#FD8208"
        elif self.label in [
            "allpain_v_nopain",
            "allpain_v_duloxetine",
            "high_v_low",
            "control_pre_v_park_pre",
            "control_v_parkinsons",
        ]:
            c1, c2 = "#FD8208", "#000000"
        else:
            c1, c2 = "#EA00FF", "#FD8208"
        df1 = pd.read_pickle(self.levelvar[0]).set_index("L")
        df2 = pd.read_pickle(self.levelvar[1]).set_index("L")
        L = np.array(df1.index)
        _configure_sbn_style()
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()

        # plot curves for each group
        for col in df1:
            sbn.lineplot(x=L, y=df1[col], color=c1, alpha=0.05, ax=ax)
        for col in df2:
            sbn.lineplot(x=L, y=df2[col], color=c2, alpha=0.05, ax=ax)

        # plot bootstrapped means and CIs for each group
        boots = _percentile_boot(df1)
        sbn.lineplot(x=L, y=boots["mean"], color=c1, label=self.subgroup1, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c1, alpha=0.3)

        boots = _percentile_boot(df2)
        sbn.lineplot(x=L, y=boots["mean"], color=c2, label=self.subgroup2, ax=ax)
        ax.fill_between(x=L, y1=boots["low"], y2=boots["high"], color=c2, alpha=0.3)

        # plot theoretically-expected curves
        # sbn.lineplot(x=L, y=Poisson.level_variance(L=L), color="#08FD4F", label="Poisson", ax=ax)
        sbn.lineplot(x=L, y=GOE.level_variance(L=L), color="#0066FF", label="GOE", ax=ax)

        ax.legend().set_visible(True)
        ax.set_title(f"{self.label}: Level Variance" if title is None else title)
        ax.set_xlabel("L")
        ax.set_ylabel("Σ²(L)", fontname="DejaVu Sans")
        fig.set_size_inches(w=8, h=8)
        if outdir is None:
            plt.show()
            plt.close()
        else:
            os.makedirs(outdir, exist_ok=True)
            outfile = outdir / f"{self.levelvar[0].stem}_{self.label}.png"
            plt.savefig(outfile, dpi=300)
            plt.close()
            print(f"Pooled levelvar plots saved to {relpath(outfile)}")

    def print(self) -> None:
        def short(p: Path) -> str:
            return f"[..]/{p.parent.name}/{p.name}"

        print(f"{self.label}")
        print(f"   eigs1:     {short(self.eigs1[0])}")
        print(f"   eigs2:     {short(self.eigs2[0])}")
        print(f"   rigidity:  {short(self.rigidity[0])} v. {short(self.rigidity[1])}")
        print(f"   levelvar:  {short(self.levelvar[0])} v. {short(self.levelvar[1])}")
        print(f"   brody:     {short(self.brody[0])} v. {short(self.brody[1])}")
        print(f"   marchenko: {short(self.marchenko[0])} v. {short(self.marchenko[1])}")

    def format(self) -> str:
        def short(p: Path) -> str:
            return f"[..]/{p.parent.name}/{p.name}"

        s = []

        s.append(f"{self.label}")
        s.append(f"   eigs1:     {short(self.eigs1[0])}")
        s.append(f"   eigs2:     {short(self.eigs2[0])}")
        s.append(f"   rigidity:  {short(self.rigidity[0])} v. {short(self.rigidity[1])}")
        s.append(f"   levelvar:  {short(self.levelvar[0])} v. {short(self.levelvar[1])}")
        s.append(f"   brody:     {short(self.brody[0])} v. {short(self.brody[1])}")
        s.append(f"   marchenko: {short(self.marchenko[0])} v. {short(self.marchenko[1])}")
        return "\n".join(s)


class Pairings:
    """A helper class for bundling together data for each comparison.

    NOTE: Constructor only really intended to be called from "pairings_from_precomputed".

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    dataset_name: str
        The Dataset name (e.g. the key value for indexing into _data_constants.DATASETS).

    Returns
    -------
    val1: Any
    """

    def __init__(self, args: Any, dataset_name: str):
        self.args: Any = args
        self.dataset_name: str = dataset_name
        self.pairs: List[Pair] = self.pairings_from_precomputed()

    def pairings_from_precomputed(self) -> List[Pair]:
        """Generate a list of the subgrup pairings for dataset `dataset_name`.

        Parameters
        ----------
        dataset_name: str
            The Dataset name (e.g. the key value for indexing into _data_constants.DATASETS).

        args: Args
            Contains the unfolding, trimming, normalization, etc options defined in
            run.py

        Returns
        -------
        pairings: List[Pairing]
        """
        args = self.args
        dataset_name = self.dataset_name
        summary_paths: Dict[str, Dict[Observable, Path]] = {}

        dataset = DATASETS_FULLPRE[dataset_name] if args.fullpre else DATASETS[dataset_name]
        for groupname in dataset:
            summary_paths[groupname] = precomputed_subgroup_paths_from_args(dataset_name, groupname, args)
        pairings = []
        for subgroupname1, observables1 in summary_paths.items():
            for subgroupname2, observables2 in summary_paths.items():
                if subgroupname1 >= subgroupname2:
                    continue
                label = f"{subgroupname1}_v_{subgroupname2}"
                if label in DUDS:
                    continue
                pairing = {"label": label, "subgroup1": subgroupname1, "subgroup2": subgroupname2}
                observables = ["rigidity", "levelvar", "brody", "marchenko"]
                for obs in observables:
                    pairing[obs] = (observables1[obs], observables2[obs])  # type: ignore
                pairings.append(
                    Pair(
                        args=self.args,
                        dataset_name=dataset_name,
                        label=label,
                        eigs1=dataset[subgroupname1],
                        eigs2=dataset[subgroupname2],
                        subgroup1=subgroupname1,
                        subgroup2=subgroupname2,
                        rigidity=pairing["rigidity"],  # type: ignore
                        levelvar=pairing["levelvar"],  # type: ignore
                        brody=pairing["brody"],  # type: ignore
                        marchenko=pairing["marchenko"],  # type: ignore
                    )
                )
        return pairings
