# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import re
import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from empyricalRMT.eigenvalues import Eigenvalues, Unfolded
from empyricalRMT.observables.levelvariance import level_number_variance
from empyricalRMT.observables.rigidity import spectral_rigidity
from empyricalRMT.smoother import SmoothMethod
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import mode
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.preprocess.unify_attention_data import get_comparison_df, median_split_labels

CACHE_DIR = ROOT.parent / "__OBSERVABLES_CACHE__"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
MEMOIZER = Memory(location=str(CACHE_DIR))
L_VALUES = np.arange(1.0, 21.0, step=1.0)

"""
Park_v_Control/
Rest_v_LearningRecall/
Rest_w_Bilinguality/
Rest_w_Depression_v_Control/
Rest_w_Healthy_v_OsteoPain/
Rest_w_Older_v_Younger/
Rest_w_VigilanceAttention/
"""


def parse_source(source: Path) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    sid_ = re.search(r"sub-(?:[a-zA-Z]*)?([0-9]+)_.*", source.stem)
    ses = re.search(r"ses-([0-9]+)_.*", source.stem)
    run_ = re.search(r"run-([0-9]+)_.*", source.stem)
    data_ = re.search(r"data/updated/(.*)/ds00.*", str(source))

    if sid_ is None:
        raise RuntimeError(f"Didn't find an SID in filename {source}")
    sid = sid_[1]
    session = ses[1] if ses is not None else None
    run = run_[1] if run_ is not None else None
    data = data_[1] if data_ is not None else None
    return sid, session, run, data


class UpdatedProcessedDataset:
    def __init__(self, source: UpdatedDataset, preproc_level: PreprocLevel) -> None:
        self.source = source
        self.preproc_level = preproc_level
        self.info: DataFrame = self.get_information_frame()
        # self.full_pre = full_pre
        # data = PATH_DATA_PRE if self.full_pre else PATH_DATA
        # self.path_info = data[self.source]
        self.id = f"{self.source.name}__preproc={self.preproc_level.name}"

    def get_information_frame(self) -> DataFrame:
        """Get a DataFrame linking each path to all phenotypic info"""
        files = self.source.eig_files(self.preproc_level)
        dfs = []

        if self.source is UpdatedDataset.Older:
            for file in files:
                sid, session, run, _ = parse_source(file)
                label = "younger" if sid.startswith("10") else "older"
                df = DataFrame(
                    {"sid": sid, "label": label, "session": session, "run": run},
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            return df

        if self.source is UpdatedDataset.Parkinsons:
            for file in files:
                sid, session, run, _ = parse_source(file)
                label = "ctrl" if "41" in sid else "park"  # 41=ctrl, 42=park
                df = DataFrame(
                    {"sid": sid, "label": label, "session": session, "run": run},
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            if len(df.label.unique()) == 1:
                raise ValueError(f"Labeling issue for {self.source}")
            return df

        table_path: Path = self.source.participants_file()  # type: ignore
        if "sv" in table_path.suffix:  # non-vigilance cases
            sep = "\t" if table_path.suffix == ".tsv" else ","
            table = pd.read_csv(table_path, sep=sep)
            table["sid"] = table["participant_id"].apply(lambda s: s.replace("sub-", ""))
            table.drop(columns="participant_id", inplace=True)
        else:  #
            table = pd.read_json(table_path)

        if self.source is UpdatedDataset.Learning:
            """Table looks like:
            participant_id  age sex  group
                    sub-01   38   M  sleep
                    sub-02   30   F   wake
                    sub-03   32   F   wake
                    sub-04   29   F  sleep
            """
            for file in files:
                sid, session, run, _ = parse_source(file)
                label = "rest" if "task-rest" in str(file) else "task"
                group = table.loc[table["sid"] == sid]["group"]

                df = DataFrame(
                    {
                        "sid": sid,
                        "label": label,
                        "group": group,
                        "session": session,
                        "run": run,
                    },
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            return df

        """Table looks like:
        participant_id group sex  age  ethnicity  years_education  eng_proficiency_score
            sub-2975    MC   F   21          1               14                   10.0
            sub-3156    LB   M   22          1               14                   10.0
            sub-3225    LB   M   25          1               18                   10.0

        sp_proficiency_score  num_lang
                            5.0       3.0
                            NaN       4.0
                            8.0       2.0
        group:
            MC = monolingual control
            EB = "Early-acquisition bilingual"
            LB = "Late-acquisition bilingual"
        """
        if self.source is UpdatedDataset.Bilinguality:
            for file in files:
                sid, session, run, _ = parse_source(file)
                group = table.loc[table["sid"] == sid]["group"].item()
                label = "monolingual" if group == "MC" else "bilingual"

                df = DataFrame(
                    {
                        "sid": sid,
                        "label": label,
                        "session": session,
                        "run": run,
                    },
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            return df

        """
        Osteo data table:

        sid      type  study  gender      Age        Drug  ...
         01   control    NaN  female       59         NaN
         02   control    NaN    male       59         NaN
         03   control    NaN    male       52         NaN
         04   control    NaN    male       48         NaN
         05   control    NaN    male       78         NaN
         ..       ...    ...     ...      ...         ...  ...
         72   patient    2.0    male       49     placebo
         73   patient    2.0    male       73     placebo
         74   patient    2.0  female       63     placebo
         75   patient    2.0  female       70     placebo
         76   patient    2.0    male       64  duloxetine  ...
        """

        if self.source is UpdatedDataset.Osteo:
            labels = {
                "control+nan": "nopain",
                "patient+placebo": "pain",
                "patient+duloxetine": "duloxetine",
            }
            for file in files:
                sid, session, run, _ = parse_source(file)
                group = table.loc[table["sid"] == sid]["type"].item()
                drug = str(table.loc[table["sid"] == sid]["Drug"].item())
                label = labels[f"{group}+{drug}"]

                df = DataFrame(
                    {
                        "sid": sid,
                        "label": label,
                        "session": session,
                        "run": run,
                    },
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            return df

        """
        Depression data table looks like:

        sid     age gender    group  IQ_Raven ICD-10  MADRS  Zung_SDS   BDI
        sub-01   39      m     depr     113.0  F32.0    NaN      43.0  17.0
        sub-02   50      m     depr      80.0  F32.0    NaN      47.0  10.0
        sub-03   47      f     depr      87.0  F32.0    NaN      44.0  19.0
        sub-52   32      m  control     128.0    NaN    NaN      38.0  13.0
        sub-53   44      f  control     102.0    NaN    NaN      38.0   2.0
        """
        if self.source is UpdatedDataset.Depression:
            for file in files:
                sid, session, run, _ = parse_source(file)
                group = str(table.loc[table["sid"] == sid]["group"].item())
                label = "depress" if group == "depr" else "control"

                df = DataFrame(
                    {
                        "sid": sid,
                        "label": label,
                        "session": session,
                        "run": run,
                    },
                    index=[file],
                )
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            if len(df.label.unique()) == 1:
                raise ValueError(f"Bad table construction\n{df}")
            return df

        # now handle Vigilance data
        if "Weekly" in self.source.name:
            attention = "weekly"
        elif "Vigil" in self.source.name:
            attention = "vigilance"
        elif "Task" in self.source.name:
            attention = "task"
        else:
            raise RuntimeError("Impossible")
        if "Ses" in self.source.name:
            session = self.source.name[-1]
        else:
            session = None
        # table = get_comparison_df(table, attention, session=session)  # type: ignore
        # table["sid"] = table["sid"].apply(lambda s: f"{s:02d}")
        labeled = median_split_labels(attention=attention, session=session)

        dfs = []
        for file in files:
            sid, session, run, _ = parse_source(file)
            session = f"ses-{int(session)}"
            sid_idx = labeled["sid"] == sid
            idx = sid_idx
            if session is not None:
                ses_idx = labeled["session"] == session
                idx &= ses_idx
            label = str(labeled.loc[idx]["label"].item())

            df = DataFrame(
                {
                    "sid": sid,
                    "label": label,
                    "session": session,
                    "run": run,
                },
                index=[file],
            )
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        return df

    def labels(self) -> ndarray:
        return cast(ndarray, self.info["label"].to_numpy())

    def eigs(self) -> list[ndarray]:
        return list(map(lambda p: np.load(p), self.info.index))

    def eigs_df(
        self, unify: Literal["pad", "percentile"] = "pad", diff: bool = False
    ) -> DataFrame:
        raw = self.eigs()
        if diff:
            raw = [np.diff(r) for r in raw]
        lengths = np.array([len(e) for e in raw])
        if unify == "pad" and not np.all(lengths == lengths[0]):
            # front zero-pad
            length = np.max(lengths)
            resized = []
            for eig in raw:
                padded = np.zeros(length)
                padded[-len(eig) :] = eig
                resized.append(padded)
        elif unify == "percentile":
            resized = []
            for eig in raw:
                resized.append(np.percentile(eig, np.linspace(0, 1, 100)))
        else:
            resized = raw
        vals = np.stack(resized, axis=0)
        vals[vals < 0.0] = 0.0
        eigs = DataFrame(vals, columns=range(vals.shape[1]))
        eigs["y"] = self.labels()
        return eigs

    def trimmed(self, trim_method: TrimMethod | None) -> list[Eigenvalues]:
        if trim_method is None:
            eigs: list[Eigenvalues] = [Eigenvalues(e) for e in self.eigs()]
            return eigs

        to_hash = (
            self.id,
            trim_method.name if trim_method is not None else "none",
        )
        hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
        outfile = CACHE_DIR / f"{hsh}.npz"
        if outfile.exists():
            vals: list[ndarray] = [*np.load(outfile).values()]
            return [Eigenvalues(val) for val in vals]

        eigs: list[Eigenvalues] = [Eigenvalues(e) for e in self.eigs()]
        eigs = [trim(e, trim_method) for e in eigs]
        vals = [e.vals for e in eigs]
        np.savez_compressed(outfile, *vals)
        return eigs

    def unfolded(
        self, smoother: SmoothMethod, degree: int, trim_method: TrimMethod | None
    ) -> list[Unfolded]:
        eigs = self.trimmed(trim_method=trim_method)
        unfs = [eig.unfold(smoother=smoother, degree=degree) for eig in eigs]
        return unfs

    def unfolded_df(self, degree: int, trim_method: TrimMethod | None) -> DataFrame:
        unfoldeds = self.unfolded(
            smoother=SmoothMethod.Polynomial, degree=degree, trim_method=trim_method
        )
        unfs = [u.vals for u in unfoldeds]
        lengths = np.array([len(u) for u in unfs])  # type: ignore
        # front zero-pad
        length = np.max(lengths)
        resized = []
        for unf in unfs:
            padded = np.zeros(length)
            padded[-len(unf) :] = unf  # type: ignore
            resized.append(padded)
        vals = np.stack(resized, axis=0)
        vals[vals < 0.0] = 0.0
        df = DataFrame(vals, columns=range(vals.shape[1]))
        df["y"] = self.labels()
        return df

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(source={self.source.name}, preproc={self.preproc_level})"
        )


@dataclass
class ObservableArgs:
    unfolded: ndarray
    L: ndarray


def _compute_rigidity(args: ObservableArgs) -> ndarray | None:
    """helper for `process_map`"""
    try:
        return spectral_rigidity(unfolded=args.unfolded, L=args.L, show_progress=False)[1]  # type: ignore # noqa
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


def _compute_levelvar(args: ObservableArgs) -> ndarray | None:
    """helper for `process_map`"""
    try:
        unfolded = args.unfolded
        return level_number_variance(unfolded=unfolded, L=args.L, show_progress=False)[1]  # type: ignore # noqa
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


def precision_trim(eigs: ndarray) -> ndarray:
    # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html  # noqa
    # see https://netlib.org/lapack/lug/node89.html
    # see https://netlib.org/lapack/lug/node90.html
    eigs = np.copy(eigs)
    eps = np.finfo(np.float64).eps
    emax = eigs.max()
    cut = max(np.sqrt(emax) * eps * len(eigs), emax * eps)
    idx = eigs > cut
    eigs = eigs[idx]
    return eigs  # type: ignore


def kmeans_trim(eigs: ndarray, k: int = 3, log: bool = False) -> ndarray:
    e = eigs.reshape(-1, 1)
    e = np.log(e) if log else e
    labels = KMeans(k).fit(e).labels_
    most = mode(labels, keepdims=False).mode
    idx = labels == most
    eigs = np.copy(eigs)
    eigs[~idx] = np.nan
    return eigs


def trim(eigenvalues: Eigenvalues, method: TrimMethod) -> Eigenvalues:
    eigs = precision_trim(eigenvalues.values)
    if method is TrimMethod.Largest:
        # inspection of these for all data shows this to be most consistent at
        # removing only some large eigenvalues
        trimmed = kmeans_trim(eigs, k=2, log=True)
    elif method is TrimMethod.Middle:
        trimmed = kmeans_trim(eigs, k=2, log=True)
        n_upper_trimmed = int(np.sum(np.isnan(eigs[len(eigs) // 2 :])))
        trimmed[:n_upper_trimmed] = np.nan
    elif method is TrimMethod.Precision:
        trimmed = eigs
    trimmed = trimmed[~np.isnan(trimmed)]
    return Eigenvalues(trimmed)


# @MEMOIZER.cache
def rigidities(
    dataset: UpdatedProcessedDataset,
    degree: int,
    smoother: SmoothMethod = SmoothMethod.Polynomial,
    trim_method: TrimMethod | None = None,
    L: ndarray = L_VALUES,
    parallel: bool = True,
    silent: bool = True,
) -> DataFrame:
    L_hash = sha256(L.data.tobytes()).hexdigest()
    if trim_method is None:
        to_hash: tuple[Any, ...] = (
            "rigidity",
            dataset.id,
            str(degree),
            smoother.name,
            L_hash,
        )
    else:
        to_hash = (
            "rigidity",
            dataset.id,
            str(degree),
            smoother.name,
            trim_method.name,
            L_hash,
        )
    # quick and dirty hashing for caching  https://stackoverflow.com/a/1151705
    hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
    outfile = CACHE_DIR / f"{hsh}.json"
    if outfile.exists():
        if not silent:
            print(f"Loading pre-computed rigidities from {outfile}")
        return pd.read_json(outfile)

    unfoldeds = dataset.unfolded(
        smoother=smoother, degree=degree, trim_method=trim_method
    )
    args = [ObservableArgs(unfolded=unf.vals, L=L) for unf in unfoldeds]  # type: ignore
    trim_label = "None" if trim_method is None else trim_method.name
    desc = f"Computing rigidities for {dataset}"
    desc = desc.replace(")", f", trim={trim_label}, degree={degree})")
    if parallel:
        rigidities = process_map(_compute_rigidity, args, desc=desc)
    else:
        rigidities = list(map(_compute_rigidity, tqdm(args, desc=desc)))

    rigs, labels = [], []
    for rig, label in zip(rigidities, dataset.labels()):
        if rig is not None:
            rigs.append(rig)
            labels.append(label)
    df = DataFrame(data=np.stack(rigs, axis=0), columns=L)
    df["y"] = labels
    df.to_json(outfile, indent=2)
    print(f"Saved rigidities to {outfile}")
    return df


# @MEMOIZER.cache
def levelvars(
    dataset: UpdatedProcessedDataset,
    degree: int,
    smoother: SmoothMethod = SmoothMethod.Polynomial,
    trim_method: TrimMethod | None = None,
    L: ndarray = L_VALUES,
    parallel: bool = True,
    silent: bool = True,
) -> DataFrame:
    L_hash = sha256(L.data.tobytes()).hexdigest()
    if trim_method is None:
        to_hash: tuple[Any, ...] = (
            "levelvars",
            dataset.id,
            str(degree),
            smoother.name,
            L_hash,
        )
    else:
        to_hash = (
            "levelvars",
            dataset.id,
            str(degree),
            smoother.name,
            trim_method.name,
            L_hash,
        )
    # quick and dirty hashing for caching  https://stackoverflow.com/a/1151705
    hsh = sha256(str(tuple(sorted(to_hash))).encode()).hexdigest()
    outfile = CACHE_DIR / f"{hsh}.json"
    if outfile.exists():
        if not silent:
            print(f"Loading pre-computed levelvars from {outfile}")
        return pd.read_json(outfile)

    unfoldeds = dataset.unfolded(
        smoother=smoother, degree=degree, trim_method=trim_method
    )
    args = [ObservableArgs(unfolded=unf.vals, L=L) for unf in unfoldeds]  # type: ignore
    trim_label = "None" if trim_method is None else trim_method.name
    desc = f"Computing levelvars for {dataset}"
    desc = desc.replace(")", f", trim={trim_label}, degree={degree})")
    if parallel:
        lvars = process_map(_compute_levelvar, args, desc=desc)
    else:
        lvars = list(map(_compute_levelvar, tqdm(args, desc=desc)))

    lvars_all, labels = [], []
    for lvar, label in zip(lvars, dataset.labels()):
        if lvar is not None:
            lvars_all.append(lvar)
            labels.append(label)
    df = DataFrame(data=np.stack(lvars_all, axis=0), columns=L)
    df["y"] = labels
    df.to_json(outfile, indent=2)
    print(f"Saved levelvars to {outfile}")
    return df


if __name__ == "__main__":
    for source in UpdatedDataset:
        if ("Vigil" not in source.name) and ("Task" not in source.name):
            continue
        for preproc in PreprocLevel:
            # for degree in [5, 7, 9]:
            data = UpdatedProcessedDataset(source=source, preproc_level=preproc)
            print(data.eigs_df())
            # rigs = rigidities(dataset=data, degree=degree, parallel=True)
            # level_vars = levelvars(dataset=data, degree=degree, parallel=True)
            #   data = UpdatedProcessedDataset(source=source, full_pre=True)
            #   # rigs = rigidities(dataset=data, degree=degree, parallel=True)
            #   # level_vars = levelvars(dataset=data, degree=degree, parallel=True)
