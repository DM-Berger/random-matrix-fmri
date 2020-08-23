import numpy as np
import os
import pandas as pd

from empyricalRMT.eigenvalues import Eigenvalues
from pathlib import Path
from tqdm import tqdm
from typing import Any, List

from rmt._data_constants import DATASETS, DATASETS_FULLPRE
from rmt._filenames import precomputed_subgroup_paths_from_args, relpath
from rmt._types import DataSummaryPaths


def precompute_largest(eigpaths: List[Path], out: Path, force: bool = False, silent: bool = False) -> Path:
    """Take the eigenvalues saved in `eigpaths`, compute the largest, and save that in a DataFrame
    in `out`

    Parameters
    ----------
    eigpaths: List[Path]
        The values of either DATASETS or DATASETS_FULLPRE

    out: Path
        See usage below.

    force: bool
        If False (default), don't recompute the values if they already exist.

    silent: bool
        If False (default) display a tqdm progress bar while calculating.

    Returns
    -------
    pickle: Path
        Path to the pickle file saving the precomputed values.
    """
    if not force and out.exists():
        return out
    largest = pd.DataFrame(index=["largest"], columns=[path.stem for path in eigpaths])
    desc = "{} - Largest"
    pbar = tqdm(total=len(eigpaths), desc=desc.format("eigs-XX"), disable=silent)
    for path in eigpaths:
        eigname = path.stem
        vals = np.load(path)
        maximum = vals[-1]
        largest.loc["largest", eigname] = maximum
        pbar.set_description(desc=desc.format(path.stem))
        pbar.update()
    pbar.close()
    # print(marchenko_df)
    largest.to_pickle(out)
    return out


def precompute_marchenko(eigpaths: List[Path], out: Path, force: bool = False, silent: bool = False) -> Path:
    """Take the eigenvalues saved in `eigpaths`, compute the Marchenko-Pastur
    endpoints (both shifted and unshifted), and save that in a DataFrame at
    `out`. DataFrame will also contain information rated to proportion of
    eigenvalues within those bounds "noise_ratio" and "noise_ratio" shifted.

    Parameters
    ----------
    eigpaths: List[Path]
        The values of either DATASETS or DATASETS_FULLPRE

    out: Path
        See usage below.

    force: bool
        If False (default), don't recompute the values if they already exist.

    silent: bool
        If False (default) display a tqdm progress bar while calculating.

    Returns
    -------
    pickle: Path
        Path to the pickle file saving the precomputed values.
    """
    if not force and out.exists():
        return out
    marchenko_df = pd.DataFrame(
        index=["low", "high", "low_shift", "high_shift", "noise_ratio", "noise_ratio_shifted"], dtype=int
    )
    desc = "{} - Marchenko"
    pbar = tqdm(total=len(eigpaths), desc=desc.format("eigs-XX"), disable=silent)
    for path in eigpaths:
        eigname = path.stem
        vals = np.load(path)
        # trim the phoney zero eigenvalue due to correlation rank
        vals = vals[1:]
        N, T = np.load(str(path).replace("eigs", "shapes"))
        eigs = Eigenvalues(vals)
        _, marchenko = eigs.trim_marchenko_pastur(series_length=T, n_series=N, use_shifted=False)
        _, marchenko_shifted = eigs.trim_marchenko_pastur(series_length=T, n_series=N, use_shifted=True)
        noise_ratio = np.mean((vals > marchenko[0]) & (vals < marchenko[1]))
        noise_ratio_shifted = np.mean((vals > marchenko_shifted[0]) & (vals < marchenko_shifted[1]))
        marchenko_df[eigname] = [
            marchenko[0],
            marchenko[1],
            marchenko_shifted[0],
            marchenko_shifted[1],
            noise_ratio,
            noise_ratio_shifted,
        ]
        pbar.set_description(desc=desc.format(path.stem))
        pbar.update()
    pbar.close()
    # print(marchenko_df)
    marchenko_df.to_pickle(out)
    return out


def precompute_brody(eigpaths: List[Path], args: Any, out: Path, force: bool = False, silent: bool = False) -> Path:
    """Take the eigenvalues saved in `eigpaths`, compute the Brody parameter beta,
    and save that in a DataFrame in `out`

    Parameters
    ----------
    eigpaths: List[Path]
        The values of either DATASETS or DATASETS_FULLPRE

    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    out: Path
        See usage below.

    force: bool
        If False (default), don't recompute the values if they already exist.

    silent: bool
        If False (default) display a tqdm progress bar while calculating.

    Returns
    -------
    pickle: Path
        Path to the pickle file saving the precomputed values.
    """
    if not force and out.exists():
        return out
    brod_df = pd.DataFrame(index=["beta"])
    desc = "{} - Brody"
    pbar = tqdm(total=len(eigpaths), desc=desc.format("eigs-XX"), disable=silent)
    for path in eigpaths:
        eigname = path.stem
        vals = np.load(path)
        if args.trim in ["(1,:)", "", "(0,:)"]:
            vals = vals[1:]  # smallest eigenvalue is always spurious here
        else:
            low, high = eval(args.trim)
            vals = vals[low:high]
        eigs = Eigenvalues(vals)
        unfolded = eigs.unfold(**args.unfold)
        # print(f"\t\tComputing Brody fit for {str(path.resolve().name)}...")
        pbar.set_description(desc=desc.format(path.stem))
        pbar.update()
        brody = unfolded.fit_brody(**args.brody)
        brod_df[eigname] = brody["beta"]
    pbar.close()
    brod_df.to_pickle(out)
    return out


def precompute_rigidity(eigpaths: List[Path], args: Any, out: Path, force: bool = False, silent: bool = False) -> Path:
    """Take the eigenvalues saved in `eigpaths`, compute the rigidity, and save that in a DataFrame
    in `out`

    Parameters
    ----------
    eigpaths: List[Path]
        The values of either DATASETS or DATASETS_FULLPRE

    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    out: Path
        See usage below.

    force: bool
        If False (default), don't recompute the values if they already exist.

    silent: bool
        If False (default) display a tqdm progress bar while calculating.

    Returns
    -------
    pickle: Path
        Path to the pickle file saving the precomputed values.
    """
    if not force and out.exists():
        return out
    rig_df = pd.DataFrame()
    desc = "{} - Rigidity"
    pbar = tqdm(total=len(eigpaths), desc=desc.format("eigs-XX"), disable=silent)
    for path in eigpaths:
        eigname = path.stem
        vals = np.load(path)
        if args.trim in ["(1,:)", "", "(0,:)"]:
            vals = vals[1:]  # smallest eigenvalue is always spurious here
        else:
            low, high = eval(args.trim)
            vals = vals[low:high]
        eigs = Eigenvalues(vals)
        unfolded = eigs.unfold(**args.unfold)
        pbar.set_description(desc=desc.format(path.stem))
        rigidity = unfolded.spectral_rigidity(**args.rigidity)
        pbar.update()
        if rig_df.get("L") is None:
            rig_df["L"] = rigidity["L"]
        rig_df[eigname] = rigidity["delta"]
    pbar.close()
    rig_df.to_pickle(out)
    return out


def precompute_levelvar(eigpaths: List[Path], args: Any, out: Path, force: bool = False, silent: bool = False) -> Path:
    """Take the eigenvalues saved in `eigpaths`, compute the levelvar, and save that in a DataFrame
    in `out`

    Parameters
    ----------
    eigpaths: List[Path]
        The values of either DATASETS or DATASETS_FULLPRE

    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    out: Path
        See usage below.

    force: bool
        If False (default), don't recompute the values if they already exist.

    silent: bool
        If False (default) display a tqdm progress bar while calculating.

    Returns
    -------
    pickle: Path
        Path to the pickle file saving the precomputed values.
    """
    if not force and out.exists():
        return out
    var_df = pd.DataFrame()
    desc = "{} - Levelvar"
    pbar = tqdm(total=len(eigpaths), desc=desc.format("eigs-XX"), disable=silent)
    for path in eigpaths:
        eigname = path.stem
        vals = np.load(path)
        if args.trim in ["(1,:)", "", "(0,:)"]:
            vals = vals[1:]  # smallest eigenvalue is always spurious here
        else:
            low, high = eval(args.trim)
            vals = vals[low:high]
        eigs = Eigenvalues(vals)
        unfolded = eigs.unfold(**args.unfold)
        pbar.set_description(desc=desc.format(path.stem))
        levelvar = unfolded.level_variance(**args.levelvar)
        pbar.update()
        if var_df.get("L") is None:
            var_df["L"] = levelvar["L"]
        var_df[eigname] = levelvar["sigma"]
    pbar.close()
    var_df.to_pickle(out)
    return out


def precompute_dataset(
    dataset_name: str,
    args: Any,
    force_all: bool = False,
    force_levelvar: bool = False,
    force_brody: bool = False,
    force_largest: bool = False,
    silent: bool = False,
) -> DataSummaryPaths:
    """Compute computationally expensive values (e.g. rigidity, level variance) for
    all subjects of all subgroups of a dataset into `precompute_root`. If files
    are already there, and no computation is `forced`, then just return the
    output paths immediately.

    Parameters
    ----------
    dataset_name: str
        The Dataset name (e.g. the key value for indexing into _data_constants.DATASETS).

    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    force_all: bool
        If True, recompute ALL summary dataframes even if files already exist.

    Returns
    -------
    computed: Dict[str, Dict[Observable, Path]]
        The dict of paths for the dataset. Initial keys are subgroup names,
        subkeys for each value are "rigidity", "levelvar", "brody", "marchenko"

        ```
        {
            "groupname1": {
                "eigs": Path,
                "rigidity": Path,
                "levelvar": Path,
                "brody": Path,
                "marchenko": Path,
            }
        }
        ```
    """
    if force_all:
        force_brody = force_levelvar = force_largest = True

    dataset = DATASETS_FULLPRE[dataset_name] if args.fullpre else DATASETS[dataset_name]
    ret: DataSummaryPaths = {}
    for subgroupname, eigpaths in dataset.items():
        outpaths = precomputed_subgroup_paths_from_args(dataset_name, subgroupname, args=args)
        rig_out = outpaths["rigidity"]
        var_out = outpaths["levelvar"]
        brod_out = outpaths["brody"]
        march_out = outpaths["marchenko"]
        largest_out = outpaths["largest"]

        os.makedirs(str(rig_out.parent), exist_ok=True)
        os.makedirs(str(var_out.parent), exist_ok=True)
        os.makedirs(str(brod_out.parent), exist_ok=True)
        os.makedirs(str(march_out.parent), exist_ok=True)
        os.makedirs(str(largest_out.parent), exist_ok=True)

        ret[subgroupname] = {
            "rigidity": rig_out,
            "levelvar": var_out,
            "brody": brod_out,
            "marchenko": march_out,
            "largest": largest_out,
        }

        sublabel = subgroupname.upper()

        if not force_all:
            if np.alltrue([rig_out.exists(), var_out.exists(), march_out.exists(), largest_out.exists()]):
                if not silent:
                    print(f"All observables for subgroup {sublabel} exist. Skipping to next subgroup.")
                continue

        log = [f"\nComputing measures for subgroup {sublabel} to {relpath(rig_out)}:"]

        do_largest = force_largest or not largest_out.exists()
        log.append(f"Computing largest eigenvalues for subgroup {sublabel}:")
        precompute_largest(eigpaths, largest_out, do_largest, silent)
        log.append(f"Saved largest eigenvalue data to {relpath(largest_out)}")

        do_march = force_all or not march_out.exists()
        log.append(f"Computing Marchenko trims for subgroup {sublabel}:")
        precompute_marchenko(eigpaths, march_out, do_march, silent)
        log.append(f"Saved Marchenko-Pastur data to {relpath(march_out)}")

        # compute and append rigidities for subgroup
        do_rig = force_all or not rig_out.exists()
        log.append(f"Computing rigidities for subgroup {sublabel}:")
        precompute_rigidity(eigpaths, args, rig_out, do_rig, silent)
        log.append(f"Saved rigidity data to {relpath(rig_out)}")
        # else:
        # log.append(f"Rigidities for subgroup {sublabel} already exist.")

        # compute and append level variances
        do_var = force_levelvar or not var_out.exists()
        log.append(f"Computing level variances for subgroup {sublabel}:")
        precompute_levelvar(eigpaths, args, var_out, do_var, silent)
        log.append(f"Saved levelvar data to {relpath(var_out)}")
        # else:
        #     log.append(f"Levelvars for subgroup {sublabel} already exist.")

        # compute and append Brody fits
        do_brody = force_brody or not brod_out.exists()
        log.append(f"Computing Brody fits for subgroup {sublabel}:")
        precompute_brody(eigpaths, args, brod_out, do_brody, silent)
        log.append(f"Saved brody data to {relpath(brod_out)}")
        # else:
        #     log.append(f"Brody fits for subgroup {sublabel} already exist.")

        if not silent:
            print("\n".join(log))

    return ret
