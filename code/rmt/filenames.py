"""Utilities for working with paths and filenames"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from rmt._types import Observable
from rmt.constants import (
    DATASETS,
    DATASETS_FULLPRE,
    LEGACY_DATA_ROOT,
    PRECOMPUTE_OUTDIRS,
    PRECOMPUTE_OUTDIRS_FULLPRE,
)
from rmt.defaults import (
    BRODY_ARG_DEFAULTS,
    LEVELVAR_ARG_DEFAULTS,
    RIGIDITY_ARG_DEFAULTS,
    UNFOLD_ARG_DEFAULTS,
)


def _prefix(trim_args: str, unfold_args: dict) -> str:
    unfold_args = {**UNFOLD_ARG_DEFAULTS, **unfold_args}
    prefix = (
        f"{'' if (trim_args == '(1,:)') else f'tr{trim_args}_'}"
        f"{unfold_args['smoother']}_deg{unfold_args['degree']}"
        f"{'_detrended' if unfold_args['detrend'] else ''}"
    )
    return prefix


def relpath(p: Path) -> str:
    return str(p.relative_to(LEGACY_DATA_ROOT))


def argstrings_from_args(args: Any) -> Tuple[str, Dict[Observable, str]]:
    """Generate unique "argument strings" (useful for naming files) from `args`

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    Returns
    -------
    unfold_str: str
        A string that uniquely and readably identifies the unfolding arguments

    prefix_dict: Dict[str, str]
        A dict with keys "prefix", "rigidity", "levelvar", "brody", "marchenko", "nnsd",
        and "largest", and values that are strings that uniquely correspond to
        the arguments
    """
    # defaults
    unfold_args = {
        **UNFOLD_ARG_DEFAULTS,
        **args.unfold,
    }  # splat (yes, that's the name for this syntax)
    levelvar_args = {**LEVELVAR_ARG_DEFAULTS, **args.levelvar}
    rigidity_args = {**RIGIDITY_ARG_DEFAULTS, **args.rigidity}
    brody_args = {**BRODY_ARG_DEFAULTS, **args.brody}

    prefix = _prefix(args.trim, unfold_args)
    L_max_levelvar = int(np.ceil(levelvar_args["L"].max()))
    L_max_rigidity = int(np.ceil(rigidity_args["L"].max()))

    levelvar = f"{prefix}_var-maxL-{L_max_levelvar}"
    rigidity = f"{prefix}_rig-maxL-{L_max_rigidity}"
    brody = f"{prefix}_brody-{brody_args['method']}"
    nnsd = f"{prefix}_nnsd"  # variable spacings, can't have a dataframe
    largest = "largest"  # no need for prefixes, no unfolding or trimming
    marchenko = "marchenko"  # no need for prefixes, no unfolding or trimming

    return (
        prefix,
        {
            "rigidity": rigidity,
            "levelvar": levelvar,
            "brody": brody,
            "nnsd": nnsd,
            "largest": largest,
            "marchenko": marchenko,
        },
    )


def stats_fnames(
    args: Any, normalize: bool, extension: str = "pickle"
) -> Tuple[str, str]:
    """Generates filenames for the complete statistics tables.

    Parameters
    ----------
    args: Args
        Contains the unfolding, trimming, normalization, etc options defined in
        run.py

    Returns
    -------
    preds: str
        The path of the predictions file. This file contains all the mean LOOCV
        ("mean" here meaning only averaging across LOOCV folds) predictions for
        all classifiers, all comparison groups with the unfolding options given
        by `args`.

    diffs: str
        The path of the diffs stats file. This file contains the descriptive
        statistics (e.g. means, Cohen's d's, Welch's, t-test, etc) of the scalar
        features (e.g. Largest, Noise Ratios, Brody) for unfolding options given
        by `args`.
    """
    prefix, labs = argstrings_from_args(args)
    prefix = prefix + "_"
    rig_label = labs["rigidity"].replace(prefix, "")
    var_label = labs["levelvar"].replace(prefix, "")
    brod_label = labs["brody"].replace(prefix, "")
    fullpre = "_fullpre" if args.fullpre else ""
    preds_fname = f"predicts{fullpre}_{prefix}{'_normed' if normalize else ''}_{rig_label}_{var_label}_{brod_label}.{extension}"  # noqa E501
    diffs_fname = (
        f"diffs{fullpre}_{prefix}_{rig_label}_{var_label}_{brod_label}.{extension}"
    )
    return preds_fname, diffs_fname


def precomputed_subgroup_paths_from_args(
    dataset_name: str, subgroupname: str, args: Any
) -> Dict[Observable, Path]:
    """Generate the paths of files output for given `args`

    Parameters
    ----------
    dataset_name: str
        The Dataset name (e.g. the key value for indexing into constants.DATASETS).

    subgroupname: str
        The name of the SUBGROUP being analyzed (not dataset name). E.g. the
        top-level key values defined in constants.get_all_filepath_groupings().

    Returns
    -------
    path_dict: Dict[str, Path]
        A dict with keys "rigidity", "levelvar", "brody", "nnsd", "marchenko", and
        values that are the paths to the pickled, precomputed summary dataframes
    """
    _, argnames = argstrings_from_args(args)
    levelvar_name = argnames["levelvar"]
    rigidity_name = argnames["rigidity"]
    brody_name = argnames["brody"]
    nnsd_name = argnames["nnsd"]
    largest_name = argnames["largest"]
    marchenko_name = argnames["marchenko"]

    extension = ".zip"

    precompute = PRECOMPUTE_OUTDIRS_FULLPRE if args.fullpre else PRECOMPUTE_OUTDIRS
    root = (precompute[dataset_name] / subgroupname).resolve()
    rig_out = root / (rigidity_name + extension)
    var_out = root / (levelvar_name + extension)
    brod_out = root / (brody_name + extension)
    nnsd_out = root / nnsd_name
    largest_out = root / (largest_name + extension)
    marchenko_out = root / (marchenko_name + extension)

    return {
        "rigidity": rig_out,
        "levelvar": var_out,
        "brody": brod_out,
        "nnsd": nnsd_out,
        "largest": largest_out,
        "marchenko": marchenko_out,
    }


def previewprecompute_outpaths(args: Any) -> None:
    """Log all precompute paths to stdout, as well as whether the precompute
    files already exist"""
    dataset_name: str
    data: Dict[str, Dict[str, List[Path]]]
    datasets = DATASETS_FULLPRE if args.fullpre else DATASETS
    for dataset_name, dataset in datasets.items():
        print(f"\nDataset: {dataset_name}")
        for subgroupname in sorted(dataset.keys()):
            print(f"\tSubgroup: {subgroupname}")
            outpaths = precomputed_subgroup_paths_from_args(
                dataset_name, subgroupname, args
            )
            for key, val in outpaths.items():
                print(f"\t\t{key} data saved at: {val.relative_to(LEGACY_DATA_ROOT)}")
                print(f"\t\tAlready exists: {val.exists()}")
