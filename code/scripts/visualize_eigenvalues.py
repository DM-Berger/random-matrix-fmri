# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from rmt.dataset import ProcessedDataset, levelvars
from rmt.enumerables import Dataset
from rmt.features import Eigenvalues
from rmt.predict import predict_feature
from rmt.visualize import plot_all_features

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots/eigenvalues"
PLOT_OUTDIR.mkdir(exist_ok=True, parents=True)


def predict_data(args: Namespace) -> DataFrame:
    data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
    # eigs = data.eigs_df(unify="percentile", diff=True)
    eigs = data.eigs_df(unify="pad", diff=False)  # highest accs
    return predict_feature(
        eigs,
        data,
        feature_idx=args.eig_idx,
        norm=args.norm,
        logarithm=True,
    )


def summarize_all_predictions(
    sources: Optional[list[Dataset]] = None,
    eig_idxs: Optional[list[int | None | slice]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 100,
) -> None:
    sources = sources or [*Dataset]
    eig_idxs = eig_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                eig_idx=eig_idxs,
                full_pre=full_pres,
                norm=norms,
            )
        )
    ]
    # grid = grid[:100]
    dfs = process_map(predict_data, grid, desc="Predicting")
    df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by="acc+", ascending=False)
    print(df.iloc[:print_rows, :].to_markdown(index=False, tablefmt="simple"))

    corrs = pd.get_dummies(df.drop(columns=["data", "comparison"]))
    print("-" * 80)
    print("Spearman correlations")
    print("-" * 80)
    print(corrs.corr(method="spearman").loc["acc+"])
    corrs = corrs.loc[corrs["acc+"] > 0.0]
    print("-" * 80)
    print("Spearman correlations of actual predictive pairs")
    print("-" * 80)
    print(corrs.corr(method="spearman").loc["acc+"])


if __name__ == "__main__":
    # EIG_IDXS: List[int | None] = [None]
    # EIG_IDXS = [None, -2]
    # EIG_IDXS: List[int | slice | None] = [
    #     None,
    #     slice(-80, -1),
    #     slice(-40, -1),
    #     slice(-20, -1),
    #     slice(-10, -1),
    #     # slice(-5, -1),
    # ]
    # plot_all_eigvals(
    #     # plot_separations=False,
    #     plot_separations=True,
    #     save=False,
    # )
    plot_all_features(
        feature_cls=Eigenvalues,
        plot_separations=True,
        save=False,
    )
    sys.exit()
    summarize_all_predictions(
        eig_idxs=EIG_IDXS,
    )
