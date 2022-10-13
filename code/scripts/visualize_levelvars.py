# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

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
from rmt.features import Levelvars
from rmt.visualize import plot_all_features

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots/levelvars"
PLOT_OUTDIR.mkdir(exist_ok=True, parents=True)


def predict_levelvar_sep(
    lvars: DataFrame,
    data: ProcessedDataset,
    degree: int,
    L_idx: int | None = None,
    norm: bool = False,
) -> DataFrame:
    DUDS = [
        "control v control_pre",
        "control v park_pre",
        "parkinsons v control_pre",
        "parkinsons v park_pre",
    ]

    labels = lvars.y.unique().tolist()
    L_label = lvars.columns[L_idx - 1] if L_idx is not None else "All"
    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            df = lvars if L_idx is None else lvars.iloc[:, [L_idx - 1, -1]]
            title = f"{labels[i]} v {labels[j]}"
            skip = False
            for dud in DUDS:
                if dud in title:
                    skip = True
                    break
            if skip:
                continue
            idx = (df.y == labels[i]) | (df.y == labels[j])
            df = df.loc[idx]
            X = df.drop(columns="y").applymap(np.log).to_numpy()
            y = LabelEncoder().fit_transform(df.y.to_numpy())
            result_dfs = [
                kfold_eval(X, y, SVC, norm=norm, title=title),
                kfold_eval(X, y, LR, norm=norm, title=title),
                kfold_eval(X, y, GBC, norm=norm, title=title),
            ]
            results.append(pd.concat(result_dfs, axis=0, ignore_index=True))
    result = pd.concat(results, axis=0, ignore_index=True)

    result["deg"] = str(degree)
    result["data"] = data.source.name
    result["preproc"] = "full" if data.full_pre else "minimal"
    result["L"] = str(L_label)
    return result.loc[
        :,
        [
            "data",
            "preproc",
            "deg",
            "norm",
            "L",
            "comparison",
            "classifier",
            "acc+",
            "mean",
            "min",
            "max",
        ],
    ]


def predict_data(args: Namespace) -> DataFrame:
    data = ProcessedDataset(source=args.source, full_pre=args.full_pre)
    lvars = levelvars(dataset=data, degree=args.degree, parallel=True)
    return predict_levelvar_sep(
        lvars,
        data,
        degree=args.degree,
        L_idx=args.L_idx,
        norm=args.norm,
    )


def summarize_all_predictions(
    sources: Optional[list[Dataset]] = None,
    degrees: Optional[list[int]] = None,
    L_idxs: Optional[list[int | None]] = None,
    full_pres: Optional[list[bool]] = None,
    norms: Optional[list[bool]] = None,
    print_rows: int = 200,
) -> None:
    sources = sources or [*Dataset]
    degrees = degrees or [3, 5, 7, 9]
    L_idxs = L_idxs or [None, -2, 3]
    full_pres = full_pres or [True, False]
    norms = norms or [True, False]

    grid = [
        Namespace(**p)
        for p in ParameterGrid(
            dict(
                source=sources,
                degree=degrees,
                L_idx=L_idxs,
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
    DEGREES = [5, 7, 9]
    # L_IDXS: List[int | None] = [None]
    L_IDXS = [None, 3, -2]
    # plot_all_levelvars(
    #     degrees=DEGREES,
    #     plot_separations=True,
    #     save=True,
    # )

    plot_all_features(
        feature_cls=Levelvars,
        plot_separations=False,
        degrees=DEGREES,
        save=False,
    )
    sys.exit()
    summarize_all_predictions(
        degrees=DEGREES,
        L_idxs=L_IDXS,
    )
