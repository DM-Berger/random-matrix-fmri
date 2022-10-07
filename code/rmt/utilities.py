from typing import Any, Union

import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from statsmodels.nonparametric.kde import KDEUnivariate as KDE


def _percentile_boot(df: pd.DataFrame, B: int = 2000) -> pd.DataFrame:
    """Compute the percentile bootstrapped mean and lower and upper 95% CI bounds

    Parameters
    ----------
    df: DataFrame
        DataFrame of observable values, with each column being a subject.
        Assumes index has been correctly set (e.g. is "L" or "beta")

    B: int
        Number of bootstrap resamples.

    Returns
    -------
    bootstraps: DataFrame
        DataFrame with columns "mean", "low, "high", and with index same as df.
    """
    m, n = df.shape
    out = pd.DataFrame(index=df.index, columns=["mean", "low", "high"], dtype=float)
    for i in range(m):  # get bootstrap estimates for each row
        boot_resamples = np.random.choice(df.iloc[i, :], size=(B, n), replace=True)
        boot_means = np.sort(boot_resamples.mean(axis=1))
        mean = np.mean(boot_means)
        low, high = np.percentile(boot_means, [0.5, 99.5])
        out.iloc[i, :] = [mean, low, high]
    return out


def _configure_sbn_style(prepend_black: bool = True) -> None:
    """
    NOTE
    ----
    This must be called every time *before* creating new figs and axes. In order
    to fully ensure all plots get the correct style.
    """
    palette = sbn.color_palette("dark").copy()
    if prepend_black:
        palette.insert(0, (0.0, 0.0, 0.0))
    sbn.set()
    sbn.set_palette(palette)


def _kde(values: ndarray, grid: ndarray, bw: Union[float, str] = "scott") -> ndarray:
    """Calculate KDE for observed spacings.

    Parameters
    ----------
    values: ndarray
        the values used to compute (fit) the kernel density estimate

    grid: ndarray
        the grid of values over which to evaluate the computed KDE curve

    axes: pyplot.Axes
        the current axes object to be modified

    bw: bandwidh
        The `bw` argument for statsmodels KDEUnivariate .fit

    Returns
    -------
    evaluated: ndarray
        Value of KDE evaluated on `grid`.

    Notes
    -----
    we are doing this manually because we want to ensure consistency of the KDE
    calculation and remove Seaborn control over the process, while also avoiding
    inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938
    and https://github.com/mwaskom/seaborn/issues/796
    """
    values = values[values > 0]  # prevent floating-point bad behaviour
    kde = KDE(values)
    # kde.fit(kernel="gau", bw="scott", cut=0)
    kde.fit(kernel="gau", bw=bw, cut=0)
    evaluated = np.empty_like(grid)
    for i, _ in enumerate(evaluated):
        evaluated[i] = kde.evaluate(grid[i])
    return evaluated


def _cohen_d(g1: Any, g2: Any) -> float:
    n0 = len(g1) - 1  # a little algebra to save some space
    n1 = len(g2) - 1
    diff = np.mean(g1) - np.mean(g2)
    v0, v1 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_sd = np.sqrt((n0 * v0 + n1 * v1) / (n0 + n1))
    d = 0 if pooled_sd == 0 else diff / pooled_sd
    return d


def merge(a, b, path=None):
    """Recursively merge b into a"""
    import copy

    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
        else:
            a[key] = copy.deepcopy(b[key])
    return a
