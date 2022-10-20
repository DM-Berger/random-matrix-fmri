# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
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
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

from rmt.dataset import ProcessedDataset, kmeans_trim, precision_trim
from rmt.enumerables import Dataset
from rmt.visualize import PLOT_OUTDIR

OUTDIR = PLOT_OUTDIR / "trim_plots"
OUTDIR.mkdir(exist_ok=True, parents=True)

def dodge(eigs: list[ndarray], space: float) -> list[ndarray]:
    return [e + n * space for n, e in enumerate(eigs)]


if __name__ == "__main__":
    for source in Dataset:
        for preproc in [True, False]:
            data = ProcessedDataset(source=source, full_pre=preproc)
            eigs = [precision_trim(e) for e in data.eigs()]
            e_max = np.max([np.log10(e.max()) for e in eigs])
            space = e_max / 2
            eigs_log = dodge([np.log10(e) for e in eigs], space)

            sbn.set_style("darkgrid")
            ax: Axes
            fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
            for i, k in tqdm(enumerate([2, 3, 4, 5]), total=4, desc=f"Computing trims for {source.name}, fullpre={preproc}"):
                trims = dodge(
                    [np.log10(kmeans_trim(e, k=k, log=False)) for e in eigs], space
                )
                trims_log = dodge(
                    [np.log10(kmeans_trim(e, k=k, log=True)) for e in eigs], space
                )

                for eig, trim in zip(eigs_log, trims):
                    axes[0][i].plot(eig, color="black", lw=2.0)
                    axes[0][i].plot(trim, color="orange", lw=1.5)
                    axes[0][i].set_title(f"k={k}, log=False")
                for eig, trim in zip(eigs_log, trims_log):
                    axes[1][i].plot(eig, color="black", lw=2.0)
                    axes[1][i].plot(trim, color="orange", lw=1.5)
                    axes[1][i].set_title(f"k={k}, log=True")
                # for ax in axes.flat:
                #     ax.set_yscale("log")
            fig.suptitle(f"{source.name}, preproc={preproc}")
            fig.set_size_inches(w=16, h=32)
            fig.tight_layout()
            outfile = OUTDIR / f"{source.name}_preproc={preproc}.png"
            fig.savefig(outfile, dpi=600)
            print(f"Saved figure to {outfile}")
            plt.close()
