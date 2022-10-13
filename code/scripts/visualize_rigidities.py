# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tqdm.contrib.concurrent import process_map

from rmt.dataset import ProcessedDataset, levelvars
from rmt.enumerables import Dataset
from rmt.features import Rigidities
from rmt.predict import predict_feature, summarize_all_predictions
from rmt.visualize import plot_all_features

if __name__ == "__main__":
    DEGREES = [3, 5, 7, 9]
    # L_IDXS: List[int | None] = [None]
    L_IDXS: list[int | None] = [-2]
    # plot_all_features(
    #     feature_cls=Rigidities,
    #     plot_separations=False,
    #     degrees=DEGREES,
    #     save=False,
    # )
    # sys.exit()
    summarize_all_predictions(
        feature_cls=Rigidities,
        degrees=DEGREES,
        feature_idxs=L_IDXS,
    )
