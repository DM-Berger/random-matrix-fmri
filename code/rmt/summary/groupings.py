# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

import numpy as np
from rmt.summary.constants import (
    BLUE,
    ORNG,
    GREY,
    BLCK,
    PURP,
    RED,
    PINK,
)


def is_tseries(s: str) -> bool:
    return "T-" in s


def is_ts_loc(s: str) -> bool:
    locs = ["T-max", "T-mean", "T-med", "T-min", "T-p05", "T-p95"]
    return s in locs


def is_ts_scale(s: str) -> bool:
    scales = ["T-iqr", "T-rng", "T-rrng", "T-std"]
    return s in scales


def is_rmt(s: str) -> bool:
    return ("rig" in s) or ("level" in s) or ("unf" in s)


def is_rmt_plus(s: str) -> bool:
    return is_rmt(s) and ("eig" in s)


def is_rmt_only(s: str) -> bool:
    return is_rmt(s) and ("eigs" not in s)


def is_max(s: str) -> bool:
    return ("max" in s) and (not is_tseries(s))


def is_smoothed(s: str) -> bool:
    return ("savgol" in s) or ("smooth" in s)


def is_eigs_only(s: str) -> bool:
    return s == "eigs" or ("middle" in s)


def fine_feature_grouping(s: str) -> str:
    if is_ts_loc(s):
        return "tseries loc"
    if is_ts_scale(s):
        return "tseries scale"
    if is_smoothed(s):
        return "eigs smooth"
    if is_max(s):
        return "eigs max"
    if "middle" in s:
        return "eigs middle"
    if s == "eigs":
        return "eigs"
    if is_rmt_only(s):
        return "rmt only"
    if is_rmt_plus(s):
        return "rmt + eigs"
    raise ValueError(f"Invalid feature: {s}")


def coarse_feature_grouping(s: str) -> str:
    if is_ts_loc(s) or is_ts_scale(s):
        return "tseries"
    if is_rmt_only(s) or is_rmt_plus(s):
        return "rmt"
    return "eigs"


def slice_grouping(s: str) -> str:
    if "min" in s:
        return "min"
    if "max" in s:
        return "max"
    if "mid" in s:
        return "mid"
    return s


def make_palette(features: list[str]) -> dict[str, str]:
    palette = {}
    for feature in np.unique(features):
        if is_rmt_plus(feature):
            palette[feature] = PURP
        elif is_ts_loc(feature):
            palette[feature] = RED
        elif is_ts_scale(feature):
            palette[feature] = PINK
        elif is_rmt_only(feature):
            palette[feature] = BLUE
        elif is_max(feature):
            palette[feature] = ORNG
        elif is_smoothed(feature):
            palette[feature] = GREY
        else:
            palette[feature] = BLCK
    return palette


def get_feature_ordering(features: list[str]) -> list[str]:
    ts_loc = sorted(filter(is_ts_loc, features))
    ts_scale = sorted(filter(is_ts_scale, features))
    rmt_only = sorted(filter(is_rmt_only, features))
    rmt_plus = sorted(filter(is_rmt_plus, features))
    eigs_smooth = sorted(filter(lambda s: is_smoothed(s), features))
    eigs_max = sorted(filter(is_max, features))
    eigs_only = sorted(filter(lambda s: is_eigs_only(s), features))
    ordering = (
        ts_loc + ts_scale + eigs_smooth + eigs_max + eigs_only + rmt_only + rmt_plus
    )
    return ordering
