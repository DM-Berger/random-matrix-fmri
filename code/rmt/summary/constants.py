# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from joblib import Memory

PROJECT = ROOT.parent
MEMORY = Memory(PROJECT / "__JOBLIB_CACHE__")

BLUE = "#004cc7"
LBLUE = "#8dacfb"
ORNG = "#f68a0e"
GREEN = "#01f91e"
GREY = "#5c5c5c"
BLCK = "#000000"
PURP = "#8e02c5"
RED = "#de0202"
PINK = "#ff6bd8"

SPIE_OUTDIR = PROJECT / "results/plots/figures/SPIE"
SPIE_PAPER_OUTDIR = PROJECT / "paper/paper/SPIE/figures"
SPIE_OUTDIR.mkdir(exist_ok=True, parents=True)
SPIE_PAPER_OUTDIR.mkdir(exist_ok=True, parents=True)
SPIE_JMI_MAX_WIDTH_INCHES = 6.75
SPIE_JMI_MAX_COL_WIDTH_INCHES = 3 + 5 / 16
SPIE_MIN_LINE_WEIGHT = 0.5

HEADER = "=" * 80 + "\n"
FOOTER = "\n" + ("=" * 80)
DROPS = [
    "acc+",
    "auroc",
    "classifier_GradientBoostingClassifier",
]

ALL_GROUPERS = [
    "data",
    "comparison",
    "classifier",
    "preproc",
    "deg",
    "trim",
    "slice",
    "norm",
]

SUBGROUPERS = [
    ["data"],
    ["data", "comparison"],
    ["data", "comparison", "classifier"],
    ["data", "comparison", "classifier", "preproc"],
    ["data", "comparison", "classifier", "preproc", "deg"],
    ["data", "comparison", "classifier", "preproc", "trim"],
    ["data", "comparison", "classifier", "preproc", "slice"],
    ["data", "comparison", "classifier", "preproc", "deg", "trim"],
    ["data", "comparison", "classifier", "preproc", "deg", "norm"],
    ["data", "comparison", "classifier", "preproc", "deg", "trim", "norm"],
]

RMT_FEATURE_PALETTE = {
    "unfolded": GREEN,
    "rigidity": BLUE,
    "levelvar": PINK,
    "rigidity + levelvar": ORNG,
    "unfolded + levelvar": BLCK,
    "unfolded + rigidity": GREY,
    "unfolded + rigidity + levelvar": PURP,
}

RMT_FEATURE_ORDER = list(RMT_FEATURE_PALETTE.keys())

FEATURE_GROUP_PALETTE = {
    "tseries loc": RED,
    "tseries scale": PINK,
    "eigs": BLCK,
    "eigs max": ORNG,
    "eigs smooth": GREEN,
    "eigs middle": GREY,
    "rmt + eigs": PURP,
    "rmt only": BLUE,
}

NON_BASELINE_PALETTE = {
    "eigs": BLCK,
    "eigs max": ORNG,
    "eigs smooth": GREEN,
    "eigs middle": GREY,
    "rmt + eigs": PURP,
    "rmt only": BLUE,
}

GROSS_FEATURE_PALETTE = {
    "tseries": ORNG,
    "eigs": BLCK,
    "rmt": BLUE,
}

TRIM_PALETTE = {
    "none": BLCK,
    "precision": PURP,
    "largest": BLUE,
    "middle": ORNG,
}

TRIM_ORDER = list(TRIM_PALETTE.keys())

SLICE_ORDER = [
    "all",
    "max-25",
    "max-10",
    "max-05",
    "mid-25",
    "mid-10",
    "mid-05",
    "min-25",
    "min-10",
    "min-05",
]

DEGREE_ORDER = [3, 5, 7, 9]

SUBGROUP_ORDER = [
    "Bilinguality - monolingual v bilingual",
    "Depression - depress v control",
    "Learning - rest v task",
    "Aging - younger v older",
    "Osteo - nopain v duloxetine",
    "Osteo - nopain v pain",
    "Osteo - pain v duloxetine",
    "Parkinsons - ctrl v park",
    "TaskAttention - task_attend v task_nonattend",
    # "TaskAttentionSes1 - task_attend v task_nonattend",
    # "TaskAttentionSes2 - task_attend v task_nonattend",
    "Vigilance - vigilant v nonvigilant",
    # "VigilanceSes1 - vigilant v nonvigilant",
    # "VigilanceSes2 - vigilant v nonvigilant",
    "WeeklyAttention - trait_nonattend v trait_attend",
    # "WeeklyAttentionSes1 - trait_nonattend v trait_attend",
    # "WeeklyAttentionSes2 - trait_attend v trait_nonattend",
]

"""Defined as comparisons where bulk (>50%) of AUROC distribution is right of
0.5 under at least one preproc regime
"""
OVERALL_PREDICTIVE_GROUP_ORDER = [
    "Learning - rest v task",
    "Aging - younger v older",
    "Osteo - nopain v duloxetine",
    "Osteo - nopain v pain",
    # "Osteo - pain v duloxetine",
    "Parkinsons - ctrl v park",  # questionable, only tseries seem predictive
    "TaskAttention - task_attend v task_nonattend",
    "Vigilance - vigilant v nonvigilant",
]


CLASSIFIER_ORDER = [
    # "RandomForestClassifier",
    # "GradientBoostingClassifier",
    "RF",
    "GBDT",
    "KNN3",
    "KNN5",
    "KNN9",
    "SVC",
]
PREPROC_ORDER = [
    "BrainExtract",
    "SliceTimeAlign",
    "MotionCorrect",
    "MNIRegister",
]
NORM_ORDER = [
    False,
    True,
]


def get_aggregates(subgroupers: list[list[str]]) -> list[list[str]]:
    aggregates = [  # just excludes the column used for grouping and keeps ordering
        [colname for colname in ALL_GROUPERS if colname not in subgrouper]
        for subgrouper in subgroupers
    ]
    return aggregates


AGGREGATES = get_aggregates(SUBGROUPERS)
