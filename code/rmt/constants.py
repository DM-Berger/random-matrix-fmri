"""Holds constants related to filepaths."""
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from rmt._types import Dataset

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
FULLPRE_DIRNAME = "rmt_fullpre"


# fmt: off
LEARNING_DATA                  = DATA_ROOT / "Rest_v_LearningRecall" / "rmt"
OSTEO_DATA                     = DATA_ROOT / "Rest_w_Healthy_v_OsteoPain" / "rmt"
PSYCHOLOGICAL_DATA             = DATA_ROOT / "Rest_w_VigilanceAttention" / "rmt"
PARKINSONS_DATA                = DATA_ROOT / "Park_v_Control" / "rmt"
REFLECTIVE_SUMMED              = DATA_ROOT / "Rest_v_Various_MultiEcho" / "rmt" / "summed"
REFLECTIVE_INTERLEAVED         = DATA_ROOT / "Rest_v_Various_MultiEcho" / "rmt" / "interleaved"

LEARNING_DATA_FULLPRE          = DATA_ROOT / "Rest_v_LearningRecall" / FULLPRE_DIRNAME
OSTEO_DATA_FULLPRE             = DATA_ROOT / "Rest_w_Healthy_v_OsteoPain" / FULLPRE_DIRNAME
PSYCHOLOGICAL_DATA_FULLPRE     = DATA_ROOT / "Rest_w_VigilanceAttention" / FULLPRE_DIRNAME
PARKINSONS_DATA_FULLPRE        = DATA_ROOT / "Park_v_Control" / FULLPRE_DIRNAME
REFLECTIVE_SUMMED_FULLPRE      = DATA_ROOT / "Rest_v_Various_MultiEcho" / FULLPRE_DIRNAME / "summed"
REFLECTIVE_INTERLEAVED_FULLPRE = DATA_ROOT / "Rest_v_Various_MultiEcho" / FULLPRE_DIRNAME / "interleaved"

VIGILANCE_SES1                 = PSYCHOLOGICAL_DATA / "selfreported_vigilance_ses-1.csv"
VIGILANCE_SES2                 = PSYCHOLOGICAL_DATA / "selfreported_vigilance_ses-2.csv"
TASK_ATTEND_SES1               = PSYCHOLOGICAL_DATA / "task_attention_ses-1.csv"
TASK_ATTEND_SES2               = PSYCHOLOGICAL_DATA / "task_attention_ses-2.csv"
WEEKLY_ATTEND_SES1             = PSYCHOLOGICAL_DATA / "weekly_attentions_ses-1.csv"
WEEKLY_ATTEND_SES2             = PSYCHOLOGICAL_DATA / "weekly_attentions_ses-2.csv"
# fmt: on

# PAIRED_DATA = ["learning"]
PAIRED_DATA: List[str] = []


def extract_psych_groups(path: Path, fullpre: bool) -> Dict[str, List[Path]]:
    """Obtain a dict with "high" and "low" keys, with values being the lists of paths
    of the extracted eigenvalues.
    """

    def is_in(ids: List[str]) -> Callable:
        def closure(path: Path) -> bool:
            p = str(path.resolve())
            loc = p.find("eigs-")
            p_id = p[loc : loc + 7]
            for eigname in ids:
                if eigname == p_id:
                    return True
            return False

        return closure

    df = pd.read_csv(path, index_col="subject_id")
    high_ids = list(
        map(lambda idx: "eigs-{:02d}".format(idx), df[df["high_attender"]].index)
    )
    low_ids = list(
        map(lambda idx: "eigs-{:02d}".format(idx), df[~df["high_attender"]].index)
    )
    all_eigs = sorted(
        [
            Path(p)
            for p in glob(
                f"{PSYCHOLOGICAL_DATA_FULLPRE if fullpre else PSYCHOLOGICAL_DATA}/**/*eigs*.npy",
                recursive=True,
            )
        ]
    )
    high = sorted(list(filter(is_in(high_ids), all_eigs)))
    low = sorted(list(filter(is_in(low_ids), all_eigs)))
    return {"high": high, "low": low}


def get_all_filepath_groupings(fullpre: bool) -> Any:
    learning_groups = {
        "task": [
            Path(f) for f in glob(str((LEARNING_DATA / "task" / "*eigs*.npy").resolve()))
        ],
        "rest": [
            Path(f) for f in glob(str((LEARNING_DATA / "rest" / "*eigs*.npy").resolve()))
        ],
    }
    learning_groups["task"].sort(key=lambda p: p.name)
    learning_groups["rest"].sort(key=lambda p: p.name)

    learning_groups_fullpre = {
        "task": [
            Path(f)
            for f in glob(str((LEARNING_DATA_FULLPRE / "task" / "*eigs*.npy").resolve()))
        ],
        "rest": [
            Path(f)
            for f in glob(str((LEARNING_DATA_FULLPRE / "rest" / "*eigs*.npy").resolve()))
        ],
    }
    learning_groups_fullpre["task"].sort(key=lambda p: p.name)
    learning_groups_fullpre["rest"].sort(key=lambda p: p.name)

    osteo_groups = {
        "allpain": [
            Path(f) for f in glob(str((OSTEO_DATA / "pain" / "**/*eigs*.npy").resolve()))
        ],
        "nopain": [
            Path(f) for f in glob(str((OSTEO_DATA / "controls" / "*eigs*.npy").resolve()))
        ],
        "duloxetine": [
            Path(f)
            for f in glob(str((OSTEO_DATA / "pain/duloxetine" / "*eigs*.npy").resolve()))
        ],
        "pain": [
            Path(f)
            for f in glob(str((OSTEO_DATA / "pain/placebo" / "*eigs*.npy").resolve()))
        ],
    }
    for key in osteo_groups.keys():
        osteo_groups[key].sort(key=lambda p: p.name)

    osteo_groups_fullpre = {
        "allpain": [
            Path(f)
            for f in glob(str((OSTEO_DATA_FULLPRE / "pain" / "**/*eigs*.npy").resolve()))
        ],
        "nopain": [
            Path(f)
            for f in glob(str((OSTEO_DATA_FULLPRE / "controls" / "*eigs*.npy").resolve()))
        ],
        "duloxetine": [
            Path(f)
            for f in glob(
                str((OSTEO_DATA_FULLPRE / "pain/duloxetine" / "*eigs*.npy").resolve())
            )
        ],
        "pain": [
            Path(f)
            for f in glob(
                str((OSTEO_DATA_FULLPRE / "pain/placebo" / "*eigs*.npy").resolve())
            )
        ],
    }
    for key in osteo_groups_fullpre.keys():
        osteo_groups_fullpre[key].sort(key=lambda p: p.name)

    psych_groups = {
        "vigilance_ses-1": extract_psych_groups(VIGILANCE_SES1, fullpre=False),
        "vigilance_ses-2": extract_psych_groups(VIGILANCE_SES2, fullpre=False),
        "task_attend_ses-1": extract_psych_groups(TASK_ATTEND_SES1, fullpre=False),
        "task_attend_ses-2": extract_psych_groups(TASK_ATTEND_SES2, fullpre=False),
        "weekly_attend_ses-1": extract_psych_groups(WEEKLY_ATTEND_SES1, fullpre=False),
        "weekly_attend_ses-2": extract_psych_groups(WEEKLY_ATTEND_SES2, fullpre=False),
    }
    psych_groups_fullpre = {
        "vigilance_ses-1": extract_psych_groups(VIGILANCE_SES1, fullpre=True),
        "vigilance_ses-2": extract_psych_groups(VIGILANCE_SES2, fullpre=True),
        "task_attend_ses-1": extract_psych_groups(TASK_ATTEND_SES1, fullpre=True),
        "task_attend_ses-2": extract_psych_groups(TASK_ATTEND_SES2, fullpre=True),
        "weekly_attend_ses-1": extract_psych_groups(WEEKLY_ATTEND_SES1, fullpre=True),
        "weekly_attend_ses-2": extract_psych_groups(WEEKLY_ATTEND_SES2, fullpre=True),
    }

    psych_scores = {
        "vigilance_ses-1": pd.read_csv(VIGILANCE_SES1)[["subject_id", "overall"]],
        "vigilance_ses-2": pd.read_csv(VIGILANCE_SES2)[["subject_id", "overall"]],
        "task_attend_ses-1": pd.read_csv(TASK_ATTEND_SES1)[["subject_id", "score"]],
        "task_attend_ses-2": pd.read_csv(TASK_ATTEND_SES2)[["subject_id", "score"]],
        "weekly_attend_ses-1": pd.read_csv(WEEKLY_ATTEND_SES1)[["subject_id", "overall"]],
        "weekly_attend_ses-2": pd.read_csv(WEEKLY_ATTEND_SES2)[["subject_id", "overall"]],
    }

    park_groups = {
        "control": [
            Path(f)
            for f in glob(
                str((PARKINSONS_DATA / "raw/controls" / "*eigs*.npy").resolve())
            )
        ],
        "parkinsons": [
            Path(f)
            for f in glob(
                str((PARKINSONS_DATA / "raw/parkinsons" / "*eigs*.npy").resolve())
            )
        ],
        "control_pre": [
            Path(f)
            for f in glob(
                str((PARKINSONS_DATA / "afni_preproc/controls" / "*eigs*.npy").resolve())
            )
        ],
        "park_pre": [
            Path(f)
            for f in glob(
                str(
                    (PARKINSONS_DATA / "afni_preproc/parkinsons" / "*eigs*.npy").resolve()
                )
            )
        ],
    }

    park_groups_fullpre = {
        "control": [
            Path(f)
            for f in glob(
                str((PARKINSONS_DATA_FULLPRE / "raw/controls" / "*eigs*.npy").resolve())
            )
        ],
        "parkinsons": [
            Path(f)
            for f in glob(
                str((PARKINSONS_DATA_FULLPRE / "raw/parkinsons" / "*eigs*.npy").resolve())
            )
        ],
        "control_pre": [
            Path(f)
            for f in glob(
                # NOTE: INTENTIONALLY THE SAME!!!
                str((PARKINSONS_DATA / "afni_preproc/controls" / "*eigs*.npy").resolve())
            )
        ],
        "park_pre": [
            Path(f)
            for f in glob(
                # NOTE: INTENTIONALLY THE SAME!!!
                str(
                    (PARKINSONS_DATA / "afni_preproc/parkinsons" / "*eigs*.npy").resolve()
                )
            )
        ],
    }

    reflect_sum_groups = {
        "task": [
            Path(f)
            for f in glob(str((REFLECTIVE_SUMMED / "task" / "*eigs*.npy").resolve()))
        ],
        "rest": [
            Path(f)
            for f in glob(
                str((REFLECTIVE_SUMMED / "rest_reshaped" / "*eigs*.npy").resolve())
            )
        ],
    }
    reflect_sum_groups_fullpre = {
        "task": [
            Path(f)
            for f in glob(
                str((REFLECTIVE_SUMMED_FULLPRE / "task" / "*eigs*.npy").resolve())
            )
        ],
        "rest": [
            Path(f)
            for f in glob(
                str(
                    (REFLECTIVE_SUMMED_FULLPRE / "rest_reshaped" / "*eigs*.npy").resolve()
                )
            )
        ],
    }

    reflect_interleave_groups = {
        "task": [
            Path(f)
            for f in glob(str((REFLECTIVE_INTERLEAVED / "task" / "*eigs*.npy").resolve()))
        ],
        "rest": [
            Path(f)
            for f in glob(
                str((REFLECTIVE_INTERLEAVED / "rest_reshaped" / "*eigs*.npy").resolve())
            )
        ],
    }

    reflect_interleave_groups_fullpre = {
        "task": [
            Path(f)
            for f in glob(
                str((REFLECTIVE_INTERLEAVED_FULLPRE / "task" / "*eigs*.npy").resolve())
            )
        ],
        "rest": [
            Path(f)
            for f in glob(
                str(
                    (
                        REFLECTIVE_INTERLEAVED_FULLPRE / "rest_reshaped" / "*eigs*.npy"
                    ).resolve()
                )
            )
        ],
    }
    if fullpre:
        return (
            learning_groups_fullpre,
            osteo_groups_fullpre,
            psych_groups_fullpre,
            psych_scores,
            park_groups_fullpre,
            reflect_sum_groups_fullpre,
            reflect_interleave_groups_fullpre,
        )
    return (
        learning_groups,
        osteo_groups,
        psych_groups,
        psych_scores,
        park_groups,
        reflect_sum_groups,
        reflect_interleave_groups,
    )


(
    learning_groups,
    osteo_groups,
    psych_groups,
    psych_scores,
    park_groups,
    reflect_sum,
    reflect_interleave,
) = get_all_filepath_groupings(fullpre=False)

# Groups at the dataset level. Psych dataset is treated as if it is three Datasets.
# a "Dataset" is a dict with keys for each subgroup
DATASETS: Dict[str, Dataset] = {
    "LEARNING": learning_groups,
    "OSTEO": osteo_groups,
    "PSYCH_VIGILANCE_SES-1": psych_groups["vigilance_ses-1"],
    "PSYCH_VIGILANCE_SES-2": psych_groups["vigilance_ses-2"],
    "PSYCH_TASK_ATTENTION_SES-1": psych_groups["task_attend_ses-1"],
    "PSYCH_TASK_ATTENTION_SES-2": psych_groups["task_attend_ses-2"],
    "PSYCH_WEEKLY_ATTENTION_SES-1": psych_groups["weekly_attend_ses-1"],
    "PSYCH_WEEKLY_ATTENTION_SES-2": psych_groups["weekly_attend_ses-2"],
    "PARKINSONS": park_groups,
    "REFLECT_SUMMED": reflect_sum,
    "REFLECT_INTERLEAVED": reflect_interleave,
}

(
    learning_groups_pre,
    osteo_groups_pre,
    psych_groups_pre,
    psych_scores_pre,
    park_groups_pre,
    reflect_sum_pre,
    reflect_interleave_pre,
) = get_all_filepath_groupings(fullpre=True)

DATASETS_FULLPRE: Dict[str, Dataset] = {
    "LEARNING": learning_groups_pre,
    "OSTEO": osteo_groups_pre,
    "PSYCH_VIGILANCE_SES-1": psych_groups_pre["vigilance_ses-1"],
    "PSYCH_VIGILANCE_SES-2": psych_groups_pre["vigilance_ses-2"],
    "PSYCH_TASK_ATTENTION_SES-1": psych_groups_pre["task_attend_ses-1"],
    "PSYCH_TASK_ATTENTION_SES-2": psych_groups_pre["task_attend_ses-2"],
    "PSYCH_WEEKLY_ATTENTION_SES-1": psych_groups_pre["weekly_attend_ses-1"],
    "PSYCH_WEEKLY_ATTENTION_SES-2": psych_groups_pre["weekly_attend_ses-2"],
    "PARKINSONS": park_groups_pre,
    "REFLECT_SUMMED": reflect_sum_pre,
    "REFLECT_INTERLEAVED": reflect_interleave_pre,
}
for dataset_name, subgroups in DATASETS_FULLPRE.items():
    for subgroup_name, paths in subgroups.items():
        for path in paths:
            try:
                assert path.exists()
            except AssertionError:
                print(f"Could not find file {path}")
                sys.exit(1)

# fmt: off
PRECOMPUTE_OUTDIRS: Dict[str, Path] = {
    "LEARNING":                     LEARNING_DATA          / "precompute",
    "OSTEO":                        OSTEO_DATA             / "precompute",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA     / "precompute" / "ses-1" / "vigilance",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA     / "precompute" / "ses-2" / "vigilance",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA     / "precompute" / "ses-1" / "task-attend",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA     / "precompute" / "ses-2" / "task-attend",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA     / "precompute" / "ses-1" / "weekly-attend",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA     / "precompute" / "ses-2" / "weekly-attend",
    "PARKINSONS":                   PARKINSONS_DATA        / "precompute",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED      / "precompute" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED / "precompute" / "interleaved",
}

PRECOMPUTE_OUTDIRS_FULLPRE: Dict[str, Path] = {
    "LEARNING":                     LEARNING_DATA_FULLPRE      / "precompute",
    "OSTEO":                        OSTEO_DATA_FULLPRE         / "precompute",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-1" / "vigilance",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-2" / "vigilance",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-1" / "task-attend",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-2" / "task-attend",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-1" / "weekly-attend",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA_FULLPRE / "precompute" / "ses-2" / "weekly-attend",
    "PARKINSONS":                   PARKINSONS_DATA_FULLPRE    / "precompute",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED_FULLPRE  / "precompute" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED_FULLPRE / "precompute" / "interleaved",
}

PLOT_OUTDIRS: Dict[str, Path] = {
    "LEARNING":                     LEARNING_DATA      / "plots",
    "OSTEO":                        OSTEO_DATA         / "plots",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA / "vigilance" / "ses-1" / "plots",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA / "vigilance" / "ses-2" / "plots",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA / "task-attend" / "ses-1" / "plots",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA / "task-attend" / "ses-2" / "plots",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA / "weekly-attend" / "ses-1" / "plots",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA / "weekly-attend" / "ses-2" / "plots",
    "PARKINSONS":                   PARKINSONS_DATA    / "plots",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED  / "plots" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED / "plots" / "interleaved",
}

PLOT_OUTDIRS_FULLPRE: Dict[str, Path] = {
    "LEARNING":                     LEARNING_DATA_FULLPRE / "plots",
    "OSTEO":                        OSTEO_DATA_FULLPRE / "plots",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA_FULLPRE / "vigilance" / "ses-1" / "plots",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA_FULLPRE / "vigilance" / "ses-2" / "plots",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA_FULLPRE / "task-attend" / "ses-1" / "plots",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA_FULLPRE / "task-attend" / "ses-2" / "plots",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA_FULLPRE / "weekly-attend" / "ses-1" / "plots",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA_FULLPRE / "weekly-attend" / "ses-2" / "plots",
    "PARKINSONS":                   PARKINSONS_DATA_FULLPRE    / "plots",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED_FULLPRE  / "plots" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED_FULLPRE / "plots" / "interleaved",
}
# fmt: off

STAT_OUTDIRS = {
    "LEARNING":                     LEARNING_DATA / "stats",
    "OSTEO":                        OSTEO_DATA / "stats",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA / "vigilance" / "ses-1" / "stats",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA / "vigilance" / "ses-2" / "stats",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA / "task-attend" / "ses-1" / "stats",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA / "task-attend" / "ses-2" / "stats",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA / "weekly-attend" / "ses-1" / "stats",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA / "weekly-attend" / "ses-2" / "stats",
    "PARKINSONS":                   PARKINSONS_DATA / "stats",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED / "stats" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED / "stats" / "interleaved",
}

STAT_OUTDIRS_FULLPRE = {
    "LEARNING":                     LEARNING_DATA_FULLPRE / "stats",
    "OSTEO":                        OSTEO_DATA_FULLPRE / "stats",
    "PSYCH_VIGILANCE_SES-1":        PSYCHOLOGICAL_DATA_FULLPRE / "vigilance" / "ses-1" / "stats",
    "PSYCH_VIGILANCE_SES-2":        PSYCHOLOGICAL_DATA_FULLPRE / "vigilance" / "ses-2" / "stats",
    "PSYCH_TASK_ATTENTION_SES-1":   PSYCHOLOGICAL_DATA_FULLPRE / "task-attend" / "ses-1" / "stats",
    "PSYCH_TASK_ATTENTION_SES-2":   PSYCHOLOGICAL_DATA_FULLPRE / "task-attend" / "ses-2" / "stats",
    "PSYCH_WEEKLY_ATTENTION_SES-1": PSYCHOLOGICAL_DATA_FULLPRE / "weekly-attend" / "ses-1" / "stats",
    "PSYCH_WEEKLY_ATTENTION_SES-2": PSYCHOLOGICAL_DATA_FULLPRE / "weekly-attend" / "ses-2" / "stats",
    "PARKINSONS":                   PARKINSONS_DATA_FULLPRE / "stats",
    "REFLECT_SUMMED":               REFLECTIVE_SUMMED_FULLPRE / "stats" / "summed",
    "REFLECT_INTERLEAVED":          REFLECTIVE_INTERLEAVED_FULLPRE / "stats" / "interleaved",
}
# fmt : on

# meaningless subgroup comparisons we don't want to generate data for
DUDS = [
    "control_pre_v_parkinsons",
    "park_pre_v_parkinsons",
    "control_v_park_pre",
    "control_v_control_pre",
]
