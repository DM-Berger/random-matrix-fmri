from enum import Enum
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/updated"
RMT_DATA = DATA / "rmt"

"""
Park_v_Control/
Rest_v_LearningRecall/
Rest_w_Bilinguality/
Rest_w_Depression_v_Control/
Rest_w_Healthy_v_OsteoPain/
Rest_w_Older_v_Younger/
Rest_w_VigilanceAttention/
"""


class Dataset(Enum):
    """
    Learning = "LEARNING"\n
    Osteo = "OSTEO"\n
    Parkinsons = "PARKINSONS"\n
    ReflectionSummed = "REFLECT_SUMMED"\n
    ReflectionInterleaved = "REFLECT_INTERLEAVED"\n
    VigilanceSes1 = "PSYCH_VIGILANCE_SES-1"\n
    VigilanceSes2 = "PSYCH_VIGILANCE_SES-2"\n
    TaskAttentionSes1 = "PSYCH_TASK_ATTENTION_SES-1"\n
    TaskAttentionSes2 = "PSYCH_TASK_ATTENTION_SES-2"\n
    WeeklyAttentionSes1 = "PSYCH_WEEKLY_ATTENTION_SES-1"\n
    WeeklyAttentionSes2 = "PSYCH_WEEKLY_ATTENTION_SES-2"\n
    Vigilance = "VIGILANCE"\n
    TaskAttention = "TASK_ATTENTION"\n
    WeeklyAttention = "WEEKLY_ATTENTION"\n
    """

    Learning = "LEARNING"
    Osteo = "OSTEO"
    Parkinsons = "PARKINSONS"
    ReflectionSummed = "REFLECT_SUMMED"
    ReflectionInterleaved = "REFLECT_INTERLEAVED"
    VigilanceSes1 = "PSYCH_VIGILANCE_SES-1"
    VigilanceSes2 = "PSYCH_VIGILANCE_SES-2"
    TaskAttentionSes1 = "PSYCH_TASK_ATTENTION_SES-1"
    TaskAttentionSes2 = "PSYCH_TASK_ATTENTION_SES-2"
    WeeklyAttentionSes1 = "PSYCH_WEEKLY_ATTENTION_SES-1"
    WeeklyAttentionSes2 = "PSYCH_WEEKLY_ATTENTION_SES-2"
    Vigilance = "VIGILANCE"
    TaskAttention = "TASK_ATTENTION"
    WeeklyAttention = "WEEKLY_ATTENTION"

    def subgroup_names(self) -> list[str]:
        return {
            Dataset.Learning: ["task", "rest"],
            Dataset.Osteo: ["allpain", "nopain", "duloxetine", "pain"],
            Dataset.Parkinsons: ["control", "parkinsons", "control_pre", "park_pre"],
            Dataset.ReflectionSummed: ["task", "rest"],
            Dataset.ReflectionInterleaved: ["task", "rest"],
            Dataset.VigilanceSes1: ["high", "low"],
            Dataset.VigilanceSes2: ["high", "low"],
            Dataset.TaskAttentionSes1: ["high", "low"],
            Dataset.TaskAttentionSes2: ["high", "low"],
            Dataset.WeeklyAttentionSes1: ["high", "low"],
            Dataset.WeeklyAttentionSes2: ["high", "low"],
            Dataset.Vigilance: ["high", "low"],
            Dataset.TaskAttention: ["high", "low"],
            Dataset.WeeklyAttention: ["high", "low"],
        }[self]


class UpdatedDataset(Enum):
    """
    Learning = "LEARNING"\n
    Osteo = "OSTEO"\n
    Parkinsons = "PARKINSONS"\n
    ReflectionSummed = "REFLECT_SUMMED"\n
    ReflectionInterleaved = "REFLECT_INTERLEAVED"\n
    VigilanceSes1 = "PSYCH_VIGILANCE_SES-1"\n
    VigilanceSes2 = "PSYCH_VIGILANCE_SES-2"\n
    TaskAttentionSes1 = "PSYCH_TASK_ATTENTION_SES-1"\n
    TaskAttentionSes2 = "PSYCH_TASK_ATTENTION_SES-2"\n
    WeeklyAttentionSes1 = "PSYCH_WEEKLY_ATTENTION_SES-1"\n
    WeeklyAttentionSes2 = "PSYCH_WEEKLY_ATTENTION_SES-2"\n
    Vigilance = "VIGILANCE"\n
    TaskAttention = "TASK_ATTENTION"\n
    WeeklyAttention = "WEEKLY_ATTENTION"\n
    """

    Learning = "LEARNING"
    Osteo = "OSTEO"
    Parkinsons = "PARKINSONS"
    Bilinguality = "BILINGUAL"
    Depression = "DEPRESSION"
    Older = "OLDER"
    VigilanceSes1 = "PSYCH_VIGILANCE_SES-1"
    VigilanceSes2 = "PSYCH_VIGILANCE_SES-2"
    TaskAttentionSes1 = "PSYCH_TASK_ATTENTION_SES-1"
    TaskAttentionSes2 = "PSYCH_TASK_ATTENTION_SES-2"
    WeeklyAttentionSes1 = "PSYCH_WEEKLY_ATTENTION_SES-1"
    WeeklyAttentionSes2 = "PSYCH_WEEKLY_ATTENTION_SES-2"
    Vigilance = "VIGILANCE"
    TaskAttention = "TASK_ATTENTION"
    WeeklyAttention = "WEEKLY_ATTENTION"

    def root_dir(self) -> Path:
        return {
            UpdatedDataset.Learning: DATA / "Rest_v_LearningRecall",
            UpdatedDataset.Osteo: DATA / "Rest_w_Healthy_v_OsteoPain",
            UpdatedDataset.Parkinsons: DATA / "Park_v_Control",
            UpdatedDataset.Bilinguality: DATA / "Rest_w_Bilinguality",
            UpdatedDataset.Depression: DATA / "Rest_w_Depression_v_Control",
            UpdatedDataset.Older: DATA / "Rest_w_Older_v_Younger",
            UpdatedDataset.VigilanceSes1: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.VigilanceSes2: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttentionSes1: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttentionSes2: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttentionSes1: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttentionSes2: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.Vigilance: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttention: DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttention: DATA / "Rest_w_VigilanceAttention",
        }[self]

    def participants_file(self) -> Optional[Path]:
        root = self.root_dir()
        root = Path(sorted(filter(lambda p: p.is_dir(), root.glob("*")))[0])
        return {
            UpdatedDataset.Learning: root / "participants.tsv",
            UpdatedDataset.Osteo: root / "participants.tsv",
            UpdatedDataset.Parkinsons: root / "demographics.csv",
            UpdatedDataset.Bilinguality: root / "participants.tsv",
            UpdatedDataset.Depression: root / "participants.tsv",
            UpdatedDataset.Older: None,
            UpdatedDataset.VigilanceSes1: root / "participants.tsv",
            UpdatedDataset.VigilanceSes2: root / "participants.tsv",
            UpdatedDataset.TaskAttentionSes1: root / "participants.tsv",
            UpdatedDataset.TaskAttentionSes2: root / "participants.tsv",
            UpdatedDataset.WeeklyAttentionSes1: root / "participants.tsv",
            UpdatedDataset.WeeklyAttentionSes2: root / "participants.tsv",
            UpdatedDataset.Vigilance: root / "participants.tsv",
            UpdatedDataset.TaskAttention: root / "participants.tsv",
            UpdatedDataset.WeeklyAttention: root / "participants.tsv",
        }[self]

    def subgroup_names(self) -> list[str]:
        return {
            UpdatedDataset.Learning: ["task", "rest"],
            UpdatedDataset.Osteo: ["allpain", "nopain", "duloxetine", "pain"],
            UpdatedDataset.Parkinsons: [
                "control",
                "parkinsons",
                "control_pre",
                "park_pre",
            ],
            UpdatedDataset.Bilinguality: ["", ""],
            UpdatedDataset.Depression: ["", ""],
            UpdatedDataset.Older: ["", ""],
            UpdatedDataset.VigilanceSes1: ["high", "low"],
            UpdatedDataset.VigilanceSes2: ["high", "low"],
            UpdatedDataset.TaskAttentionSes1: ["high", "low"],
            UpdatedDataset.TaskAttentionSes2: ["high", "low"],
            UpdatedDataset.WeeklyAttentionSes1: ["high", "low"],
            UpdatedDataset.WeeklyAttentionSes2: ["high", "low"],
            UpdatedDataset.Vigilance: ["high", "low"],
            UpdatedDataset.TaskAttention: ["high", "low"],
            UpdatedDataset.WeeklyAttention: ["high", "low"],
        }[self]


class PreprocLevel(Enum):
    BrainExtract = 0
    SliceTimeAlign = 1
    MotionCorrect = 2
    MNIRegister = 3

    def eig_files(self, root: Path) -> List[Path]:
        globs = {
            PreprocLevel.BrainExtract: "*bold_extracted.npy",
            PreprocLevel.SliceTimeAlign: "*slicetime-corrected.npy",
            PreprocLevel.MotionCorrect: "*motion-corrected.npy",
            PreprocLevel.MNIRegister: "*mni-reg.npy",
        }
        glob = globs[self]
        return sorted(root.rglob(glob))


class TrimMethod(Enum):
    Largest = "largest"
    Middle = "middle"
    Precision = "precision"


class NormMethod(Enum):
    pass
