from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from numpy import ndarray

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


class SeriesKind(Enum):
    # measures of location
    Mean = "mean"
    Max = "max"
    Min = "min"
    Median = "median"
    Percentile95 = "p95"
    Percentile05 = "p05"
    # measures of scale
    StdDev = "sd"
    IQR = "iqr"
    Range = "range"
    RobustRange = "r90"

    def suffix(self) -> str:
        return {
            SeriesKind.Mean: ".tseries.mean.npy",
            SeriesKind.Max: ".tseries.max.npy",
            SeriesKind.Min: ".tseries.min.npy",
            SeriesKind.Median: ".tseries.median.npy",
            SeriesKind.Percentile95: ".tseries.p95.npy",
            SeriesKind.Percentile05: ".tseries.p05.npy",
            SeriesKind.StdDev: ".tseries.sd.npy",
            SeriesKind.IQR: ".tseries.iqr.npy",
            SeriesKind.Range: ".tseries.range.npy",
            SeriesKind.RobustRange: ".tseries.r90.npy",
        }[self]

    def reducer(self) -> Callable[[ndarray], ndarray]:
        return {
            # measures of location
            SeriesKind.Mean: lambda x: np.mean(x, axis=0),
            SeriesKind.Max: lambda x: np.max(x, axis=0),
            SeriesKind.Min: lambda x: np.min(x, axis=0),
            SeriesKind.Median: lambda x: np.median(x, axis=0),
            SeriesKind.Percentile95: lambda x: np.percentile(x, 95, axis=0),
            SeriesKind.Percentile05: lambda x: np.percentile(x, 5, axis=0),
            # measures of scale
            SeriesKind.StdDev: lambda x: np.std(x, ddof=1, axis=0),
            SeriesKind.IQR: lambda x: np.abs(
                np.diff(np.percentile(x, q=[25, 75], axis=0), axis=0)
            ),
            SeriesKind.Range: lambda x: np.max(x, axis=0) - np.min(x, axis=0),
            SeriesKind.RobustRange: lambda x: np.abs(
                np.diff(np.percentile(x, q=[5, 95], axis=0), axis=0)
            ),
        }[self]


class PreprocLevel(Enum):
    BrainExtract = 0
    SliceTimeAlign = 1
    MotionCorrect = 2
    MNIRegister = 3


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

    def rmt_dir(self) -> Path:
        return {
            UpdatedDataset.Learning: RMT_DATA / "Rest_v_LearningRecall",
            UpdatedDataset.Osteo: RMT_DATA / "Rest_w_Healthy_v_OsteoPain",
            UpdatedDataset.Parkinsons: RMT_DATA / "Park_v_Control",
            UpdatedDataset.Bilinguality: RMT_DATA / "Rest_w_Bilinguality",
            UpdatedDataset.Depression: RMT_DATA / "Rest_w_Depression_v_Control",
            UpdatedDataset.Older: RMT_DATA / "Rest_w_Older_v_Younger",
            UpdatedDataset.VigilanceSes1: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.VigilanceSes2: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttentionSes1: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttentionSes2: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttentionSes1: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttentionSes2: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.Vigilance: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.TaskAttention: RMT_DATA / "Rest_w_VigilanceAttention",
            UpdatedDataset.WeeklyAttention: RMT_DATA / "Rest_w_VigilanceAttention",
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
            UpdatedDataset.VigilanceSes1: root / "all_participants.json",
            UpdatedDataset.VigilanceSes2: root / "all_participants.json",
            UpdatedDataset.TaskAttentionSes1: root / "all_participants.json",
            UpdatedDataset.TaskAttentionSes2: root / "all_participants.json",
            UpdatedDataset.WeeklyAttentionSes1: root / "all_participants.json",
            UpdatedDataset.WeeklyAttentionSes2: root / "all_participants.json",
            UpdatedDataset.Vigilance: root / "all_participants.json",
            UpdatedDataset.TaskAttention: root / "all_participants.json",
            UpdatedDataset.WeeklyAttention: root / "all_participants.json",
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

    def eig_files(self, preproc_level: PreprocLevel) -> List[Path]:
        globs = {
            PreprocLevel.BrainExtract: "*bold_extracted_eigs.npy",
            PreprocLevel.SliceTimeAlign: "*slicetime-corrected_eigs.npy",
            PreprocLevel.MotionCorrect: "*motion-corrected_eigs.npy",
            PreprocLevel.MNIRegister: "*mni-reg_eigs.npy",
        }
        glob = globs[preproc_level]
        files = sorted(self.rmt_dir().rglob(glob))
        # filter out by session
        if "Ses" in self.name:
            session = self.name[-1]
            files = sorted(filter(lambda p: f"ses-{session}" in p.name, files))
        return files

    def tseries_files(
        self, preproc_level: PreprocLevel, tseries: SeriesKind
    ) -> List[Path]:
        globs = {
            PreprocLevel.BrainExtract: f"*bold_extracted{tseries.suffix()}",
            PreprocLevel.SliceTimeAlign: f"*slicetime-corrected{tseries.suffix()}",
            PreprocLevel.MotionCorrect: f"*motion-corrected{tseries.suffix()}",
            PreprocLevel.MNIRegister: f"*mni-reg_eigs{tseries.suffix()}",
        }
        glob = globs[preproc_level]
        files = sorted(self.rmt_dir().rglob(glob))
        # filter out by session
        if "Ses" in self.name:
            session = self.name[-1]
            files = sorted(filter(lambda p: f"ses-{session}" in p.name, files))
        return files


class TrimMethod(Enum):
    Largest = "largest"
    Middle = "middle"
    Precision = "precision"


class NormMethod(Enum):
    pass


if __name__ == "__main__":
    source = UpdatedDataset.Learning
    path = source.participants_file()
    print(path)
