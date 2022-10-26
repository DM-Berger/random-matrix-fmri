from enum import Enum


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


class PreprocLevel(Enum):
    BrainExtract = 0
    SliceTimeAlign = 1
    FullRegister = 2

class TrimMethod(Enum):
    Largest = "largest"
    Middle = "middle"
    Precision = "precision"


class NormMethod(Enum):
    pass