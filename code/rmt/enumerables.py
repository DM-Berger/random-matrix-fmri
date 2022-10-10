from enum import Enum


class Dataset(Enum):
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
