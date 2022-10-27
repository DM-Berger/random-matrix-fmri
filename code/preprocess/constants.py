from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/updated"

PSYCH_HIGH_RES = DATA / "Rest_w_VigilanceAttention"
PARKINSONS = DATA / "Park_v_Control"
BILINGUALITY = DATA / "Rest_w_Bilinguality"
OSTEOPATHIC = DATA / "Rest_w_Healthy_v_OsteoPain"

# fmt: off
SUBJECTS_TO_DROP = [
    PSYCH_HIGH_RES / "ds001168-download/sub-11/ses-1/func/sub-11_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz",  # noqa
    PSYCH_HIGH_RES / "ds001168-download/sub-11/ses-1/func/sub-11_ses-1_task-rest_acq-fullbrain_run-2_bold.nii.gz",  # noqa
    PSYCH_HIGH_RES / "ds001168-download/sub-11/ses-2/func/sub-11_ses-2_task-rest_acq-fullbrain_run-1_bold.nii.gz",  # noqa
    PSYCH_HIGH_RES / "ds001168-download/sub-11/ses-2/func/sub-11_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz",  # noqa
    BILINGUALITY / 'ds001747-download/sub-3957/func/sub-3957_task-rest_run-01_bold.nii.gz',  # noqa
    BILINGUALITY / 'ds001747-download/sub-3925/func/sub-3925_task-rest_run-01_bold.nii.gz',  # noqa
    PARKINSONS / "ds001907-download/sub-RC4121/ses-2/func/sub-RC4121_ses-2_task-rest_bold.nii.gz",  # noqa
    OSTEOPATHIC / 'ds000208-download/sub-59/func/sub-59_task-rest_bold.nii.gz',  # noqa
    OSTEOPATHIC / 'ds000208-download/sub-70/func/sub-70_task-rest_bold.nii.gz',  # noqa
]
