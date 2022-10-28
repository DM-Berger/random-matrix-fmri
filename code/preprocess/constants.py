from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/updated"

PSYCH_HIGH_RES = DATA / "Rest_w_VigilanceAttention"
PARKINSONS = DATA / "Park_v_Control"
BILINGUALITY = DATA / "Rest_w_Bilinguality"
OSTEOPATHIC = DATA / "Rest_w_Healthy_v_OsteoPain"
LEARNING = DATA / "Rest_v_LearningRecall"
DEPRESSION = DATA / "Rest_w_Depression_v_Control"

"""
Reasoning for dropping can be seen from below: we drop scans with abnormal shapes

data                         x_n  y_n  z_n  x_mm  y_mm  z_mm  t    TR       orient  count
Park_v_Control               80   80   43   3.00  3.00  3.00  149  2.40     RPI       552
                             96   114  96   2.00  2.00  2.00  149  2.40     LPI       552
Rest_v_LearningRecall        64   64   36   3.00  3.00  3.00  195  2.00     RPI       432
Rest_w_Bilinguiality         100  100  72   1.80  1.80  1.80  823  0.88     RPI        90
Rest_w_VigilanceAttention    128  128  70   1.50  1.50  1.50  300  3000.00  RPI        84
Rest_w_Healthy_v_OsteoPain   64   64   36   3.44  3.44  3.00  300  2.50     RPI        74
Rest_w_Depression_v_Control  112  112  25   1.96  1.96  5.00  100  2.50     RPI        72
Rest_w_Older_v_Younger       74   74   32   2.97  2.97  4.00  300  2.00     RPI        62
Rest_w_VigilanceAttention    200  60   40   0.75  0.75  0.75  150  4000.00  RPI        44
                             64   64   35   3.00  3.00  3.00  300  3000.00  RPI         4
Park_v_Control               80   80   43   3.00  3.00  3.00  300  2.40     RPI         1
Rest_w_Bilinguiality         100  96   72   1.80  1.80  1.80  823  0.88     RPI         1
                                  100  72   1.80  1.80  1.80  823  0.93     RIA         1
Rest_w_Healthy_v_OsteoPain   64   64   36   3.44  3.44  3.00  244  2.50     RPI         1
                                                              292  2.50     RPI         1


"""

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

BRAIN_EXTRACT_FAILS = [
    PARKINSONS / "ds001907-download/sub-RC4109/ses-1/anat/",
    BILINGUALITY / "ds001747-download/sub-4128/anat/",
    BILINGUALITY / "ds001747-download/sub-3941/anat/",
    DEPRESSION / "ds002748-download/sub-70/anat/",
    DEPRESSION / "ds002748-download/sub-56/anat/",

]