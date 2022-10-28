#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=all_obs
#SBATCH --output=freesurf_%A_%a_%j.out
#SBATCH --array=1-22
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

ID=$(seq -f "%02g" "$SLURM_ARRAY_TASK_ID")

SCRATCH="$(readlink -f "$SCRATCH")"

PROJECT="$SCRATCH/random-matrix-fmri"
FILE="$PROJECT/data/updated/Rest_w_VigilanceAttention/ds001168-download/sub-$ID/ses-1/anat/sub-$ID""_ses-1_T1w.nii.gz"
OUTDIR="$PROJECT/data/updated/Rest_w_VigilanceAttention/ds001168-download/sub-$ID/ses-1/anat"
export SUBJECTS_DIR="$OUTDIR/freesurfer"

module load freesurfer/7.1.0
recon-all -subject "sub-$ID""_ses-1" -i "$FILE"