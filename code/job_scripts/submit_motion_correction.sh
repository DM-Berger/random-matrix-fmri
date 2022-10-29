#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=motioncorr
#SBATCH --output=motioncorr_%j.out
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80


SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/random-matrix-fmri"
SCRIPT="$PROJECT/code/rmt/preprocess/motion_correct.py"
source .venv_preproc/bin/activate
python "$SCRIPT"
