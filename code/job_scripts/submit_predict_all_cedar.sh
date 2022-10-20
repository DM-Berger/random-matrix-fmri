#!/bin/bash
#SBATCH --account=rrg-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=all_obs
#SBATCH --output=all_observables_%A_%a_%j.out
#SBATCH --array=0-27
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/random-matrix-fmri"
CODE="$PROJECT/code"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$CODE/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/predict_everything_array.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"