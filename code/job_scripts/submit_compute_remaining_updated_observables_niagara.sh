#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=remain_obs
#SBATCH --array=0-15
#SBATCH --output=remaining_observables_%A_%a_%j.out
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/random-matrix-fmri"
CODE="$PROJECT/code"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$CODE/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/compute_remaining_updated_observables.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"