#!/bin/bash
#SBATCH --account=rrg-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=pred_tseries
#SBATCH --output=predict_tseries_%A_%a_%j.out
#SBATCH --array=0-9
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0

module load nixpkgs/16.09 intel/2018.3 fsl/6.0.1

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/random-matrix-fmri"
CODE="$PROJECT/code"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$CODE/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/predict_tseries_array.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"