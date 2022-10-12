#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=all_obs
#SBATCH --output=all_observables_%j.out
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
CODE="$(dirname "$THIS_SCRIPT_DIR")"
PROJECT="$(dirname "$CODE")"
JOB_SCRIPTS="$CODE/job_scripts"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$CODE/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/compute_all_observables.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"