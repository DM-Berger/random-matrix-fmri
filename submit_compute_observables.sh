#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=observables
#SBATCH --output=observables_%j.out
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

bash run_python.sh code/rmt/dataset.py
