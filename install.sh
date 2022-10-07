#!/bin/bash
PYTHON="$PYENV_ROOT/versions/3.10.4/bin/python"
$PYTHON --version || exit 1
$PYTHON -m pip install --upgrade pip
$PYTHON -m venv .venv
echo "Created virtual environment in .venv"

PYTHON=".venv/bin/python"
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install empyricalRMT==1.1.1
$PYTHON -m pip install matplotlib numpy pandas typing_extensions tqdm seaborn statsmodels scikit-learn scipy nibabel pytest mypy isort flake8