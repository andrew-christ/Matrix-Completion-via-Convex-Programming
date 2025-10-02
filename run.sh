#!/bin/bash
# Usage: ./run.sh path/to/script.py [args...]
# Example: ./run.sh matrix_completion/numpy/NNMF.py --epochs 10 --lr 0.01
# (NOTE) Run command to make it executable: chmod +x run.sh

set -e  # stop if any command fails

# 1) Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ No .venv found. Create one first with: python -m venv .venv"
    exit 1
fi
source .venv/bin/activate

# 2) Ensure required packages are installed
if [ ! -f "requirements.txt" ]; then
    echo "❌ No requirements.txt found in project root."
    exit 1
fi
pip install --quiet -r requirements.txt

# 3) Run the specified Python file
if [ $# -lt 1 ]; then
    echo "❌ No script specified."
    echo "Usage: ./run.sh path/to/script.py [args...]"
    exit 1
fi

python -m "$@"