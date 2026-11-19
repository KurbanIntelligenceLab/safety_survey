#!/bin/bash
#SBATCH --job-name=op_probe
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Per-label one-vs-rest AUC probe over ALL labeled papers (see op_label_probe.py).
set -e
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repo root: sbatch -A <account> pipeline/07_open_problems/op_label_probe.sbatch
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python pipeline/07_open_problems/op_label_probe.py
echo "EXIT $?"
