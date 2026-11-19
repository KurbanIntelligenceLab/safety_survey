#!/bin/bash
#SBATCH --job-name=op_validate
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bottom-up validation of the open-problem label taxonomy (mirrors app:bottomup).
# Batch job for the same reason as embed_op_sections.sbatch: numba JIT + 14
# subfields x 5 UMAP seeds runs past the login-node arbiter's SIGKILL window.
set -e
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repo root: sbatch -A <account> pipeline/07_open_problems/op_cluster_validate.sbatch
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python pipeline/07_open_problems/op_cluster_validate.py
echo "EXIT $?"
