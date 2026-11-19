#!/bin/bash
#SBATCH --job-name=op_embed
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SPECTER-embed the open-problem closing sections. Run as a batch job because the
# login-node arbiter SIGKILLs this workload ~6 min in (observed exit 137 twice).
# Chunked + resumable, so a re-submit picks up from the last completed chunk.
set -e
cd "${SLURM_SUBMIT_DIR:-.}"   # submit from the repo root: sbatch -A <account> pipeline/07_open_problems/embed_op_sections.sbatch

unset PYTHONHOME PYTHONPATH
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # model is cached; nodes lack internet
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export EMB_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export EMB_BATCH=32

python pipeline/07_open_problems/embed_op_sections.py
echo "EXIT $?"
