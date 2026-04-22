#!/usr/bin/env bash
# =============================================================================
# run_gold_coverage.sh
# =============================================================================
# Evaluate the Top-K gold-label coverage of a distilled student (or any
# causal LM) on a chosen dataset, sweeping over multiple budget values K.
#
# For each K, it reports what percentage of ground-truth next tokens fall
# OUTSIDE the model's Top-K predictions — a direct measure of information
# loss suffered by a student trained with a budget-K sparse teacher.
#
# Requirements
# ------------
#   • A trained student model saved as a HuggingFace checkpoint, e.g.:
#       output/real_topk_k16/
#   • Or any hub model ID, e.g.: EleutherAI/pythia-160m
#
# Quick start
# -----------
#   # Evaluate the real_topk_k16 student on wikitext val (defaults)
#   bash run_gold_coverage.sh
#
#   # Evaluate a specific checkpoint on github-code-python
#   MODEL_PATH=output/real_full_kd \
#   DATASET=github-code-python \
#   SPLIT=val \
#   bash run_gold_coverage.sh
#
#   # Compare several students — call the script in a loop
#   for MODEL in output/real_topk_k4 output/real_topk_k8 output/real_topk_k16; do
#       MODEL_PATH=$MODEL bash run_gold_coverage.sh
#   done
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override any of these with environment variables
# ---------------------------------------------------------------------------

# Path to a HuggingFace model directory or hub model ID.
# MODEL_PATH="${MODEL_PATH:-output/real_topk_k16}"
MODEL_PATH="${MODEL_PATH:-output/real_full_kd}"

# Optional separate tokenizer path (leave empty to use MODEL_PATH's tokenizer,
# or the teacher tokenizer as fallback).
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# Dataset key: wikitext | github-code | github-code-python | pubmed
DATASET="${DATASET:-wikitext}"

# Split to evaluate: train | val
SPLIT="${SPLIT:-val}"

# Number of text samples to load for evaluation.
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

# Sequence length (must match what was used during training).
SEQ_LEN="${SEQ_LEN:-256}"

# Batch size for forward passes (reduce if OOM).
BATCH_SIZE="${BATCH_SIZE:-8}"

# Space-separated list of Top-K budgets to sweep.
BUDGETS="${BUDGETS:-4 8 16 32 64}"

# Where to write CSV output (leave empty to skip).
OUT_CSV="${OUT_CSV:-output/gold_coverage_${DATASET}_${SPLIT}.csv}"

# ---------------------------------------------------------------------------
# Derived args
# ---------------------------------------------------------------------------

TOKENIZER_ARG=""
if [ -n "${TOKENIZER_PATH}" ]; then
    TOKENIZER_ARG="--tokenizer_path ${TOKENIZER_PATH}"
fi

CSV_ARG=""
if [ -n "${OUT_CSV}" ]; then
    mkdir -p "$(dirname "${OUT_CSV}")"
    CSV_ARG="--out_csv ${OUT_CSV}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "=============================================="
echo " Gold-Label Coverage Analysis"
echo "=============================================="
echo "  Model      : ${MODEL_PATH}"
echo "  Dataset    : ${DATASET}"
echo "  Split      : ${SPLIT}"
echo "  Num samples: ${NUM_SAMPLES}"
echo "  Seq length : ${SEQ_LEN}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Budgets    : ${BUDGETS}"
[ -n "${OUT_CSV}" ] && echo "  CSV out    : ${OUT_CSV}"
echo "----------------------------------------------"

python scripts/analyze_gold_coverage.py \
    --model_path  "${MODEL_PATH}" \
    --dataset     "${DATASET}" \
    --split       "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --seq_len     "${SEQ_LEN}" \
    --batch_size  "${BATCH_SIZE}" \
    --budgets     ${BUDGETS} \
    ${TOKENIZER_ARG} \
    ${CSV_ARG}

echo "Done."
