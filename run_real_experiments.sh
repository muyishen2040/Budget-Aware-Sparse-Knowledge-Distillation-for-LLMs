#!/bin/bash
set -euo pipefail   # stop immediately on any error, unset variable, or pipe failure

# Ensure 'src' is in the python path
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ==========================================
# Real Experiment Parameters Setup
# ==========================================

# DATASET=wikitext          # main dataset (WikiText-103)
# DATASET=github-code-python  # second experiment (GitHub Code, Python only)
DATASET=pubmed


NUM_TRAIN_SAMPLES=50000
SEQ_LEN=256
BATCH_SIZE=16
EPOCHS=1
LR=5e-6
ALPHA=0.5

# Budgets to compare
K_VALUES=(4 8 16)

# ------------------------------------------
# Derived path prefixes — change DATASET above,
# all cache and output dirs update automatically.
# Existing wiki dirs (teacher_cache_real, output/real_*) are untouched.
# ------------------------------------------
CACHE_DIR="teacher_cache_real_${DATASET}"        # e.g. teacher_cache_real_wikitext
OUTPUT_PREFIX="output/real_${DATASET}"           # e.g. output/real_wikitext

echo "Starting Real Experiments..."
echo "Train Samples: $NUM_TRAIN_SAMPLES"
echo "Sequence Length: $SEQ_LEN"
echo "Budgets: ${K_VALUES[*]}"
echo "Dataset: $DATASET"
echo "Cache dir: $CACHE_DIR"
echo "Output prefix: $OUTPUT_PREFIX"
echo "=========================================="

echo "0. Evaluating Raw Teacher Baseline (Pythia-1.4b)..."
python scripts/evaluate.py \
    --model_path EleutherAI/pythia-1.4b \
    --method full \
    --log_file experiment_log.csv \
    --batch_size $BATCH_SIZE \
    --train_dataset $DATASET \
    --val_dataset $DATASET


# 0. Evaluate Raw Student Baseline
echo "0. Evaluating Raw Student Baseline (Pythia-160m)..."
python scripts/evaluate.py \
    --model_path EleutherAI/pythia-160m \
    --method full \
    --log_file experiment_log.csv \
    --batch_size $BATCH_SIZE \
    --train_dataset $DATASET \
    --val_dataset $DATASET

# 1. Full KD (Online)
echo "1. Running Full KD..."
OUTPUT=$(python -u scripts/train_full_kd.py \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --dataset $DATASET \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --alpha $ALPHA \
    --num_epochs $EPOCHS \
    --output_dir ${OUTPUT_PREFIX}_full_kd | tee /dev/tty)

TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)

TRAIN_LOSS=${TRAIN_LOSS:-0.0}
RUN_TIME=${RUN_TIME:-0.0}

python scripts/evaluate.py \
    --model_path ${OUTPUT_PREFIX}_full_kd \
    --method full \
    --log_file experiment_log.csv \
    --train_dataset $DATASET \
    --val_dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --lr $LR \
    --train_loss $TRAIN_LOSS \
    --run_time_seconds $RUN_TIME

# 2. Caching for Sparse Methods First (Top-K cache, K=16 is the max budget)
echo "2. Caching Teacher Predictions..."
python -u scripts/cache_teacher.py \
    --mode topk \
    --dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --topk_k 16 \
    --cache_dir ${CACHE_DIR}_topk

# 3. Top-K Training (For varying budgets)
echo "3. Running Top-K Training..."
for K in "${K_VALUES[@]}"; do
    echo "--- Training Top-K with K=${K} ---"
    OUTPUT=$(python -u scripts/train_topk_kd.py \
        --k $K \
        --dataset $DATASET \
        --cache_dir ${CACHE_DIR}_topk \
        --output_dir ${OUTPUT_PREFIX}_topk_k${K} \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --alpha $ALPHA \
        --lr $LR | tee /dev/tty)
        
    TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
    RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
    TRAIN_LOSS=${TRAIN_LOSS:-0.0}
    RUN_TIME=${RUN_TIME:-0.0}

    python scripts/evaluate.py \
        --model_path ${OUTPUT_PREFIX}_topk_k${K} \
        --method topk \
        --k $K \
        --log_file experiment_log.csv \
        --train_dataset $DATASET \
        --val_dataset $DATASET \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --epochs $EPOCHS \
        --train_batch_size $BATCH_SIZE \
        --lr $LR \
        --train_loss $TRAIN_LOSS \
        --run_time_seconds $RUN_TIME \
        --cache_path ${CACHE_DIR}_topk/topk_train.pt
done

# 4. Sampling-based KD
# echo "4. Running Sampling KD..."
# for K in "${K_VALUES[@]}"; do
#     echo "--- Caching Teacher Sampling with K=${K} withdrawals ---"
#     python -u scripts/cache_teacher.py \
#         --mode sampling \
#         --dataset $DATASET \
#         --num_train_samples $NUM_TRAIN_SAMPLES \
#         --seq_len $SEQ_LEN \
#         --batch_size $BATCH_SIZE \
#         --sampling_num_draws $K \
#         --cache_dir ${CACHE_DIR}_sampling_k${K}

#     echo "--- Training Sampling with K=${K} ---"
#     OUTPUT=$(python -u scripts/train_sampling_kd.py \
#         --k $K \
#         --dataset $DATASET \
#         --cache_dir ${CACHE_DIR}_sampling_k${K} \
#         --output_dir ${OUTPUT_PREFIX}_sampling_k${K} \
#         --num_epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --alpha $ALPHA \
#         --lr $LR | tee /dev/tty)
        
#     TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
#     RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
#     TRAIN_LOSS=${TRAIN_LOSS:-0.0}
#     RUN_TIME=${RUN_TIME:-0.0}

#     python scripts/evaluate.py \
#         --model_path ${OUTPUT_PREFIX}_sampling_k${K} \
#         --method sampling \
#         --k $K \
#         --log_file experiment_log.csv \
#         --train_dataset $DATASET \
#         --val_dataset $DATASET \
#         --num_train_samples $NUM_TRAIN_SAMPLES \
#         --epochs $EPOCHS \
#         --train_batch_size $BATCH_SIZE \
#         --lr $LR \
#         --train_loss $TRAIN_LOSS \
#         --run_time_seconds $RUN_TIME \
#         --cache_path ${CACHE_DIR}_sampling_k${K}/sampling_train.pt
# done
