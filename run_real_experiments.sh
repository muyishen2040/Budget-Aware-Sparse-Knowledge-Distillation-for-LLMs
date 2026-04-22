#!/bin/bash
set -euo pipefail   # stop immediately on any error, unset variable, or pipe failure

# Ensure 'src' is in the python path
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ==========================================
# Real Experiment Parameters Setup
# ==========================================

DATASET=wikitext          # main dataset (WikiText-103)
# DATASET=github-code-python  # second experiment (GitHub Code, Python only)
# DATASET=pubmed


NUM_TRAIN_SAMPLES=200000
SEQ_LEN=256
BATCH_SIZE=16
EPOCHS=3
LR=5e-5
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

# # 2. Caching for Sparse Methods First (Top-K cache, K=16 is the max budget)
echo "2. Caching Teacher Predictions..."
python -u scripts/cache_teacher.py \
    --mode topk \
    --dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --topk_k 16 \
    --cache_dir ${CACHE_DIR}_topk

# 3. Top-K Training (3 epochs, K=4,8,16 — fair comparison with adaptive)
echo "3. Running Top-K Training (3 epochs)..."
for K in "${K_VALUES[@]}"; do
    echo "--- Training Top-K with K=${K}, ${EPOCHS} epochs ---"
    OUTPUT=$(python -u scripts/train_topk_kd.py \
        --k $K \
        --dataset $DATASET \
        --cache_dir ${CACHE_DIR} \
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
        --cache_path ${CACHE_DIR}/topk_train.pt
done

# 4. Sampling-based KD
echo "4. Running Sampling KD..."
for K in "${K_VALUES[@]}"; do
    echo "--- Caching Teacher Sampling with K=${K} withdrawals ---"
    python -u scripts/cache_teacher.py \
        --mode sampling \
        --dataset $DATASET \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --seq_len $SEQ_LEN \
        --batch_size $BATCH_SIZE \
        --sampling_num_draws $K \
        --cache_dir ${CACHE_DIR}_sampling_k${K}

    echo "--- Training Sampling with K=${K} ---"
    OUTPUT=$(python -u scripts/train_sampling_kd.py \
        --k $K \
        --dataset $DATASET \
        --cache_dir ${CACHE_DIR}_sampling_k${K} \
        --output_dir ${OUTPUT_PREFIX}_sampling_k${K} \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --alpha $ALPHA \
        --lr $LR | tee /dev/tty)
        
    TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
    RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
    TRAIN_LOSS=${TRAIN_LOSS:-0.0}
    RUN_TIME=${RUN_TIME:-0.0}

    python scripts/evaluate.py \
        --model_path ${OUTPUT_PREFIX}_sampling_k${K} \
        --method sampling \
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
        --cache_path ${CACHE_DIR}_sampling_k${K}/sampling_train.pt
done

# # 5. Adaptive Top-K KD (token-wise K in {4, 8, 16} based on teacher entropy)
echo "5. Running Adaptive Top-K KD..."
ENTROPY_LOW=1.5
ENTROPY_HIGH=3.5

echo "--- Caching Adaptive Top-K Teacher Predictions ---"
python -u scripts/cache_teacher.py \
    --mode adaptive_topk \
    --dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --entropy_low $ENTROPY_LOW \
    --entropy_high $ENTROPY_HIGH \
    --cache_dir ${CACHE_DIR}_adaptive_topk

echo "--- Training Adaptive Top-K ---"
OUTPUT=$(python -u scripts/train_adaptive_topk_kd.py \
    --dataset $DATASET \
    --cache_dir ${CACHE_DIR}_adaptive_topk \
    --output_dir ${OUTPUT_PREFIX}_adaptive_topk \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --lr $LR | tee /dev/tty)

TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
AVG_K=$(echo "$OUTPUT" | grep "METRICS_AVG_K" | cut -d'=' -f2)
TRAIN_LOSS=${TRAIN_LOSS:-0.0}
RUN_TIME=${RUN_TIME:-0.0}
AVG_K=${AVG_K:-16.0}

python scripts/evaluate.py \
    --model_path ${OUTPUT_PREFIX}_adaptive_topk \
    --method adaptive_topk \
    --avg_k $AVG_K \
    --log_file experiment_log.csv \
    --train_dataset $DATASET \
    --val_dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --lr $LR \
    --train_loss $TRAIN_LOSS \
    --run_time_seconds $RUN_TIME \
    --cache_path ${CACHE_DIR}_adaptive_topk/adaptive_topk_train.pt

# echo ""
# echo "=========================================="
# echo "6. Running Adaptive Top-K Tail Summary KD..."
# echo "=========================================="
# Note: we reuse the identical cache from Adaptive Top-K because the raw probabilities 
# were perfectly preserved, so we don't need to re-cache anything!

# echo "--- Training Adaptive Top-K Tail Summary ---"
# OUTPUT=$(python -u scripts/train_adaptive_topk_tail_summary.py \
#     --dataset $DATASET \
#     --cache_dir ${CACHE_DIR}_adaptive_topk \
#     --output_dir ${OUTPUT_PREFIX}_adaptive_topk_tail_summary \
#     --num_epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --alpha $ALPHA \
#     --tail_weight 0.1 \
#     --lr $LR | tee /dev/tty)
    

# TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
# RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
# AVG_K=$(echo "$OUTPUT" | grep "METRICS_AVG_K" | cut -d'=' -f2)
# TRAIN_LOSS=${TRAIN_LOSS:-0.0}
# RUN_TIME=${RUN_TIME:-0.0}
# AVG_K=${AVG_K:-16.0}

# python scripts/evaluate.py \
#     --model_path ${OUTPUT_PREFIX}_adaptive_topk_tail_summary \
#     --method adaptive_topk \
#     --avg_k $AVG_K \
#     --log_file experiment_log.csv \
#     --train_dataset $DATASET \
#     --val_dataset $DATASET \
#     --num_train_samples $NUM_TRAIN_SAMPLES \
#     --epochs $EPOCHS \
#     --train_batch_size $BATCH_SIZE \
#     --lr $LR \
#     --train_loss $TRAIN_LOSS \
#     --run_time_seconds $RUN_TIME \
#     --cache_path ${CACHE_DIR}_adaptive_topk/adaptive_topk_train.pt

# echo "=========================================="
# echo "All done!"

echo ""
echo "=========================================="
echo "7. Running Head-Mass Weighted Adaptive Top-K KD..."
echo "=========================================="

echo "--- Training Head-Mass Weighted Adaptive Top-K ---"
OUTPUT=$(python -u scripts/train_adaptive_topk_weighted.py \
    --dataset $DATASET \
    --cache_dir ${CACHE_DIR}_adaptive_topk \
    --output_dir ${OUTPUT_PREFIX}_adaptive_topk_weighted \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --no_normalize_weights \
    --lr $LR | tee /dev/tty)

TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
AVG_K=$(echo "$OUTPUT" | grep "METRICS_AVG_K" | cut -d'=' -f2)
TRAIN_LOSS=${TRAIN_LOSS:-0.0}
RUN_TIME=${RUN_TIME:-0.0}
AVG_K=${AVG_K:-16.0}

python scripts/evaluate.py \
    --model_path ${OUTPUT_PREFIX}_adaptive_topk_weighted \
    --method adaptive_topk \
    --avg_k $AVG_K \
    --log_file experiment_log.csv \
    --train_dataset $DATASET \
    --val_dataset $DATASET \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --lr $LR \
    --train_loss $TRAIN_LOSS \
    --run_time_seconds $RUN_TIME \
    --cache_path ${CACHE_DIR}_adaptive_topk/adaptive_topk_train.pt

echo "=========================================="
echo "All done!"
