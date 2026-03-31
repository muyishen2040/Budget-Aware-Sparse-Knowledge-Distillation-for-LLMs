#!/bin/bash

# Ensure 'src' is in the python path
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ==========================================
# Real Experiment Parameters Setup
# ==========================================

NUM_TRAIN_SAMPLES=200000
SEQ_LEN=256
BATCH_SIZE=16
EPOCHS=1
LR=5e-5
ALPHA=0.5

# Budgets to compare
K_VALUES=(50)

echo "Starting Real Experiments..."
echo "Train Samples: $NUM_TRAIN_SAMPLES"
echo "Sequence Length: $SEQ_LEN"
echo "Budgets: ${K_VALUES[*]}"
echo "=========================================="

# # 0. Evaluate Raw Student Baseline
# echo "0. Evaluating Raw Student Baseline (Pythia-160m)..."
# python scripts/evaluate.py \
#     --model_path EleutherAI/pythia-160m \
#     --method full \
#     --log_file experiment_log.csv \
#     --batch_size $BATCH_SIZE \
#     --val_dataset wikitext-103-raw-v1

# # 1. Full KD (Online)
# echo "1. Running Full KD..."
# OUTPUT=$(python -u scripts/train_full_kd.py \
#     --num_train_samples $NUM_TRAIN_SAMPLES \
#     --dataset wikitext-103-raw-v1 \
#     --seq_len $SEQ_LEN \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --alpha $ALPHA \
#     --num_epochs $EPOCHS \
#     --output_dir output/real_full_kd | tee /dev/tty)

# TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
# RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)

# TRAIN_LOSS=${TRAIN_LOSS:-0.0}
# RUN_TIME=${RUN_TIME:-0.0}

# python scripts/evaluate.py \
#     --model_path output/real_full_kd \
#     --method full \
#     --log_file experiment_log.csv \
#     --train_dataset wikitext-103-raw-v1 \
#     --val_dataset wikitext-103-raw-v1 \
#     --num_train_samples $NUM_TRAIN_SAMPLES \
#     --epochs $EPOCHS \
#     --train_batch_size $BATCH_SIZE \
#     --lr $LR \
#     --train_loss $TRAIN_LOSS \
#     --run_time_seconds $RUN_TIME

# 2. Caching for Sparse Methods First
# echo "2. Caching Teacher Predictions..."
# python -u scripts/cache_teacher.py \
#     --mode topk \
#     --dataset wikitext-103-raw-v1 \
#     --num_train_samples $NUM_TRAIN_SAMPLES \
#     --seq_len $SEQ_LEN \
#     --batch_size $BATCH_SIZE \
#     --topk_k 16 \
#     --cache_dir teacher_cache_real

# 3. Top-K Training (For varying budgets)
# echo "3. Running Top-K Training..."
# for K in "${K_VALUES[@]}"; do
#     echo "--- Training Top-K with K=${K} ---"
#     OUTPUT=$(python -u scripts/train_topk_kd.py \
#         --k $K \
#         --dataset wikitext-103-raw-v1 \
#         --cache_dir teacher_cache_real \
#         --output_dir output/real_topk_k${K} \
#         --num_epochs $EPOCHS \
#         --batch_size $BATCH_SIZE \
#         --alpha $ALPHA \
#         --lr $LR | tee /dev/tty)
        
#     TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
#     RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
#     TRAIN_LOSS=${TRAIN_LOSS:-0.0}
#     RUN_TIME=${RUN_TIME:-0.0}

#     python scripts/evaluate.py \
#         --model_path output/real_topk_k${K} \
#         --method topk \
#         --k $K \
#         --log_file experiment_log.csv \
#         --train_dataset wikitext-103-raw-v1 \
#         --val_dataset wikitext-103-raw-v1 \
#         --num_train_samples $NUM_TRAIN_SAMPLES \
#         --epochs $EPOCHS \
#         --train_batch_size $BATCH_SIZE \
#         --lr $LR \
#         --train_loss $TRAIN_LOSS \
#         --run_time_seconds $RUN_TIME \
#         --cache_path teacher_cache_real/topk_train.pt
# done

# 4. Sampling-based KD
echo "4. Running Sampling KD..."
for K in "${K_VALUES[@]}"; do
    # echo "--- Caching Teacher Sampling with K=${K} withdrawals ---"
    # python -u scripts/cache_teacher.py \
    #     --mode sampling \
    #     --dataset wikitext-103-raw-v1 \
    #     --num_train_samples $NUM_TRAIN_SAMPLES \
    #     --seq_len $SEQ_LEN \
    #     --batch_size $BATCH_SIZE \
    #     --sampling_num_draws $K \
    #     --cache_dir teacher_cache_real_sampling_k${K}

    echo "--- Training Sampling with K=${K} ---"
    OUTPUT=$(python -u scripts/train_sampling_kd.py \
        --k $K \
        --dataset wikitext-103-raw-v1 \
        --cache_dir teacher_cache_real_sampling_k${K} \
        --output_dir output/real_sampling_k${K} \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --alpha $ALPHA \
        --lr $LR | tee /dev/tty)
        
    TRAIN_LOSS=$(echo "$OUTPUT" | grep "METRICS_TRAIN_LOSS" | cut -d'=' -f2)
    RUN_TIME=$(echo "$OUTPUT" | grep "METRICS_RUN_TIME" | cut -d'=' -f2)
    TRAIN_LOSS=${TRAIN_LOSS:-0.0}
    RUN_TIME=${RUN_TIME:-0.0}

    python scripts/evaluate.py \
        --model_path output/real_sampling_k${K} \
        --method sampling \
        --k $K \
        --log_file experiment_log.csv \
        --train_dataset wikitext-103-raw-v1 \
        --val_dataset wikitext-103-raw-v1 \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --epochs $EPOCHS \
        --train_batch_size $BATCH_SIZE \
        --lr $LR \
        --train_loss $TRAIN_LOSS \
        --run_time_seconds $RUN_TIME \
        --cache_path teacher_cache_real_sampling_k${K}/sampling_train.pt
done
