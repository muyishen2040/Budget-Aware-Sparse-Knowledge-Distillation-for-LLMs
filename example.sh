#!/bin/bash
set -e

echo "=== Sparse KD Example Pipeline ==="

# 1. Cache teacher outputs
echo "1) Caching teacher outputs (Top-K = 8)..."
python scripts/cache_teacher.py \
    --mode topk \
    --seq_len 256 \
    --batch_size 4 \
    --num_train_samples 2000 \
    --cache_dir teacher_cache \
    --topk_k 8

# 2. Train student models using the different methods
echo -e "\n2) Training Full KD Baseline..."
python scripts/train_full_kd.py \
    --batch_size 4 \
    --num_epochs 1 \
    --num_train_samples 2000 \
    --output_dir output/full_kd

echo -e "\n3) Training Top-K KD Baseline..."
python scripts/train_topk_kd.py \
    --batch_size 4 \
    --num_epochs 1 \
    --cache_dir teacher_cache \
    --output_dir output/topk_kd \
    --k 8

# 3. Evaluate the trained models
echo -e "\n4) Evaluating Full KD..."
python scripts/evaluate.py \
    --model_path output/full_kd \
    --method full \
    --batch_size 4

echo -e "\n5) Evaluating Top-K KD..."
python scripts/evaluate.py \
    --model_path output/topk_kd \
    --method topk \
    --k 8 \
    --cache_path teacher_cache/topk_train.pt

echo -e "\nExample sequence completed successfully!"
