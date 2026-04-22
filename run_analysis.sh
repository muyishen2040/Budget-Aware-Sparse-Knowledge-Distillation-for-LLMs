#!/bin/bash
set -euo pipefail

# Activate the PyTorch environment
source /opt/pytorch/bin/activate

# Ensure 'src' is in the python path
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=========================================="
echo "Analyzing Teacher Training Entropy Distribution"
echo "=========================================="
python scripts/analyze_training_entropy.py --num_train_samples 2000

echo ""
echo "=========================================="
echo "Analyzing Teacher Tail Noise"
echo "=========================================="
python scripts/analyze_teacher_tail_noise.py --num_train_samples 2000

echo ""
echo "Analysis completely strictly."
