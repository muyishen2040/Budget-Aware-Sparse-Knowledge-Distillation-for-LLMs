# Noise-Aware Sparse Knowledge Distillation for LLMs

## Overview

Standard Knowledge Distillation (KD) for language models minimizes the KL-divergence between a student and teacher across the entire vocabulary at every token position. The conventional assumption is that Full KD, which uses the complete teacher distribution, serves as an upper bound that sparse methods can only approximate.

This project challenges that assumption. Through empirical analysis of the teacher's soft-label distribution, we demonstrate that Full KD forces small student models to memorize **destructive tail noise** in the teacher's output. When the teacher is uncertain, it scatters significant probability mass across thousands of irrelevant tokens. A small student (160M parameters) wastes its limited capacity fitting this noise rather than learning meaningful linguistic structure.

Our approach, **Weighted Adaptive Top-K KD**, dynamically assigns a per-token information budget based on teacher entropy and weights each token's distillation loss by the teacher's own confidence (head probability mass). This achieves **38.82 PPL** — a 2.10 PPL improvement over Full KD (40.92) — while using only ~10 values per token instead of 50,277 (a 5,000x storage reduction).

## Key Findings

- The Pythia-1.4B teacher abandons **43.28%** of its probability mass outside the Top-16 predictions for **39.1%** of training tokens (high entropy positions), scattered across an average of 2,330 noisy tokens.
- Full KD **degrades** from 40.48 PPL (epoch 1) to 40.92 PPL (epoch 3) on WikiText-103, indicating noise overfitting.
- Our adaptive method **improves** across epochs, reaching 38.82 PPL at epoch 3.

## Results

All experiments use Pythia-1.4B (teacher) and Pythia-160M (student) on WikiText-103 with 200K training samples, alpha=0.5, LR=5e-5 with linear warmup.

| Method | Budget (per token) | Best PPL |
|--------|-------------------|----------|
| **Weighted Adaptive Top-K** | **~10 (avg)** | **38.82** |
| Adaptive Top-K | ~10 (avg) | 40.00 |
| Full KD | 50,277 | 40.48 |
| Top-K (K=16) | 32 | 41.91 |
| Top-K (K=8) | 16 | 45.23 |
| Top-K (K=4) | 8 | 47.04 |

The Weighted Adaptive Top-K method outperforms Full KD by 2.10 PPL while being 2.7x faster in training time (~1,400s vs ~3,770s for 3 epochs) and requiring 5,000x less per-token teacher storage. Fixed Top-K methods degrade at 3 epochs (K=16 drops from 41.91 to 47.75), confirming that noise overfitting is a systemic issue in multi-epoch KD without adaptive filtering.

## Project Structure

```
sparse_kd/
├── src/
│   ├── data.py              # Dataset loading and cached dataloader utilities
│   ├── losses.py            # All KD loss functions (Full, Top-K, Sampling, Adaptive, Weighted)
│   ├── models.py            # Teacher and student model loading
│   └── eval_utils.py        # Evaluation metric computation
├── scripts/
│   ├── cache_teacher.py     # Offline teacher soft-label caching
│   ├── train_full_kd.py     # Full KD training (online teacher)
│   ├── train_topk_kd.py     # Top-K KD training (from cache)
│   ├── train_sampling_kd.py # Sampling KD training (from cache)
│   ├── train_adaptive_topk_kd.py       # Adaptive Top-K KD training
│   ├── train_adaptive_topk_weighted.py  # Weighted Adaptive Top-K KD training
│   ├── train_adaptive_topk_tail_summary.py # Tail Summary experiment
│   ├── evaluate.py          # Unified evaluation and CSV logging
│   ├── analyze_training_entropy.py     # Training set entropy analysis
│   └── analyze_teacher_tail_noise.py   # Teacher tail noise quantification
├── run_real_experiments.sh  # Main experiment runner
├── experiment_log.csv       # All experiment results
└── requirements.txt         # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- A GPU with at least 16 GB VRAM (for teacher model inference)

### Installation

```bash
git clone <repo-url> sparse_kd
cd sparse_kd
# Activate your virtual environment
pip install -r requirements.txt
```

## Running Experiments

### Full Pipeline

The main experiment script handles teacher caching, student training, and evaluation end-to-end:

```bash
sh run_real_experiments.sh
```

This script executes the following steps:
1. Evaluates raw teacher and student baselines.
2. Runs Full KD training (online, 3 epochs).
3. Caches teacher soft-labels for Top-K and Adaptive modes.
4. Trains Top-K KD for K in {4, 8, 16}.
5. Trains Adaptive Top-K KD (entropy-based K selection).
6. Trains Weighted Adaptive Top-K KD (head-mass confidence weighting).
7. Evaluates all models and logs results to `experiment_log.csv`.

Configuration parameters are set at the top of the script:

```bash
DATASET=wikitext          # Dataset: wikitext, github-code-python, pubmed
NUM_TRAIN_SAMPLES=200000  # Number of training samples
SEQ_LEN=256               # Sequence length
BATCH_SIZE=16             # Batch size
EPOCHS=3                  # Training epochs
LR=5e-5                   # Learning rate
ALPHA=0.5                 # CE vs KD loss balance
```

Individual steps can be commented in or out by editing the script.

### Manual Execution

To run individual components:

```bash
# Set up environment
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Cache teacher soft-labels (adaptive mode)
python scripts/cache_teacher.py \
    --mode adaptive_topk \
    --dataset wikitext \
    --num_train_samples 200000 \
    --seq_len 256 \
    --batch_size 16 \
    --cache_dir teacher_cache_real_wikitext_adaptive_topk

# Train Weighted Adaptive Top-K
python scripts/train_adaptive_topk_weighted.py \
    --cache_dir teacher_cache_real_wikitext_adaptive_topk \
    --output_dir output/real_wikitext_adaptive_topk_weighted \
    --num_epochs 3 \
    --batch_size 16 \
    --alpha 0.5 \
    --min_weight 0.2 \
    --no_normalize_weights \
    --lr 5e-5

# Evaluate
python scripts/evaluate.py \
    --model_path output/real_wikitext_adaptive_topk_weighted \
    --method adaptive_topk \
    --log_file experiment_log.csv \
    --train_dataset wikitext \
    --val_dataset wikitext
```

### Analysis Scripts

To reproduce the teacher tail noise analysis:

```bash
python scripts/analyze_training_entropy.py
python scripts/analyze_teacher_tail_noise.py
```

## Conclusion

The central finding of this project is that in distillation with a large teacher-student capacity gap, the teacher's full distribution is not an ideal supervision target. The teacher's uncertainty manifests as diffuse probability mass across thousands of irrelevant tokens, which a small student model cannot productively absorb. By adaptively pruning this noise and weighting the distillation signal by teacher confidence, we achieve better student performance with dramatically less storage and computation. The future of efficient distillation lies not in preserving more teacher information, but in preserving the right information.
