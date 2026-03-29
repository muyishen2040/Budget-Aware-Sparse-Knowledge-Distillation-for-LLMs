# Budget-Aware Sparse Knowledge Distillation for LLMs

Large language models typically require massive computational and storage overhead during Knowledge Distillation (KD) since they compute the full vocabulary distribution per token. This project explores **Budget-Aware Sparse KD**, answering whether we can effectively distill models by explicitly restricting the teacher's outputs to a fixed per-token budget. It introduces a stable PyTorch pipeline comparing standard truncation (Top-K) and multinomial proxy (Sampling-based) methods against unconstrained models. 

The core research question investigated here is: 
> *Under a fixed per-token teacher-information budget, how should retained teacher information be represented so that the student preserves as much useful signal as possible?*

Currently, the codebase provides a robust framework for testing three foundational baselines:
1. **Full KD** (Unconstrained - Our Ceiling Baseline)
2. **Top-K KD** (Truncated Head Supervision)
3. **Sampling-based KD** (Using Multinomial Proxy Counts)

---

## 📂 Project Structure

The codebase is organized into modular packages and executable scripts:

```text
sparse_kd/
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   └── eval_utils.py
├── scripts/
│   ├── cache_teacher.py
│   ├── train_full_kd.py
│   ├── train_topk_kd.py
│   ├── train_sampling_kd.py
│   └── evaluate.py
├── example.sh
├── run_real_experiments.sh   <-- Unified Scale Experiments
├── project.md
└── requirements.txt
```

---

## 🚀 Running Experiments

Ensure you have your environment set up and the required packages installed:
```bash
pip install -r requirements.txt
```

### 1. Unified Real Experiments
The easiest way to execute the full matrix of experiments on a real scale (e.g. 200,000 samples of `wikitext-103-raw-v1`) is by using the master script:

```bash
./run_real_experiments.sh
```
This script handles everything autonomously:
1. Assesses the Raw student baseline.
2. Executes the Full KD (Dense) parameter sweep.
3. Automatically computes mathematical bounding and caches the Pythia-1.4B teacher offline.
4. Iterates across your defined sparsity parameter bounds ($K \in \{4, 8, 16\}$).
5. Trains your Pythia-160M student on Top-K and Sampling-based distributions.
6. Extensively evaluates NLL and Perplexity and exports directly to `experiment_log.csv`.

### 2. Manual Offline Teacher Caching
Because the teacher (e.g., Pythia 1.4B) is computationally heavy, you can evaluate the teacher outputs offline for modularity:

```bash
python scripts/cache_teacher.py \
    --mode topk \
    --dataset wikitext-103-raw-v1 \
    --seq_len 256 \
    --batch_size 16 \
    --num_train_samples 200000 \
    --cache_dir teacher_cache \
    --topk_k 16
```

### 3. Training the Student Manually
You can distill your chosen student (Pythia 160M) off of any generated caches or sequentially against the raw dense teacher.

```bash
python scripts/train_topk_kd.py \
    --batch_size 16 \
    --num_epochs 1 \
    --dataset wikitext-103-raw-v1 \
    --cache_dir teacher_cache \
    --output_dir output/topk_kd \
    --k 8
```

### 4. Evaluation
Eval provides the standardized `Evaluation Summary` breaking down Language Modeling Performance vs. Budget Efficiency.
```bash
python scripts/evaluate.py \
    --model_path output/topk_kd \
    --method topk \
    --k 8 \
    --train_dataset wikitext-103-raw-v1 \
    --val_dataset wikitext-103-raw-v1 \
    --cache_path teacher_cache/topk_train.pt
```

*(For a tiny toy demonstration of the flow, run `./example.sh`)*

---

## 📊 Evaluation & Metrics

Our framework evaluates progress against 3 core boundaries:

### Language Modeling Performance
We evaluate standard language modeling metrics, including negative log-likelihood (NLL):

$$
\mathcal{L}_{CE} = -\mathbb{E}_{(x,y)} \log p_S(y \mid x)
$$

and perplexity:

$$
\mathrm{PPL} = \exp(\mathcal{L}_{CE})
$$

### Budget Efficiency
Constraints are measured strictly by numerical `Scalars / Token` (e.g. A K=8 Top-K execution stores 16 scalars per token: 8 probabilities + 8 indices).

### Qualitative Analysis
Evaluating heuristic tail representations and tracking model behavior against high-uncertainty and ambiguously truncated tokens.
