# Budget-Aware Sparse Knowledge Distillation for LLMs

Large language models typically require massive computational and storage overhead during Knowledge Distillation (KD) since they compute the full vocabulary distribution per token. This project explores **Budget-Aware Sparse KD**, answering whether we can effectively distill models by explicitly restricting the teacher's outputs to a fixed per-token budget. It introduces a stable PyTorch pipeline comparing standard truncation (Top-K) and multinomial proxy (Sampling-based) methods against unconstrained models. 

The core research question investigated here is: 
> *Under a fixed per-token teacher-information budget, how should retained teacher information be represented so that the student preserves as much useful signal as possible?*

Currently, the codebase provides a robust framework for testing three foundational baselines:
1. **Full KD** (Unconstrained)
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
├── project.md
└── requirements.txt
```

---

## 🚀 Getting Started

Ensure you have your environment set up and the required packages installed:
```bash
pip install -r requirements.txt
```

### 1. Offline Teacher Caching
Because the teacher (e.g., Pythia 1.4B) is heavy, we evaluate the validation distribution outputs offline into `.pt` serialized datasets.

```bash
python scripts/cache_teacher.py \
    --mode topk \
    --seq_len 256 \
    --batch_size 4 \
    --num_train_samples 2000 \
    --cache_dir teacher_cache \
    --topk_k 8
```

### 2. Training the Student
You can distill your chosen student (e.g., Pythia 160M) off of any generated caches or sequentially against the raw dense teacher.

```bash
python scripts/train_topk_kd.py \
    --batch_size 4 \
    --num_epochs 1 \
    --cache_dir teacher_cache \
    --output_dir output/topk_kd \
    --k 8
```

### 3. Evaluation
Eval provides the standardized `Evaluation Summary` breaking down Language Modeling Performance vs. Budget Efficiency.
```bash
python scripts/evaluate.py \
    --model_path output/topk_kd \
    --method topk \
    --k 8 \
    --cache_path teacher_cache/topk_train.pt
```

*(For a complete demonstration of the flow, run `./example.sh` from the base directory!)*

---

## 📊 Evaluation & Metrics

Our framework evaluates progress against 3 core boundaries:

### Language Modeling Performance
We evaluate standard language modeling metrics, including negative log-likelihood (NLL):
$$ \mathcal{L}_{CE} = -\mathbb{E}_{(x,y)} \log p_S(y \mid x) $$
and perplexity:
$$ \mathrm{PPL} = \exp(\mathcal{L}_{CE}) $$

### Budget Efficiency
Constraints are measured strictly by numerical `Scalars / Token` (e.g. A K=8 Top-K execution stores 16 scalars per token: 8 values + 8 indices).

### Qualitative Analysis
High uncertainty distributions vs Ambiguous Tokens filtering.
