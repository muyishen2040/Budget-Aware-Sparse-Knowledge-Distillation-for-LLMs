"""
analyze_gold_coverage.py
========================
Run a model (student or teacher) forward pass on a dataset and measure
what percentage of gold (ground-truth next) tokens fall OUTSIDE the model's
Top-K predictions, sweeping over a list of budget values K.

This answers: "At budget K, how often does the model assign zero probability
mass to the correct next token?"

Usage
-----
    # Evaluate a distilled student checkpoint on wikitext val
    python scripts/analyze_gold_coverage.py \\
        --model_path  output/real_topk_k16 \\
        --dataset     wikitext \\
        --split       val \\
        --budgets     4 8 16 32 64

    # Evaluate the base teacher on github-code-python
    python scripts/analyze_gold_coverage.py \\
        --model_path  EleutherAI/pythia-1.4b \\
        --dataset     github-code-python \\
        --split       val \\
        --budgets     4 8 16 32 64 \\
        --num_samples 2000

Output
------
Prints a coverage table to stdout, e.g.:

    Model : output/real_topk_k16
    Dataset: wikitext  |  Split: val  |  Tokens: 256,000

    Budget K | Gold NOT in Top-K | Gold in Top-K
    ---------+-------------------+--------------
           4 |           62.41 % |      37.59 %
           8 |           48.07 % |      51.93 %
          16 |           32.15 % |      67.85 %

Optionally writes results to a CSV file (--out_csv).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure the src/ package is importable when running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_dataloaders


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, tokenizer_path: str | None):
    """
    Load a causal LM from a local directory (HuggingFace saved format) or
    an HuggingFace model hub ID.  Returns (model, tokenizer).

    Tokenizer is loaded from `tokenizer_path` if given, otherwise from
    `model_path`.  Falls back to the teacher tokenizer
    (EleutherAI/pythia-1.4b) if needed (distilled students may not bundle one).
    """
    tok_source = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
    except Exception:
        # Some student checkpoints don't ship a tokenizer — use teacher's
        fallback = "EleutherAI/pythia-1.4b"
        print(f"[load] Tokenizer not found at '{tok_source}', "
              f"falling back to {fallback}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(fallback)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Forward pass & coverage
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_miss_rates(
    model,
    dataloader,
    budgets: list[int],
    max_k: int,
    ignore_index: int = -100,
    device: str = "cuda",
) -> tuple[dict[int, float], int]:
    """
    Run the model forward pass over `dataloader` and compute, for each k in
    `budgets`, the fraction of valid token positions where the gold label is
    NOT in the model's top-k predictions.

    Returns (miss_rate_dict, total_valid_tokens).
    """
    # Accumulators per budget
    miss_counts  = {k: 0 for k in budgets}
    total_valid  = 0

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        input_ids      = batch["input_ids"].to(device)       # [B, T]
        attention_mask = batch["attention_mask"].to(device)  # [B, T]
        labels         = batch["labels"].to(device)          # [B, T]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # [B, T, V]

        # Teacher/student predictions are for the NEXT token; labels are
        # already shifted: labels[b, t] is the target for position t.
        # We use logits[:, :-1, :] vs labels[:, 1:] to align them.
        logits_aligned = logits[:, :-1, :].float()   # [B, T-1, V]
        labels_aligned = labels[:, 1:]               # [B, T-1]

        # Top-max_k indices at every position  [B, T-1, max_k]
        topk_ids = torch.topk(logits_aligned, k=max_k, dim=-1).indices

        # Valid mask (exclude padding / -100)
        valid = labels_aligned != ignore_index       # [B, T-1]
        n_valid = valid.sum().item()
        total_valid += n_valid

        gold = labels_aligned.unsqueeze(-1)          # [B, T-1, 1]

        for k in budgets:
            k_eff = min(k, max_k)
            found = (topk_ids[:, :, :k_eff] == gold).any(dim=-1)  # [B, T-1]
            miss_counts[k] += (valid & ~found).sum().item()

    miss_rates = {
        k: (miss_counts[k] / total_valid * 100.0) if total_valid > 0 else 0.0
        for k in budgets
    }
    return miss_rates, total_valid


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(model_path, dataset, split, n_tokens, budgets, miss_rates):
    print()
    print(f"  Model  : {model_path}")
    print(f"  Dataset: {dataset}  |  Split: {split}  |  Tokens: {n_tokens:,}")
    print()
    print(f"  {'Budget K':>8s} | {'Gold NOT in Top-K':>18s} | {'Gold in Top-K':>14s}")
    print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*14}")
    for k in budgets:
        miss = miss_rates[k]
        hit  = 100.0 - miss
        print(f"  {k:>8d} | {miss:>17.2f} % | {hit:>13.2f} %")
    print()


def write_csv(out_path, model_path, dataset, split, budgets, miss_rates):
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "split", "budget_k",
                         "miss_pct", "hit_pct"])
        for k in budgets:
            miss = miss_rates[k]
            writer.writerow([model_path, dataset, split, k,
                             f"{miss:.4f}", f"{100-miss:.4f}"])
    print(f"  [csv] Results written to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure gold-label Top-K coverage for a student/teacher model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to a HuggingFace model dir (e.g. output/real_topk_k16) "
             "or hub model ID (e.g. EleutherAI/pythia-160m).",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Optional separate tokenizer location. Defaults to --model_path.",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=["wikitext", "wikitext-103-raw-v1",
                 "github-code", "github-code-python", "pubmed"],
        help="Dataset key (default: wikitext).",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val"],
        help="Which split to evaluate (default: val).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of text samples to load for evaluation (default: 1000).",
    )
    parser.add_argument(
        "--seq_len", type=int, default=256,
        help="Sequence length for tokenisation (default: 256).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for forward passes (default: 8).",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64],
        help="Top-K budgets to evaluate (default: 4 8 16 32 64).",
    )
    parser.add_argument(
        "--ignore_index", type=int, default=-100,
        help="Label value to ignore (padding); default: -100.",
    )
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="Optional path to write results as CSV.",
    )
    args = parser.parse_args()

    budgets = sorted(set(args.budgets))
    max_k   = max(budgets)

    print(f"\n=== Gold-Label Coverage Analysis ===")
    print(f"  model_path  : {args.model_path}")
    print(f"  dataset     : {args.dataset}")
    print(f"  split       : {args.split}")
    print(f"  num_samples : {args.num_samples}")
    print(f"  budgets     : {budgets}")

    # Load model + tokenizer
    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    device = next(model.parameters()).device
    print(f"  device      : {device}")

    # Build dataloader for the requested split
    # get_dataloaders always returns (train_loader, val_loader); pick accordingly.
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_samples=args.num_samples if args.split == "train" else 512,
        num_val_samples=args.num_samples if args.split == "val"   else 512,
        train_dataset_name=args.dataset,
        val_dataset_name=args.dataset,
    )
    dataloader = train_loader if args.split == "train" else val_loader

    # Run evaluation
    miss_rates, n_tokens = compute_miss_rates(
        model, dataloader, budgets, max_k,
        ignore_index=args.ignore_index,
        device=str(device),
    )

    print_table(args.model_path, args.dataset, args.split,
                n_tokens, budgets, miss_rates)

    if args.out_csv:
        write_csv(args.out_csv, args.model_path, args.dataset,
                  args.split, budgets, miss_rates)


if __name__ == "__main__":
    main()
