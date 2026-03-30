"""
Analyze the teacher's (Pythia-1.4B) probability distribution on WikiText-103.

Reports:
  - Mean / median / p90 of tail probability mass beyond top-K
  - Mean / median / p90 of teacher entropy
  - % of tokens where the gold label falls outside top-K
  - Statistics broken down by temperature (T=1.0, 1.5, 2.0)

This tells us empirically how much information Top-K is throwing away.
"""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import sys, os
sys.path.insert(0, os.getcwd())

from src.models import load_teacher
from src.data import get_dataloaders


def analyze(teacher, dataloader, k_values=(4, 8, 16), temperatures=(1.0,)):
    teacher.eval()
    device = next(teacher.parameters()).device

    # Accumulators
    all_entropy = []
    tail_mass = {k: [] for k in k_values}
    gold_outside_topk = {k: 0 for k in k_values}
    total_valid = 0

    for batch in tqdm(dataloader, desc="Analyzing teacher"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != -100)

        for T in temperatures:
            probs = F.softmax(shift_logits / T, dim=-1)

            # Entropy: -sum p log p
            log_probs = F.log_softmax(shift_logits / T, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, T-1]
            all_entropy.append(entropy[valid_mask].cpu().numpy())

            for k in k_values:
                topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
                topk_mass = topk_probs.sum(dim=-1)  # [B, T-1]
                tail = 1.0 - topk_mass
                tail_mass[k].append(tail[valid_mask].cpu().numpy())

                # Is gold token outside top-K?
                expanded_labels = shift_labels.unsqueeze(-1)
                in_topk = (topk_ids == expanded_labels).any(dim=-1)
                gold_outside_topk[k] += ((~in_topk) & valid_mask).sum().item()

        total_valid += valid_mask.sum().item()

    # Aggregate
    all_entropy = np.concatenate(all_entropy)
    print(f"\n{'='*60}")
    print(f"Teacher Distribution Analysis (T={temperatures})")
    print(f"Total valid tokens analyzed: {total_valid:,}")
    print(f"{'='*60}")

    print(f"\n--- Entropy ---")
    print(f"  Mean:   {all_entropy.mean():.4f}")
    print(f"  Median: {np.median(all_entropy):.4f}")
    print(f"  P90:    {np.percentile(all_entropy, 90):.4f}")
    print(f"  P99:    {np.percentile(all_entropy, 99):.4f}")
    print(f"  Max:    {all_entropy.max():.4f}")

    for k in k_values:
        tails = np.concatenate(tail_mass[k])
        pct_outside = 100.0 * gold_outside_topk[k] / total_valid
        print(f"\n--- Top-{k} Tail Mass ---")
        print(f"  Mean tail mass:   {tails.mean():.6f}  ({tails.mean()*100:.4f}%)")
        print(f"  Median tail mass: {np.median(tails):.6f}  ({np.median(tails)*100:.4f}%)")
        print(f"  P90 tail mass:    {np.percentile(tails, 90):.6f}  ({np.percentile(tails, 90)*100:.4f}%)")
        print(f"  P99 tail mass:    {np.percentile(tails, 99):.6f}  ({np.percentile(tails, 99)*100:.4f}%)")
        print(f"  Max tail mass:    {tails.max():.6f}  ({tails.max()*100:.4f}%)")
        print(f"  Gold outside top-{k}: {gold_outside_topk[k]:,} / {total_valid:,} = {pct_outside:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1")
    args = parser.parse_args()

    teacher, tokenizer = load_teacher()

    train_loader, _ = get_dataloaders(
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_samples=args.num_samples,
        train_dataset_config=args.dataset,
        val_dataset_config=args.dataset,
    )

    analyze(teacher, train_loader, k_values=(4, 8, 16, 32), temperatures=(1.0,))


if __name__ == "__main__":
    main()
