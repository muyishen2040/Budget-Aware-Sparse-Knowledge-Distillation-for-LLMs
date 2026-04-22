import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataloaders
import argparse
import pandas as pd
from tqdm import tqdm

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

def compute_kl_divergence(t_logits, s_logits):
    t_probs = F.softmax(t_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    return torch.sum(t_probs * (t_log_probs - s_log_probs), dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--adaptive_path", type=str, default="output/real_wikitext_adaptive_topk")
    parser.add_argument("--full_path", type=str, default="output/real_wikitext_full_kd")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--num_val_samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, torch_dtype=torch.float16, device_map="auto").eval()
    
    adaptive_student = AutoModelForCausalLM.from_pretrained(args.adaptive_path, torch_dtype=torch.float16, device_map="auto").eval()
    full_student = AutoModelForCausalLM.from_pretrained(args.full_path, torch_dtype=torch.float16, device_map="auto").eval()

    print("Loading Data...")
    _, val_loader = get_dataloaders(tokenizer, seq_len=256, batch_size=8, num_train_samples=10, num_val_samples=args.num_val_samples, train_dataset_name=args.dataset, val_dataset_name=args.dataset)

    results = {
        "low": {"count": 0, "t_nll": 0.0, "a_nll": 0.0, "f_nll": 0.0, "a_kl": 0.0, "f_kl": 0.0},
        "mid": {"count": 0, "t_nll": 0.0, "a_nll": 0.0, "f_nll": 0.0, "a_kl": 0.0, "f_kl": 0.0},
        "high": {"count": 0, "t_nll": 0.0, "a_nll": 0.0, "f_nll": 0.0, "a_kl": 0.0, "f_kl": 0.0},
        "all":  {"count": 0, "t_nll": 0.0, "a_nll": 0.0, "f_nll": 0.0, "a_kl": 0.0, "f_kl": 0.0}
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            a_logits = adaptive_student(input_ids=input_ids, attention_mask=attention_mask).logits
            f_logits = full_student(input_ids=input_ids, attention_mask=attention_mask).logits

            # Shift
            t_shift_logits = t_logits[..., :-1, :].contiguous()
            a_shift_logits = a_logits[..., :-1, :].contiguous()
            f_shift_logits = f_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            valid_mask = shift_labels != -100

            if valid_mask.sum() == 0:
                continue

            t_shift_logits = t_shift_logits[valid_mask].float()
            a_shift_logits = a_shift_logits[valid_mask].float()
            f_shift_logits = f_shift_logits[valid_mask].float()
            shift_labels = shift_labels[valid_mask]

            # Metrics
            t_entropy = compute_entropy(t_shift_logits)
            
            t_nll = F.cross_entropy(t_shift_logits, shift_labels, reduction='none')
            a_nll = F.cross_entropy(a_shift_logits, shift_labels, reduction='none')
            f_nll = F.cross_entropy(f_shift_logits, shift_labels, reduction='none')

            a_kl = compute_kl_divergence(t_shift_logits, a_shift_logits)
            f_kl = compute_kl_divergence(t_shift_logits, f_shift_logits)

            # Buckets
            masks = {
                "low": t_entropy < 1.5,
                "mid": (t_entropy >= 1.5) & (t_entropy < 3.5),
                "high": t_entropy >= 3.5,
                "all": torch.ones_like(t_entropy, dtype=torch.bool)
            }

            for k, mask in masks.items():
                if mask.sum() > 0:
                    results[k]["count"] += mask.sum().item()
                    results[k]["t_nll"] += t_nll[mask].sum().item()
                    results[k]["a_nll"] += a_nll[mask].sum().item()
                    results[k]["f_nll"] += f_nll[mask].sum().item()
                    results[k]["a_kl"] += a_kl[mask].sum().item()
                    results[k]["f_kl"] += f_kl[mask].sum().item()

    # Format Results
    print("\n--- Entropy Distribution Analysis ---")
    data = []
    total_tokens = results["all"]["count"]
    for bucket in ["low", "mid", "high", "all"]:
        r = results[bucket]
        cnt = r["count"]
        if cnt == 0: continue
        pct = (cnt / total_tokens) * 100
        data.append({
            "Bucket": bucket.capitalize(),
            "Tokens (%)": f"{pct:.1f}%",
            "Teacher NLL": f"{r['t_nll']/cnt:.3f}",
            "Adaptive NLL": f"{r['a_nll']/cnt:.3f}",
            "Full KD NLL": f"{r['f_nll']/cnt:.3f}",
            "Adaptive KL": f"{r['a_kl']/cnt:.3f}",
            "Full KD KL": f"{r['f_kl']/cnt:.3f}",
        })
    
    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
