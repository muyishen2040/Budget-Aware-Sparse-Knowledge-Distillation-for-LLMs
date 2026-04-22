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

def analyze_tail(t_logits, k=16, threshold=1e-5):
    probs = F.softmax(t_logits, dim=-1) # [N, V]
    
    top_probs, top_idx = torch.topk(probs, k, dim=-1)
    head_mass = top_probs.sum(dim=-1)
    tail_mass = 1.0 - head_mass
    
    top_k_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, top_idx, True)
    tail_probs = probs.masked_fill(top_k_mask, 0.0)
    
    active_tail_tokens = (tail_probs > threshold).sum(dim=-1)
    
    return tail_mass, active_tail_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--num_train_samples", type=int, default=2000, help="Number of training samples to analyze")
    parser.add_argument("--k", type=int, default=16, help="K value defining the Head")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, torch_dtype=torch.float16, device_map="auto").eval()

    print("Loading Training Data...")
    train_loader, _ = get_dataloaders(
        tokenizer, seq_len=256, batch_size=8, 
        num_train_samples=args.num_train_samples, num_val_samples=10, 
        train_dataset_name=args.dataset, val_dataset_name=args.dataset
    )

    results = {
        "low": {"count": 0, "tail_mass": 0.0, "active_tokens": 0},
        "mid": {"count": 0, "tail_mass": 0.0, "active_tokens": 0},
        "high": {"count": 0, "tail_mass": 0.0, "active_tokens": 0},
        "all": {"count": 0, "tail_mass": 0.0, "active_tokens": 0}
    }

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Analyzing Teacher Tail Noise"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            t_shift_logits = t_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            valid_mask = shift_labels != -100
            
            if valid_mask.sum() == 0:
                continue

            t_shift_logits = t_shift_logits[valid_mask].float()
            
            entropy = compute_entropy(t_shift_logits)
            tail_mass, active_tail_tokens = analyze_tail(t_shift_logits, k=args.k)
            
            masks = {
                "low": entropy < 1.5,
                "mid": (entropy >= 1.5) & (entropy < 3.5),
                "high": entropy >= 3.5,
                "all": torch.ones_like(entropy, dtype=torch.bool)
            }
            
            for k, mask in masks.items():
                if mask.sum() > 0:
                    results[k]["count"] += mask.sum().item()
                    results[k]["tail_mass"] += tail_mass[mask].sum().item()
                    results[k]["active_tokens"] += active_tail_tokens[mask].sum().item()

    print(f"\n--- Teacher Tail Noise Analysis (Head K={args.k}) ---")
    data = []
    for bucket in ["low", "mid", "high", "all"]:
        r = results[bucket]
        cnt = r["count"]
        if cnt == 0: continue
        data.append({
            "Entropy Bucket": bucket.capitalize(),
            "Avg Tail Mass (%)": f"{(r['tail_mass']/cnt)*100:.2f}%",
            "Avg Noisy Tail Tokens (>1e-5)": f"{r['active_tokens']/cnt:.1f}"
        })
    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))
    print("\nNOTE: Tail Mass % is the probability mass outside Top-16.")
    print("Noisy Tail Tokens demonstrates how many tiny label probabilities Full KD forces the student to fit.")

if __name__ == "__main__":
    main()
