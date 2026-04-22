import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataloaders
import argparse
import pandas as pd
from tqdm import tqdm

def compute_head_tail_kl(t_logits, s_logits, k=16):
    """
    Splits the distribution into Top-K (Head) and the rest (Tail).
    Computes KL divergence strictly over these subset distributions (normalized).
    """
    t_probs = F.softmax(t_logits, dim=-1)
    s_probs = F.softmax(s_logits, dim=-1)

    t_top_probs, t_top_idx = torch.topk(t_probs, k, dim=-1)
    
    # Create mask for Top-K
    mask = torch.zeros_like(t_probs, dtype=torch.bool).scatter_(-1, t_top_idx, True)
    
    # Head Probs (Normalized)
    t_head = t_probs * mask
    t_head_norm = t_head / (t_head.sum(dim=-1, keepdim=True) + 1e-10)
    
    s_head = s_probs * mask
    s_head_norm = s_head / (s_head.sum(dim=-1, keepdim=True) + 1e-10)

    head_kl = torch.sum(t_head_norm * (torch.log(t_head_norm + 1e-10) - torch.log(s_head_norm + 1e-10)), dim=-1)

    # Tail Probs (Normalized)
    t_tail = t_probs * (~mask)
    t_tail_norm = t_tail / (t_tail.sum(dim=-1, keepdim=True) + 1e-10)
    
    s_tail = s_probs * (~mask)
    s_tail_norm = s_tail / (s_tail.sum(dim=-1, keepdim=True) + 1e-10)

    tail_kl = torch.sum(t_tail_norm * (torch.log(t_tail_norm + 1e-10) - torch.log(s_tail_norm + 1e-10)), dim=-1)

    return head_kl, tail_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--adaptive_path", type=str, default="output/real_wikitext_adaptive_topk")
    parser.add_argument("--full_path", type=str, default="output/real_wikitext_full_kd")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--num_val_samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=16, help="K value to separate Head from Tail")
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
        "a_head_kl": 0.0, "a_tail_kl": 0.0,
        "f_head_kl": 0.0, "f_tail_kl": 0.0,
        "count": 0
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            a_logits = adaptive_student(input_ids=input_ids, attention_mask=attention_mask).logits
            f_logits = full_student(input_ids=input_ids, attention_mask=attention_mask).logits

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

            a_head_kl, a_tail_kl = compute_head_tail_kl(t_shift_logits, a_shift_logits, k=args.k)
            f_head_kl, f_tail_kl = compute_head_tail_kl(t_shift_logits, f_shift_logits, k=args.k)

            results["count"] += valid_mask.sum().item()
            results["a_head_kl"] += a_head_kl.sum().item()
            results["a_tail_kl"] += a_tail_kl.sum().item()
            results["f_head_kl"] += f_head_kl.sum().item()
            results["f_tail_kl"] += f_tail_kl.sum().item()

    print("\n--- Tail Noise Analysis (Head K={}) ---".format(args.k))
    cnt = results["count"]
    data = [
        {"Model": "Adaptive Top-K", "Head KL (Top-16)": f"{results['a_head_kl']/cnt:.4f}", "Tail KL (Rest)": f"{results['a_tail_kl']/cnt:.4f}"},
        {"Model": "Full KD", "Head KL (Top-16)": f"{results['f_head_kl']/cnt:.4f}", "Tail KL (Rest)": f"{results['f_tail_kl']/cnt:.4f}"}
    ]
    
    df = pd.DataFrame(data)
    print(df.to_markdown(index=False))
    print("\nNOTE: Lower Head KL means better matching of the teacher's most likely tokens.")
    print("Lower Tail KL means mimicking the teacher's noise.")

if __name__ == "__main__":
    main()
