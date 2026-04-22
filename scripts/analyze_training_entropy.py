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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--num_train_samples", type=int, default=2000, help="Number of training samples to analyze")
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
        "low": 0, "mid": 0, "high": 0, "all": 0
    }

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Analyzing Training Entropy"):
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
            
            low = (entropy < 1.5).sum().item()
            mid = ((entropy >= 1.5) & (entropy < 3.5)).sum().item()
            high = (entropy >= 3.5).sum().item()
            
            results["low"] += low
            results["mid"] += mid
            results["high"] += high
            results["all"] += low + mid + high

    print("\n--- Training Set Entropy Distribution ---")
    data = []
    total = results["all"]
    if total > 0:
        data.append({"Bucket": "Low (H < 1.5) [K=4]", "Percentage (%)": f"{results['low']/total*100:.1f}%"})
        data.append({"Bucket": "Mid (1.5 <= H < 3.5) [K=8]", "Percentage (%)": f"{results['mid']/total*100:.1f}%"})
        data.append({"Bucket": "High (H >= 3.5) [K=16]", "Percentage (%)": f"{results['high']/total*100:.1f}%"})
        data.append({"Bucket": "Total", "Percentage (%)": "100.0%"})
        df = pd.DataFrame(data)
        print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
