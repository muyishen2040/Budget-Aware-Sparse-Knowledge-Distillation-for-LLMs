import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataloaders
from src.eval_utils import compute_lm_metrics, calculate_budget, print_evaluation_summary, extract_qualitative_masks
from tqdm import tqdm
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained student model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--method", type=str, required=True, choices=["full", "topk", "sampling"], help="Method type to calculate budget")
    parser.add_argument("--k", type=int, default=8, help="Budget param for TopK")
    parser.add_argument("--num_draws", type=int, default=50, help="Budget param for Sampling")
    parser.add_argument("--cache_path", type=str, default=None, help="Path to cache file to measure size")
    parser.add_argument("--log_file", type=str, default="experiment_log.csv")
    parser.add_argument("--num_train_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--train_loss", type=float, default=0.0)
    parser.add_argument("--run_time_seconds", type=float, default=0.0)
    parser.add_argument("--train_dataset", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--val_dataset", type=str, default="wikitext-103-raw-v1")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    student = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    student.eval()

    print("Loading validation data...")
    # Use 1k val samples for quick eval
    _, val_loader = get_dataloaders(
        tokenizer, seq_len=args.seq_len, batch_size=args.batch_size, 
        num_train_samples=1000, num_val_samples=1000,
        val_dataset_config=args.val_dataset
    )

    device = student.device
    total_ce_loss = 0.0
    total_tokens = 0

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate Cross Entropy Loss
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            
            # Use sum so we can calculate exact average correctly across batches of different sizes
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
            
            valid_tokens = (shift_labels != -100).sum()
            total_ce_loss += ce_loss.item()
            total_tokens += valid_tokens.item()
            
    avg_ce_loss = total_ce_loss / total_tokens
    
    # Calculate budget info
    budget_kwargs = {"k": args.k, "num_draws": args.num_draws}
    
    # Print the evaluation summary
    print_evaluation_summary(args.method, avg_ce_loss, args.cache_path, **budget_kwargs)

    # Append to CSV
    metrics = compute_lm_metrics(avg_ce_loss)
    import os
    budget = args.k * 2 if args.method == "topk" else (args.num_draws * 2 if args.method == "sampling" else 50277)
    size_mb = os.path.getsize(args.cache_path) / (1024 * 1024) if args.cache_path and os.path.exists(args.cache_path) else 0.0

    import csv
    file_exists = os.path.isfile(args.log_file)
    with open(args.log_file, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "method", "budget", "train_dataset", "val_dataset", "num_train_samples", "epochs", "batch_size", "lr", "train_loss", "val_nll", "val_ppl", "run_time_seconds", "storage_mb"])
        
        writer.writerow([
            os.path.basename(args.model_path.strip("/")),
            args.method,
            budget,
            args.train_dataset,
            args.val_dataset,
            args.num_train_samples,
            args.epochs,
            args.train_batch_size,
            args.lr,
            args.train_loss,
            metrics["nll"],
            metrics["ppl"],
            args.run_time_seconds,
            size_mb
        ])

if __name__ == "__main__":
    main()
