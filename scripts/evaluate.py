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
    _, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size, num_train_samples=2, num_val_samples=1000)

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

if __name__ == "__main__":
    main()
