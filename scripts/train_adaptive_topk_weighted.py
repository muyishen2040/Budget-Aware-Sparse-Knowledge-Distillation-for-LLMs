import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.models import load_student
from src.data import get_cached_dataloaders
from src.losses import compute_cached_adaptive_topk_weighted_kd_loss
from src.eval_utils import compute_lm_metrics
import time
from tqdm import tqdm

def evaluate(student, val_loader, device):
    student.eval()
    total_ce_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)

            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
            valid_tokens = (shift_labels != -100).sum()
            total_ce_loss += ce_loss.item()
            total_tokens += valid_tokens.item()
            
    student.train()
    return total_ce_loss / total_tokens if total_tokens > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="teacher_cache")
    parser.add_argument("--output_dir", type=str, default="output/adaptive_topk_weighted")
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        help="Dataset key (metadata only, data is loaded from cache)"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no_normalize_weights", dest="normalize_weights",
        action="store_false", default=True,
        help="Disable head-mass weight normalization (on by default).")
    parser.add_argument("--min_weight", type=float, default=0.2,
        help="Floor for per-token weight. E.g., 0.5 = at least 50%% KD signal.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading student model...")
    student, tokenizer = load_student()
    
    print(f"Loading cached data from {args.cache_dir}...")
    train_loader, val_loader = get_cached_dataloaders(cache_fmt="adaptive_topk", cache_dir=args.cache_dir, batch_size=args.batch_size)
    
    optimizer = AdamW(student.parameters(), lr=args.lr)
    
    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    device = student.device
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Starting Head-Mass Weighted Adaptive Top-K KD training on device: {device} | GPU: {gpu_name}")
    student.train()
    
    start_time = time.time()
    step = 0
    final_loss = 0.0
    total_valid_k_sum = 0.0
    total_valid_k_count = 0
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            adaptive_topk_probs = batch["adaptive_topk_probs"].to(input_device)
            adaptive_topk_ids = batch["adaptive_topk_ids"].to(input_device)
            adaptive_topk_valid_k = batch["adaptive_topk_valid_k"].to(input_device)
            
            # Track average budget
            total_valid_k_sum += adaptive_topk_valid_k.float().sum().item()
            total_valid_k_count += adaptive_topk_valid_k.numel()
            
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits
            
            loss, ce_loss, kl_loss = compute_cached_adaptive_topk_weighted_kd_loss(
                student_logits, adaptive_topk_probs, adaptive_topk_ids,
                adaptive_topk_valid_k, labels,
                temperature=args.temperature, alpha=args.alpha,
                normalize_weights=args.normalize_weights,
                min_weight=args.min_weight
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                avg_k = total_valid_k_sum / max(total_valid_k_count, 1)
                print(f"Step {step} | Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | KL: {kl_loss.item():.4f} | Avg K: {avg_k:.2f}", flush=True)
                
            if step > 0 and step % 500 == 0:
                print(f"Running validation at step {step}...", flush=True)
                val_ce_loss = evaluate(student, val_loader, device)
                metrics = compute_lm_metrics(val_ce_loss)
                print(f"Validation | NLL: {metrics['nll']:.4f} | PPL: {metrics['ppl']:.4f}", flush=True)
            
            final_loss = loss.item()
            step += 1

        # Run one final evaluation at the end of epoch
        print(f"Epoch {epoch+1} complete. Running final validation...", flush=True)
        val_ce_loss = evaluate(student, val_loader, device)
        metrics = compute_lm_metrics(val_ce_loss)
        print(f"Final Epoch Validation | NLL: {metrics['nll']:.4f} | PPL: {metrics['ppl']:.4f}", flush=True)

    avg_k_final = total_valid_k_sum / max(total_valid_k_count, 1)
    run_time = time.time() - start_time
    print(f"Saving model to {args.output_dir}")
    print(f"METRICS_TRAIN_LOSS={final_loss:.4f}")
    print(f"METRICS_RUN_TIME={run_time:.2f}")
    print(f"METRICS_AVG_K={avg_k_final:.4f}")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
