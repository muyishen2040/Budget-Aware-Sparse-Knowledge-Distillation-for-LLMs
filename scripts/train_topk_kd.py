import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.models import load_student
from src.data import get_cached_dataloaders
from src.losses import compute_cached_topk_kd_loss
from src.eval_utils import compute_lm_metrics
import time
from tqdm import tqdm
from AutoEncoder.autoencoder import KDAautoEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("LOADING AE WEIGHTS... (THIS MAY TAKE A MOMENT)")
ae_weights_dir = '/content/drive/MyDrive/ANLP_Sparse_KD/ae_trained.pth'
ae_weights = torch.load(ae_weights_dir, map_location=DEVICE)
ae_model = KDAautoEncoder().to(DEVICE)
ae_model.load_state_dict(ae_weights)
ae_model = ae_model.to(torch.float32)  # ensure model is in float32 for consistent behavior during encoding
for name, param in ae_model.named_parameters():
    assert param.dtype == torch.float32, f"{name} is not float32"
ae_model.eval()
print("AE MODEL LOADED AND READY.")
print(ae_model)


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
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="teacher_cache")
    parser.add_argument("--output_dir", type=str, default="output/topk_kd")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        help="Dataset key (metadata only, data is loaded from cache): 'wikitext', 'github-code', 'github-code-python', 'pubmed'"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading student model...")
    student, tokenizer = load_student()
    
    print(f"Loading cached data from {args.cache_dir}...")
    train_loader, val_loader = get_cached_dataloaders(cache_fmt="topk", cache_dir=args.cache_dir, batch_size=args.batch_size)
    
    optimizer = AdamW(student.parameters(), lr=args.lr)
    
    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    device = student.device
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Starting Top-K KD training on device: {device} | Detected GPU: {gpu_name}")
    student.train()
    
    start_time = time.time()
    step = 0
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            
            # Subset the cached top-k probabilities to exactly args.k
            k = args.k
            topk_probs = batch["topk_probs"][..., :k].to(input_device)
            topk_ids = batch["topk_ids"][..., :k].to(input_device)
            compressedk_probs = batch["compressedk_probs"].to(input_device)            
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits
            
            loss, ce_loss, kl_loss = compute_cached_topk_kd_loss(compressedk_probs, ae_model, student_logits, topk_probs, topk_ids, labels, temperature=args.temperature, alpha=args.alpha)  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | KL: {kl_loss.item():.4f}", flush=True)
                
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

    run_time = time.time() - start_time
    print(f"Saving model to {args.output_dir}")
    print(f"METRICS_TRAIN_LOSS={final_loss:.4f}")
    print(f"METRICS_RUN_TIME={run_time:.2f}")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
