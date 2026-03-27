import argparse
import torch
from torch.optim import AdamW
from src.models import load_student
from src.data import get_cached_dataloaders
from src.losses import compute_cached_topk_kd_loss
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="teacher_cache")
    parser.add_argument("--output_dir", type=str, default="output/topk_kd")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    print("Loading student model...")
    student, tokenizer = load_student()
    
    print(f"Loading cached data from {args.cache_dir}...")
    train_loader, val_loader = get_cached_dataloaders(cache_fmt="topk", cache_dir=args.cache_dir, batch_size=args.batch_size)
    
    optimizer = AdamW(student.parameters(), lr=args.lr)
    
    num_epochs = args.num_epochs
    device = student.device
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Starting Top-K KD training on device: {device} | Detected GPU: {gpu_name}")
    student.train()
    
    step = 0
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            topk_probs = batch["topk_probs"].to(input_device)
            topk_ids = batch["topk_ids"].to(input_device)
            
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits
            
            loss, ce_loss, kl_loss = compute_cached_topk_kd_loss(student_logits, topk_probs, topk_ids, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 2 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            
            step += 1
            if step >= 10:
                print("Completed 10 steps for verification.")
                break

    print(f"Saving model to {args.output_dir}")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
