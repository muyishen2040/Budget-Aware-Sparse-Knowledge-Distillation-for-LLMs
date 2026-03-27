import argparse
import torch
from torch.optim import AdamW
from src.models import load_models
from src.data import get_dataloaders
from src.losses import compute_full_kd_loss
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output/full_kd")
    args = parser.parse_args()

    print("Loading models...")
    teacher, student, tokenizer = load_models()
    
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size, num_train_samples=args.num_train_samples)
    
    optimizer = AdamW(student.parameters(), lr=args.lr)
    
    num_epochs = args.num_epochs
    device = student.device
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Starting training on device: {device} | Detected GPU: {gpu_name}")
    student.train()
    teacher.eval()
    
    step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            input_device = student.model.embed_tokens.weight.device if hasattr(student, 'model') else device
            
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
                
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits
            
            teacher_logits = teacher_logits.to(student_logits.device)
            labels = labels.to(student_logits.device)
            
            loss, ce_loss, kl_loss = compute_full_kd_loss(student_logits, teacher_logits, labels)
            
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
