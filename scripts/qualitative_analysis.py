import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_dataloaders
from src.eval_utils import extract_qualitative_masks
import argparse
import json
import os

def decode_top_k(logits_at_pos, tokenizer, k=5):
    """Returns the top-K tokens and their probabilities for a specific logit vector."""
    probs = F.softmax(logits_at_pos, dim=-1)
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    
    results = []
    for p, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode([idx.item()])
        token_repr = repr(token_str) # Safely exposes leading/trailing spaces visually
        results.append((token_repr, p.item()))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--student_full_path", type=str, default="output/real_full_kd")
    parser.add_argument("--student_topk_path", type=str, default="output/real_topk_k8")
    parser.add_argument("--student_sampling_path", type=str, default="output/real_sampling_k8")
    parser.add_argument("--num_val_samples", type=int, default=50)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    
    print("Loading models (This requires significant memory)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, torch_dtype=torch.float16, device_map="auto")
    teacher.eval()
    
    # We load students if their directories exist. Otherwise, we'll skip them dynamically.
    students = {}
    for name, path in [("Full_KD", args.student_full_path), 
                       ("TopK_8", args.student_topk_path), 
                       ("Sampling_8", args.student_sampling_path)]:
        if os.path.exists(path):
            students[name] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
            students[name].eval()
        else:
            print(f"Warning: {path} not found. Skipping {name} in comparison.")
            
    print("Loading Data...")
    _, val_loader = get_dataloaders(tokenizer, seq_len=256, batch_size=4, num_train_samples=2, num_val_samples=args.num_val_samples)
    
    # 3 Chosen Qualitative Results to Extract
    # 1. Ambiguous Tokens
    # 2. Top-K Truncation Failures
    # 3. High-Uncertainty Contexts (Evaluating Overconfidence Collapse)
    
    results = {
        "ambiguous": [],
        "topk_failure": [],
        "high_uncertainty": []
    }
    
    MAX_EXAMPLES_PER_CAT = 5
    
    with torch.no_grad():
        for batch in val_loader:
            if all(len(v) >= MAX_EXAMPLES_PER_CAT for v in results.values()):
                break # We have enough examples
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 1. Teacher Forward Pass
            t_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            
            # 2. Student Forward Passes
            s_logits_dict = {}
            for name, st_model in students.items():
                s_out = st_model(input_ids=input_ids, attention_mask=attention_mask)
                s_logits_dict[name] = s_out.logits
            
            # 3. Compute Masks
            masks = extract_qualitative_masks(t_outputs.logits, labels, k=8)
            
            # Iterate through batch
            for b in range(input_ids.size(0)):
                for seq_idx in range(40, input_ids.size(1) - 1): # Start at 40 for more context
                    
                    # We look at the shifted predictions. 
                    # The prediction at input_ids[..., seq_idx] is trying to predict labels[..., seq_idx+1]
                    pred_idx = seq_idx
                    target_idx = seq_idx + 1
                    
                    if labels[b, target_idx] == -100:
                        continue
                    
                    extracted_category = None
                    if len(results["ambiguous"]) < MAX_EXAMPLES_PER_CAT and masks["ambiguous"][b, pred_idx]:
                        extracted_category = "ambiguous"
                    elif len(results["topk_failure"]) < MAX_EXAMPLES_PER_CAT and masks["topk_failure"][b, pred_idx]:
                        extracted_category = "topk_failure"
                    elif len(results["high_uncertainty"]) < MAX_EXAMPLES_PER_CAT and masks["high_uncertainty"][b, pred_idx]:
                        extracted_category = "high_uncertainty"
                        
                    if extracted_category:
                        # Extract the preceding context
                        context_ids = input_ids[b, max(0, pred_idx-40):pred_idx+1]
                        context_str = tokenizer.decode(context_ids)
                        gold_token = tokenizer.decode([labels[b, target_idx].item()])
                        
                        entry = {
                            "context": context_str,
                            "gold_token": gold_token,
                            "teacher": decode_top_k(t_outputs.logits[b, pred_idx], tokenizer)
                        }
                        
                        for name in s_logits_dict:
                            entry[name] = decode_top_k(s_logits_dict[name][b, pred_idx], tokenizer)
                            
                        results[extracted_category].append(entry)
                        
                        # Stop scanning this specific sequence so we get diverse, non-contiguous examples
                        break

    # Output to markdown file
    with open("qualitative_report.md", "w") as f:
        f.write("# Qualitative Distillation Analysis\n\n")
        
        for category, items in results.items():
            f.write(f"## Category: `{category.upper()}`\n")
            for i, item in enumerate(items):
                f.write(f"### Example {i+1}\n")
                f.write(f"**Context:** `...{item['context']}`\n\n")
                f.write(f"**Gold Token:** `{item['gold_token']}`\n\n")
                
                # Build Comparison Table
                headers = ["Rank", "Teacher"] + list(s_logits_dict.keys())
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                
                for rank in range(5):
                    row = [f"**{rank+1}**"]
                    t_tok, t_prob = item['teacher'][rank]
                    row.append(f"`{t_tok}` ({(t_prob*100):.1f}%)")
                    
                    for name in s_logits_dict:
                        s_tok, s_prob = item[name][rank]
                        row.append(f"`{s_tok}` ({(s_prob*100):.1f}%)")
                        
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n---\n\n")

    print("Saved qualitative examples to qualitative_report.md!")

if __name__ == "__main__":
    main()
