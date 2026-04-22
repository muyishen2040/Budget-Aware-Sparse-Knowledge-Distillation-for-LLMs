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

def compute_kl_divergence(t_logits, s_logits):
    """Computes KL(Teacher || Student) for a single token position."""
    t_probs = F.softmax(t_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    return torch.sum(t_probs * (t_log_probs - s_log_probs), dim=-1).item()

def compute_entropy(logits):
    """Computes Shannon entropy for a single token position."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--student_paths", type=str, nargs="+", default=["output/real_full_kd", "output/real_topk_k8", "output/real_sampling_k8"])
    parser.add_argument("--student_names", type=str, nargs="+", default=["Full_KD", "TopK_8", "Sampling_8"])
    parser.add_argument("--output_path", type=str, default="qualitative_report.md")
    parser.add_argument("--num_val_samples", type=int, default=50)
    args = parser.parse_args()

    if len(args.student_paths) != len(args.student_names):
        print("Error: Number of student paths must match number of student names.")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    
    print("Loading models (This requires significant memory)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, torch_dtype=torch.float16, device_map="auto")
    teacher.eval()
    
    students = {}
    for name, path in zip(args.student_names, args.student_paths):
        if os.path.exists(path):
            print(f"Loading student {name} from {path}...")
            students[name] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
            students[name].eval()
        else:
            print(f"Warning: {path} not found. Skipping {name} in comparison.")
            
    if not students:
        print("Error: No student models were loaded correctly.")
        return

    print("Loading Data...")
    _, val_loader = get_dataloaders(tokenizer, seq_len=256, batch_size=4, num_train_samples=50, num_val_samples=args.num_val_samples)
    
    results = {
        "ambiguous": [],
        "topk_failure": [],
        "high_uncertainty": []
    }
    
    MAX_EXAMPLES_PER_CAT = 10 # Increased for better variety
    
    with torch.no_grad():
        for batch in val_loader:
            if all(len(v) >= MAX_EXAMPLES_PER_CAT for v in results.values()):
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            t_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            t_logits = t_outputs.logits.detach()
            
            s_logits_dict = {}
            for name, st_model in students.items():
                s_out = st_model(input_ids=input_ids, attention_mask=attention_mask)
                s_logits_dict[name] = s_out.logits.detach()
            
            masks = extract_qualitative_masks(t_logits, labels, k=8)
            
            for b in range(input_ids.size(0)):
                for seq_idx in range(40, input_ids.size(1) - 1):
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
                        context_ids = input_ids[b, max(0, pred_idx-40):pred_idx+1]
                        context_str = tokenizer.decode(context_ids)
                        gold_token = tokenizer.decode([labels[b, target_idx].item()])
                        
                        entry = {
                            "context": context_str,
                            "gold_token": gold_token,
                            "t_entropy": compute_entropy(t_logits[b, pred_idx]),
                            "teacher": decode_top_k(t_logits[b, pred_idx], tokenizer),
                            "students": {}
                        }
                        
                        for name, s_logits in s_logits_dict.items():
                            sl = s_logits[b, pred_idx]
                            entry["students"][name] = {
                                "topk": decode_top_k(sl, tokenizer),
                                "kl": compute_kl_divergence(t_logits[b, pred_idx], sl),
                                "entropy": compute_entropy(sl)
                            }
                            
                        results[extracted_category].append(entry)
                        break
                    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, "w") as f:
        f.write("# Enhanced Qualitative Distillation Analysis\n\n")
        f.write(f"**Teacher:** `{args.teacher_path}`\n\n")
        f.write("**Students compared:**\n")
        for name, path in zip(args.student_names, args.student_paths):
            f.write(f"- `{name}`: `{path}`\n")
        f.write("\n---\n\n")
        
        for category, items in results.items():
            f.write(f"## Category: `{category.upper()}`\n")
            if not items:
                f.write("No examples found in this category.\n\n")
                continue
                
            for i, item in enumerate(items):
                f.write(f"### Example {i+1}\n")
                f.write(f"**Context:** `...{item['context']}`\n\n")
                f.write(f"**Gold Token:** ` {item['gold_token']}`\n\n")
                
                # Metrics Table
                f.write("| Metric | Teacher | " + " | ".join(students.keys()) + " |\n")
                f.write("|---|" + "|".join(["---"] * (len(students) + 1)) + "|\n")
                
                entropy_row = ["**Entropy ($H$)**", f"{item['t_entropy']:.3f}"]
                kl_row = ["**Local $D_{KL}$**", "0.000"]
                
                for name in students.keys():
                    entropy_row.append(f"{item['students'][name]['entropy']:.3f}")
                    kl_row.append(f"{item['students'][name]['kl']:.3f}")
                
                f.write("| " + " | ".join(entropy_row) + " |\n")
                f.write("| " + " | ".join(kl_row) + " |\n\n")
                
                # Rankings Table
                headers = ["Rank", "Teacher"] + list(students.keys())
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                
                for rank in range(5):
                    row = [f"**{rank+1}**"]
                    t_tok, t_prob = item['teacher'][rank]
                    t_label = f"`{t_tok}` ({(t_prob*100):.1f}%)"
                    if t_tok.strip() == item['gold_token'].strip():
                        t_label = f"🎯 **{t_label}**"
                    row.append(t_label)
                    
                    for name in students.keys():
                        s_tok, s_prob = item['students'][name]['topk'][rank]
                        s_label = f"`{s_tok}` ({(s_prob*100):.1f}%)"
                        if s_tok.strip() == item['gold_token'].strip():
                            s_label = f"🎯 **{s_label}**"
                        row.append(s_label)
                        
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n---\n\n")

    print(f"Saved enhanced qualitative examples to {args.output_path}!")

if __name__ == "__main__":
    main()
