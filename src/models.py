from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_teacher():
    teacher_id = "EleutherAI/pythia-1.4b"
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    teacher.eval()
    return teacher, tokenizer

def load_student():
    student_id = "EleutherAI/pythia-160m"
    
    # We still need tokenizer from teacher to ensure parity
    teacher_id = "EleutherAI/pythia-1.4b"
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        student_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return student, tokenizer

def load_models():
    teacher, tokenizer = load_teacher()
    student, _ = load_student()
    return teacher, student, tokenizer
