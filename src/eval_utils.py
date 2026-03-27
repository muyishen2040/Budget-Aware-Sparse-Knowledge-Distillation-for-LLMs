import os
import torch
import torch.nn.functional as F
from typing import Dict

def compute_lm_metrics(ce_loss: float) -> Dict[str, float]:
    """
    \subsection{Language Modeling Performance}
    Compute standard language modeling metrics: NLL and Perplexity (PPL).
    """
    nll = ce_loss
    # Clamp to avoid overflow if CE loss is very high early in training
    ppl = torch.exp(torch.tensor(min(nll, 20.0))).item()
    if nll > 20.0:
        ppl = float('inf')
    return {"nll": nll, "ppl": ppl}

def calculate_budget(method: str, **kwargs) -> int:
    """
    \subsection{Budget Efficiency}
    Calculate the per-token budget B (number of individual scalar FP16/INT values stored).
    """
    if method == "full":
        vocab_size = kwargs.get("vocab_size", 50277) # Pythia default vocab size
        return vocab_size
    elif method == "topk":
        k = kwargs.get("k", 8)
        # K values + K indices
        return k * 2 
    elif method == "sampling":
        num_draws = kwargs.get("num_draws", 50)
        # Assuming we store num_draws unique subsets (at most padded to num_draws)
        # Indices + probabilities/counts
        return num_draws * 2
    else:
        # Placeholder for heuristic methods or PCA
        raise NotImplementedError(f"Budget calculation for {method} not implemented yet.")

def get_cache_size_mb(cache_path: str) -> float:
    """
    \subsection{Budget Efficiency}
    Return the total file storage size in MB to compare memory efficiency.
    """
    if os.path.exists(cache_path):
        return os.path.getsize(cache_path) / (1024 * 1024)
    return 0.0

def extract_qualitative_masks(
    teacher_logits: torch.Tensor, 
    labels: torch.Tensor, 
    k: int = 8, 
    entropy_threshold: float = 3.0, 
    ambiguity_threshold: float = 0.05
) -> Dict[str, torch.Tensor]:
    """
    \subsection{Qualitative Analysis}
    Finds interesting qualitative behavior tokens from a batch of teacher logits.
    Returns boolean masks of shape [Batch, SequenceLen] for each scenario.
    """
    probs = F.softmax(teacher_logits, dim=-1)
    
    # 1. High-uncertainty contexts (diffuse distribution)
    log_probs = F.log_softmax(teacher_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    high_uncertainty_mask = (entropy > entropy_threshold)
    
    # 2. Ambiguous tokens (top-1 and top-2 probabilities are similar)
    top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)
    prob_diff = top2_probs[..., 0] - top2_probs[..., 1]
    # We enforce a >0.1 check so we aren't looking at flat distributions (which are high uncertainty)
    ambiguous_mask = (prob_diff <= ambiguity_threshold) & (top2_probs[..., 0] > 0.1)
    
    # 3. Failure cases of Top-K truncation (gold token lies outside the top-K)
    _, topk_indices = torch.topk(probs, k, dim=-1)
    expanded_labels = labels.unsqueeze(-1)
    in_topk_mask = (topk_indices == expanded_labels).any(dim=-1)
    
    # Ignore padded labels/ignored indices (e.g. padding often set to -100 or less than 0)
    valid_labels_mask = (labels >= 0)
    truncation_failure_mask = (~in_topk_mask) & valid_labels_mask
    
    return {
        "high_uncertainty": high_uncertainty_mask,
        "ambiguous": ambiguous_mask,
        "topk_failure": truncation_failure_mask,
        "entropy": entropy
    }

def print_evaluation_summary(method: str, ce_loss: float, cache_path: str = None, **budget_kwargs):
    """
    Convenience reporting method to aggregate Language Modeling and Budget Efficiency.
    """
    metrics = compute_lm_metrics(ce_loss)
    budget = calculate_budget(method, **budget_kwargs)
    
    print(f"--- Evaluation Summary: {method.upper()} ---")
    print(f"NLL (CE Loss): {metrics['nll']:.4f}")
    print(f"Perplexity:    {metrics['ppl']:.4f}")
    print(f"Budget (B) / token: {budget} scalars")
    
    if cache_path and os.path.exists(cache_path):
        size_mb = get_cache_size_mb(cache_path)
        print(f"Cache File Size: {size_mb:.2f} MB")
    
    print("-" * 45)
