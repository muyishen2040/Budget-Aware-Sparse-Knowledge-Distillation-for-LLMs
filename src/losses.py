import torch
import torch.nn.functional as F

def compute_full_kd_loss(student_logits, teacher_logits, labels, temperature=1.0, alpha=0.1):
    shift_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    shift_student_logits = shift_logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
    
    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
    
    kl = F.kl_div(
        student_log_probs.view(-1, student_log_probs.size(-1)),
        teacher_probs.view(-1, teacher_probs.size(-1)),
        reduction='none'
    )
    kl = kl.sum(dim=-1).view(*shift_labels.shape)
    valid_mask = (shift_labels != -100)
    
    if valid_mask.any():
        kl_loss = kl[valid_mask].mean() * (temperature ** 2)
    else:
        kl_loss = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return loss, ce_loss, kl_loss

def compute_topk_kd_loss(student_logits, teacher_logits, labels, k=8, temperature=1.0, alpha=0.1):
    shift_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    shift_student_logits = shift_logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
    
    topk_teacher_logits, topk_indices = torch.topk(shift_teacher_logits, k, dim=-1)
    
    # Compute full log_softmax first to penalize non-topk probability mass
    student_full_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    
    # Gather student log_probs at top-K indices
    student_log_probs = torch.gather(student_full_log_probs, dim=-1, index=topk_indices)
    
    teacher_probs = F.softmax(topk_teacher_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        student_log_probs.view(-1, k),
        teacher_probs.view(-1, k),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return loss, ce_loss, kl_loss

def compute_sampling_kd_loss(student_logits, teacher_logits, labels, k=8, temperature=1.0, alpha=0.1):
    shift_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    shift_student_logits = shift_logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
    
    teacher_probs_full = F.softmax(shift_teacher_logits / temperature, dim=-1)
    flat_teacher_probs = teacher_probs_full.view(-1, teacher_probs_full.size(-1))
    
    sampled_indices_flat = torch.multinomial(flat_teacher_probs, num_samples=k, replacement=True)
    sampled_indices = sampled_indices_flat.view(*shift_teacher_logits.shape[:-1], k)
    
    sampled_teacher_logits = torch.gather(shift_teacher_logits, dim=-1, index=sampled_indices)
    
    # Compute full log_softmax first to enforce proper normalization
    student_full_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    
    # Gather student log_probs at sampled indices
    student_log_probs = torch.gather(student_full_log_probs, dim=-1, index=sampled_indices)
    
    teacher_probs = F.softmax(sampled_teacher_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        student_log_probs.view(-1, k),
        teacher_probs.view(-1, k),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return loss, ce_loss, kl_loss

def compute_cached_topk_kd_loss(student_logits, topk_teacher_probs, topk_indices, labels, temperature=1.0, alpha=0.1):
    shift_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    shift_topk_teacher_probs = topk_teacher_probs[..., :-1, :].contiguous().float()
    shift_topk_indices = topk_indices[..., :-1, :].contiguous()
    
    # Compute full log_softmax first to penalize non-topk probability mass
    student_full_log_probs = F.log_softmax(shift_logits / temperature, dim=-1)
    
    # Gather student log_probs based on teacher's topk_indices
    student_log_probs = torch.gather(student_full_log_probs, dim=-1, index=shift_topk_indices)
    
    # Renormalize teacher probabilities over the top-k support
    teacher_probs = shift_topk_teacher_probs / shift_topk_teacher_probs.sum(dim=-1, keepdim=True)
    
    k = shift_topk_indices.size(-1)
    kl = F.kl_div(
        student_log_probs.view(-1, k),
        teacher_probs.view(-1, k),
        reduction='none'
    )
    kl = kl.sum(dim=-1).view(*shift_labels.shape)
    valid_mask = (shift_labels != -100)
    
    if valid_mask.any():
        kl_loss = kl[valid_mask].mean() * (temperature ** 2)
    else:
        kl_loss = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    
    loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return loss, ce_loss, kl_loss

def compute_cached_sampling_kd_loss(
    student_logits,
    sampled_teacher_probs,
    sampled_indices,
    labels,
    temperature=1.0,
    alpha=0.1,
    ignore_index=-100,
):
    """
    student_logits:         [B, T, V]
    sampled_teacher_probs:  [B, T, K]   sparse teacher probs (already c_i / N)
    sampled_indices:        [B, T, K]   sampled vocab ids, padded with -1
    labels:                 [B, T]

    Returns:
        loss, ce_loss, kd_loss
    """

    # Standard next-token shift
    shift_student_logits = student_logits[..., :-1, :].contiguous().float()   # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()                               # [B, T-1]

    shift_teacher_probs = sampled_teacher_probs[..., :-1, :].contiguous().float()  # [B, T-1, K]
    shift_sampled_indices = sampled_indices[..., :-1, :].contiguous()              # [B, T-1, K]

    # CE loss
    ce_loss = F.cross_entropy(
        shift_student_logits.view(-1, shift_student_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # Valid sampled entries: padded ids are -1
    valid_sample_mask = shift_sampled_indices >= 0                            # [B, T-1, K]

    # Safe gather indices
    safe_indices = shift_sampled_indices.masked_fill(~valid_sample_mask, 0)

    # Full-vocab normalization first, then gather sampled positions
    student_full_log_probs = F.log_softmax(
        shift_student_logits / temperature, dim=-1
    )                                                                         # [B, T-1, V]

    gathered_student_log_probs = torch.gather(
        student_full_log_probs, dim=-1, index=safe_indices
    )                                                                         # [B, T-1, K]

    # Zero out padded positions so they contribute nothing
    gathered_student_log_probs = gathered_student_log_probs.masked_fill(
        ~valid_sample_mask, 0.0
    )
    teacher_probs = shift_teacher_probs.masked_fill(~valid_sample_mask, 0.0)

    # Renormalize sparse teacher probs over the valid K support so they sum to 1.
    # Raw sampling probs are c_i/N (counts/draws), which sum to << 1 when there
    # are few unique tokens. Without renormalization, the KL loss magnitude scales
    # with sum(t_i) * (-log s_i) ~ 0.3 * 10 = 3 at best, but blows up at init
    # because student probs are near-uniform (~1/50k). Renormalizing matches the
    # Top-K convention and keeps KL on the same scale as CE.
    teacher_prob_sum = teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    teacher_probs_norm = teacher_probs / teacher_prob_sum  # sum to 1 over valid K

    # True KL(teacher_norm || student) using F.kl_div for numerical stability.
    # F.kl_div(log_input, target) = sum target * (log target - log_input)
    # We pass log_student and teacher_norm; padded positions contribute 0 since teacher_norm=0.
    kl_per_entry = F.kl_div(
        gathered_student_log_probs,          # [B, T-1, K]  log s_k  (0 at padded)
        teacher_probs_norm,                  # [B, T-1, K]  t_k      (0 at padded)
        reduction="none",
        log_target=False,
    ).sum(dim=-1)                            # [B, T-1]

    # Only keep positions valid for both CE and KD
    valid_token_mask = (shift_labels != ignore_index) & valid_sample_mask.any(dim=-1)

    if valid_token_mask.any():
        kd_loss = kl_per_entry[valid_token_mask].mean() * (temperature ** 2)
    else:
        kd_loss = torch.zeros((), device=student_logits.device, dtype=shift_student_logits.dtype)

    loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return loss, ce_loss, kd_loss


def compute_cached_adaptive_topk_kd_loss(
    student_logits,
    topk_teacher_probs,
    topk_indices,
    valid_k,
    labels,
    temperature=1.0,
    alpha=0.1,
    ignore_index=-100,
):
    """
    Adaptive Top-K cached KD loss.

    student_logits:      [B, T, V]
    topk_teacher_probs:  [B, T, K_max]  padded with 0
    topk_indices:        [B, T, K_max]  padded with -1
    valid_k:             [B, T]          actual K per token (4, 8, or 16)
    labels:              [B, T]

    Returns:
        loss, ce_loss, kd_loss
    """
    # Standard next-token shift
    shift_student_logits = student_logits[..., :-1, :].contiguous().float()   # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()                               # [B, T-1]

    shift_teacher_probs = topk_teacher_probs[..., :-1, :].contiguous().float()  # [B, T-1, K_max]
    shift_indices = topk_indices[..., :-1, :].contiguous()                      # [B, T-1, K_max]
    shift_valid_k = valid_k[..., :-1].contiguous()                              # [B, T-1]

    K_max = shift_indices.size(-1)

    # CE loss
    ce_loss = F.cross_entropy(
        shift_student_logits.view(-1, shift_student_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # Build valid mask from valid_k: position j is valid iff j < valid_k[b, t]
    j_idx = torch.arange(K_max, device=shift_student_logits.device).unsqueeze(0).unsqueeze(0)  # [1, 1, K_max]
    valid_mask = j_idx < shift_valid_k.unsqueeze(-1)  # [B, T-1, K_max]

    # Safe gather indices (replace -1 with 0 for gather, then mask out)
    safe_indices = shift_indices.masked_fill(~valid_mask, 0)

    # Full-vocab normalization, then gather at adaptive positions
    student_full_log_probs = F.log_softmax(
        shift_student_logits / temperature, dim=-1
    )  # [B, T-1, V]

    gathered_student_log_probs = torch.gather(
        student_full_log_probs, dim=-1, index=safe_indices
    )  # [B, T-1, K_max]

    # Zero out padded positions
    gathered_student_log_probs = gathered_student_log_probs.masked_fill(~valid_mask, 0.0)
    teacher_probs = shift_teacher_probs.masked_fill(~valid_mask, 0.0)

    # Renormalize teacher probs over valid entries to sum to 1
    teacher_prob_sum = teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    teacher_probs_norm = teacher_probs / teacher_prob_sum

    # KL(teacher_norm || student)
    kl_per_entry = F.kl_div(
        gathered_student_log_probs,
        teacher_probs_norm,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)  # [B, T-1]

    # Only keep positions valid for both CE and KD
    valid_token_mask = (shift_labels != ignore_index) & valid_mask.any(dim=-1)

    if valid_token_mask.any():
        kd_loss = kl_per_entry[valid_token_mask].mean() * (temperature ** 2)
    else:
        kd_loss = torch.zeros((), device=student_logits.device, dtype=shift_student_logits.dtype)

    loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return loss, ce_loss, kd_loss


def compute_cached_adaptive_topk_weighted_kd_loss(
    student_logits,
    topk_teacher_probs,
    topk_indices,
    valid_k,
    labels,
    temperature=1.0,
    alpha=0.1,
    normalize_weights=False,
    min_weight=0.0,
    ignore_index=-100,
):
    """
    Head-Mass Weighted Adaptive Top-K KD loss.

    Same as compute_cached_adaptive_topk_kd_loss, but each token's KL
    contribution is weighted by the teacher's head mass (sum of raw Top-K
    probs before renormalization). Confident teacher positions get full
    weight; uncertain positions are naturally downweighted.

    Args:
        normalize_weights: If True, divide weights by their batch mean so
            the total gradient magnitude is preserved (redistribution only).
        min_weight: Floor for the per-token weight. Ensures every position
            contributes at least this fraction of the KD signal.
            E.g., min_weight=0.5 means even the most uncertain token gets
            at least 50% KD contribution.
    """
    shift_student_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()

    shift_teacher_probs = topk_teacher_probs[..., :-1, :].contiguous().float()
    shift_indices = topk_indices[..., :-1, :].contiguous()
    shift_valid_k = valid_k[..., :-1].contiguous()

    K_max = shift_indices.size(-1)

    # CE loss (unchanged)
    ce_loss = F.cross_entropy(
        shift_student_logits.view(-1, shift_student_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # Build valid mask
    j_idx = torch.arange(K_max, device=shift_student_logits.device).unsqueeze(0).unsqueeze(0)
    valid_mask = j_idx < shift_valid_k.unsqueeze(-1)

    safe_indices = shift_indices.masked_fill(~valid_mask, 0)

    # Full-vocab normalization, then gather
    student_full_log_probs = F.log_softmax(
        shift_student_logits / temperature, dim=-1
    )

    gathered_student_log_probs = torch.gather(
        student_full_log_probs, dim=-1, index=safe_indices
    )

    gathered_student_log_probs = gathered_student_log_probs.masked_fill(~valid_mask, 0.0)
    teacher_probs = shift_teacher_probs.masked_fill(~valid_mask, 0.0)

    # Head mass = sum of raw teacher probs over valid K (before renormalization)
    # This is our per-token confidence weight: high mass = confident teacher
    head_mass = teacher_probs.sum(dim=-1)  # [B, T-1]

    # Apply minimum weight floor
    weights = head_mass.clamp(min=min_weight)  # [B, T-1]

    # Optionally normalize to preserve total gradient magnitude
    if normalize_weights:
        valid_token_mask_for_norm = (shift_labels != ignore_index) & valid_mask.any(dim=-1)
        if valid_token_mask_for_norm.any():
            mean_w = weights[valid_token_mask_for_norm].mean().clamp(min=1e-9)
            weights = weights / mean_w

    # Renormalize teacher probs for KL computation
    teacher_prob_sum = head_mass.unsqueeze(-1).clamp(min=1e-9)
    teacher_probs_norm = teacher_probs / teacher_prob_sum

    # KL(teacher_norm || student)
    kl_per_entry = F.kl_div(
        gathered_student_log_probs,
        teacher_probs_norm,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)  # [B, T-1]

    # Weight KL by confidence weight
    weighted_kl = weights * kl_per_entry  # [B, T-1]

    valid_token_mask = (shift_labels != ignore_index) & valid_mask.any(dim=-1)

    if valid_token_mask.any():
        kd_loss = weighted_kl[valid_token_mask].mean() * (temperature ** 2)
    else:
        kd_loss = torch.zeros((), device=student_logits.device, dtype=shift_student_logits.dtype)

    loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return loss, ce_loss, kd_loss


def compute_cached_adaptive_topk_tail_summary_kd_loss(
    student_logits,
    topk_teacher_probs,
    topk_indices,
    valid_k,
    labels,
    temperature=1.0,
    alpha=0.1,
    tail_weight=0.1,
    ignore_index=-100,
):
    """
    Adaptive Top-K Tail Summary cached KD loss.

    Computes KL divergence on K+1 classes dynamically per token,
    with separate weighting for the head (Top-K) and tail (summary bucket)
    components of the KD loss.

    Args:
        tail_weight: Coefficient (beta) controlling how strongly the tail
            calibration signal influences training. 0.0 = pure Adaptive Top-K
            (no tail), 1.0 = equal weight to head and tail KL.
    """
    shift_student_logits = student_logits[..., :-1, :].contiguous().float()   # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()                               # [B, T-1]

    shift_teacher_probs = topk_teacher_probs[..., :-1, :].contiguous().float()  # [B, T-1, K_max]
    shift_indices = topk_indices[..., :-1, :].contiguous()                      # [B, T-1, K_max]
    shift_valid_k = valid_k[..., :-1].contiguous()                              # [B, T-1]

    K_max = shift_indices.size(-1)

    ce_loss = F.cross_entropy(
        shift_student_logits.view(-1, shift_student_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # Build valid mask
    j_idx = torch.arange(K_max, device=shift_student_logits.device).unsqueeze(0).unsqueeze(0)
    valid_mask = j_idx < shift_valid_k.unsqueeze(-1)  # [B, T-1, K_max]

    safe_indices = shift_indices.masked_fill(~valid_mask, 0)

    # Full-vocab student log probs (for head KL, computed over full vocab normalization)
    student_full_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)  # [B, T-1, V]
    student_full_probs = F.softmax(shift_student_logits / temperature, dim=-1)           # [B, T-1, V]

    gathered_student_log_probs = torch.gather(
        student_full_log_probs, dim=-1, index=safe_indices
    )  # [B, T-1, K_max]
    gathered_student_probs = torch.gather(
        student_full_probs, dim=-1, index=safe_indices
    )  # [B, T-1, K_max]

    # Zero out padded positions
    gathered_student_log_probs = gathered_student_log_probs.masked_fill(~valid_mask, 0.0)
    gathered_student_probs = gathered_student_probs.masked_fill(~valid_mask, 0.0)
    teacher_probs = shift_teacher_probs.masked_fill(~valid_mask, 0.0)

    # --- Head KL: standard adaptive top-K KL (renormalized teacher over valid K) ---
    teacher_prob_sum = teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    teacher_probs_norm = teacher_probs / teacher_prob_sum

    head_kl_per_entry = F.kl_div(
        gathered_student_log_probs,
        teacher_probs_norm,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)  # [B, T-1]

    # --- Tail KL: single-bucket calibration term ---
    student_tail_mass = (1.0 - gathered_student_probs.sum(dim=-1)).clamp(min=1e-9)  # [B, T-1]
    teacher_tail_mass = (1.0 - teacher_probs.sum(dim=-1)).clamp(min=1e-9)           # [B, T-1]

    # KL for a single Bernoulli-like bucket: t * log(t/s)
    tail_kl_per_entry = teacher_tail_mass * (
        torch.log(teacher_tail_mass) - torch.log(student_tail_mass)
    )  # [B, T-1]

    # Combined KD loss with tail weight
    kl_per_entry = head_kl_per_entry + tail_weight * tail_kl_per_entry

    valid_token_mask = (shift_labels != ignore_index) & valid_mask.any(dim=-1)

    if valid_token_mask.any():
        kd_loss = kl_per_entry[valid_token_mask].mean() * (temperature ** 2)
    else:
        kd_loss = torch.zeros((), device=student_logits.device, dtype=shift_student_logits.dtype)

    loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return loss, ce_loss, kd_loss