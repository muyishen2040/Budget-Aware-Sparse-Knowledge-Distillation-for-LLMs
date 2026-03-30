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

    # Sparse forward KL up to the teacher entropy constant:
    # KL(t || s) = sum_i t_i (log t_i - log s_i)
    # For optimization, the log t_i term is constant, so minimizing
    # -sum_i t_i log s_i is equivalent.
    per_token_kd = -(teacher_probs * gathered_student_log_probs).sum(dim=-1)  # [B, T-1]

    # Only keep token positions that are valid for both CE labels and KD samples
    valid_token_mask = (shift_labels != ignore_index) & valid_sample_mask.any(dim=-1)  # [B, T-1]

    if valid_token_mask.any():
        kd_loss = per_token_kd[valid_token_mask].mean()
    else:
        kd_loss = torch.zeros((), device=student_logits.device, dtype=shift_student_logits.dtype)

    loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return loss, ce_loss, kd_loss