import torch
import torch.nn.functional as F
from AutoEncoder.autoencoder import KDAautoEncoder


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

def compute_cached_topk_kd_loss(compressedk_probs, AE_model, student_logits, topk_teacher_probs, topk_indices, labels, temperature=1.0, alpha=0.1):
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
    
    # ADD A LOSS TERM USING THE AE-COMPRESSED PROBS
    # Given: compressedk_probs from the AE of shape [B, T, K] and student_logits of shape [B, T, V]
    # (1) Get compressed teacher probs
    compressed_teacher_probs = compressedk_probs # [B, T, K] here, K matches the AE latent dim
    # (2) shift compressed teacher probs to align with student logits
    shift_compressed_teacher_probs = compressed_teacher_probs[..., :-1, :].contiguous().float() # [B, T-1, K]
    # (3) convert the student logits to log probs over the full vocab. Here, we want the student logits to be of shape [B, T, V] (not shifted yet)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1) # [B, T, V]
    # (4) compress the student log probs from [B, T, V] to [B, T, K] using the AE encoder. 
    _, compressed_student_probs = AE_model(student_log_probs) # [B, T, V] -> [B, T, K]
    # (5) shift the compressed student probs to align with the teacher
    shift_compressed_student_probs = compressed_student_probs[..., :-1, :].contiguous().float() # [B, T-1, K]
    shift_compressed_student_logprobs = torch.log(shift_compressed_student_probs + 1e-9) # add small value for numerical stability
    # (6) compute KL divergence between compressed student and compressed teacher distributions
    kl_compressed = F.kl_div(
        shift_compressed_student_logprobs.view(-1, shift_compressed_student_logprobs.size(-1)),
        shift_compressed_teacher_probs.view(-1, shift_compressed_teacher_probs.size(-1)),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    loss = alpha * ce_loss + (1 - alpha) * kl_loss + (1 - alpha)/2 * kl_compressed  # weight the compressed KL loss as well
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