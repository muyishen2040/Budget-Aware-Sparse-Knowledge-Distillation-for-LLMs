import torch
import torch.nn.functional as F
from AutoEncoder.autoencoder import KDAautoEncoder
import os 

def hybrid_loss(compressedk_probs, ae_model, student_logits, topk_probs, topk_ids, labels, temperature=1.0, alpha=0.5):
    '''
    Adds a compression loss term to the standard KD loss.
    ''' 
    loss, ce_loss, kl_loss = compute_cached_topk_kd_loss(compressedk_probs, ae_model, student_logits, topk_probs, topk_ids, labels, temperature=temperature, alpha=alpha)
    ae_loss = compression_loss(compressedk_probs, ae_model, student_logits, topk_probs, topk_ids, labels, temperature=temperature, alpha=alpha)

    n_ae = float(os.environ.get("n_ae", 0.5))
    n_kd = float(os.environ.get("n_kd", 0.5))
    
    hybrid_loss = n_ae * ae_loss + n_kd * kl_loss
    total_loss = alpha * ce_loss + (1 - alpha) * hybrid_loss
    
    return total_loss, ce_loss, ae_loss


def compression_loss(compressedk_probs, ae_model, student_logits, topk_probs, topk_ids, labels, temperature=1.0, alpha=0.5):
    """
    Computes KL(compressed_teacher || student) with optional latent regularization.
    Args:
        compressedk_probs: Tensor of shape (B, T, K) — Compressed Teacher distributions, softmaxed along K.
        ae_model: The autoencoder model, used to compute the latent representation z if lambda_latent > 0.
        student_logits: Tensor of shape (B, T, V) — Student model logits before softmax.
        topk_probs: Tensor of shape (B, T, K) — Teacher probabilities for the top-K tokens.
        topk_ids: Tensor of shape (B, T, K) — Token IDs corresponding to the top-K probabilities.
        labels: Tensor of shape (B, T) — Ground truth token IDs for CE loss.
        temperature: Temperature for scaling logits in KD loss.
        alpha: Weighting factor between CE loss and KD loss.
    Returns:
        Total scalar loss (mean over batch)
    """
    
    # (1) time shift
    student_logits = student_logits[..., :-1, :].contiguous().float()  # [B, T-1, V]
    topk_prob = topk_probs[..., :-1, :].contiguous().float()  # [B, T-1, K]
    topk_ids = topk_ids[..., :-1, :].contiguous() # [B, T-1, K]
    compressedk_probs = compressedk_probs[..., :-1, :].contiguous().float()  # [B, T-1, K]
    shift_labels = labels[..., 1:].contiguous()
    
    # (2) compute the softmax of the student logits to get a valid distribution, then pass through the AE to get the compressed distribution in the AE latent space
    student_full_probs = F.softmax(student_logits / temperature, dim=-1)  # [B, T-1, V]
    _, student_compressed_probs = ae_model(student_full_probs.to(dtype=torch.float32)) # [B, T-1, V] -> [B, T-1, K]
    
    # (3) get the student log probs from the compressed student probs
    student_compressed_log_probs = torch.log(student_compressed_probs)  # [B, T-1, K]
    
    # (4) renormalize the compressed teacher probs to ensure it's a valid distribution over the K support
    teacher_compressed_probs = compressedk_probs / compressedk_probs.sum(dim=-1, keepdim=True)  # [B, T-1, K]
    
    # (5) compute the KL divergence between the compressed teacher distribution and the compressed student distribution as the compression loss
    k = topk_ids.size(-1)
    kl_compression_tensor = F.kl_div(
        student_compressed_log_probs.view(-1, k),
        teacher_compressed_probs.view(-1, k),
        reduction='none'
    )
    kl_compression_tensor = kl_compression_tensor.sum(dim=-1).view(*shift_labels.shape) #shape of shift_labels = [B, T-1]
    valid_mask = (shift_labels != -100)
    if valid_mask.any():
        kl_compression_loss = kl_compression_tensor[valid_mask].mean() * (temperature ** 2)
    else:
        kl_compression_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        
    return kl_compression_loss
    
 
def compute_fusion_loss(compressedk_probs, ae_model, student_logits, topk_teacher_probs, topk_indices, labels, temperature=1.0, alpha=0.5):
    '''
    A more complex loss that adaptively fuses the standard KD loss on the teacher's top-k predictions with the AE-based compression loss based on the teacher's confidence. 
    For positions where the teacher is confident in its top-k predictions, we apply the standard KD loss. 
    For positions where the teacher is not confident, we apply the compression loss which distills the student to match the teacher's compressed distribution in the AE latent space. 
    The intuition is that when the teacher is uncertain, its full distribution may be noisy and unhelpful for distillation, so we fall back to a softer signal from the compressed
    distribution which may still capture useful information about which tokens are generally more likely than others without forcing the student to match potentially spurious peaks 
    in the teacher's distribution.
    '''
    
    # (1) shift the student_logits, topk_teacher_probs/indices, labels, and compressedk_probs along the time dimension by one step.
    student_logits = student_logits[..., :-1, :].contiguous().float()  # [B, T-1, V]
    topk_prob = topk_teacher_probs[..., :-1, :].contiguous().float()  # [B, T-1, K]
    topk_ids = topk_indices[..., :-1, :].contiguous() # [B, T-1, K]
    shift_labels = labels[..., 1:].contiguous()
    compressedk_probs = compressedk_probs[..., :-1, :].contiguous().float()  # [B, T-1, K]

    # (2) find the sum over each row of topk_prob while keeping dimensions.
    prob_sum = topk_prob.sum(dim=-1)  # [B, T-1]
    #prob_max = torch.amax(topk_prob, dim=-1)  # [B, T-1]
    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
    M = (prob_sum >= confidence_threshold) 
    
    # (3) use the mask 'M' to partition out the rows from the student_logits, topk_prob, and topk_ids where the teacher is confident vs. not confident
    # (a) topk_probs. Let N+ be the number of positions where the teacher is confident, and N- be the number of positions where the teacher is not confident. Then we have:
    topk_confident_probs = topk_prob[M]  # [N+, K]
    topk_unconfident_probs = topk_prob[~M]  # [N-, K]
    # (b) topk_indices
    topk_confident_ids = topk_ids[M]  # [N+, K]
    topk_unconfident_ids = topk_ids[~M]  # [N-, K]
    # (c) student_logits
    student_confident_logits = student_logits[M]  # [N+, V]
    student_unconfident_logits = student_logits[~M]  # [N-, V]
    
    # (4) for the confident rows, compute the log softmax of the student logits, gatther the log probs at the teacher's top-k indices, and compute the KL divergence loss between the student 
    # and teacher top-k distributions as in standard top-k KD
    if student_confident_logits is not None and student_confident_logits.shape[0] > 0:
        student_full_logprobs_confident = F.log_softmax(student_confident_logits / temperature, dim=-1)  # [N+, V]
        student_logprobs_confident = torch.gather(student_full_logprobs_confident, dim=-1, index=topk_confident_ids)  # [N+, K]
        teacher_probs_confident = topk_confident_probs / topk_confident_probs.sum(dim=-1, keepdim=True)  # renormalize to ensure it's a valid distribution over the top-k support
        k = topk_ids.size(-1)
        kl_confident_tensor = F.kl_div(
            student_logprobs_confident.view(-1, k),
            teacher_probs_confident.view(-1, k),
            reduction='none'
        ) 
        kl_confident_tensor = kl_confident_tensor.sum(dim=-1).view(*shift_labels[M].shape) #shape of shift_labels[M] = [1, N+] 
        valid_mask = (shift_labels[M] != -100)
        
        if valid_mask.any():
            kl_confident = kl_confident_tensor[valid_mask].mean() * (temperature ** 2)
        else:
            kl_confident = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    else:
      kl_confident = 0.0

    # (5) for the unconfident rows, first the ae_model must be used to compress the vocab dimension from V --> K in the student logits
    if student_unconfident_logits is not None and student_unconfident_logits.shape[0] > 0:
        student_unconfident_fullprobs = F.softmax(student_unconfident_logits / temperature, dim=-1)  # [N-, V]   
        _, student_unconfident_probs = ae_model(student_unconfident_fullprobs.to(dtype=torch.float32))# [N-, V] -> [N-, K]
        student_unconfident_logprobs = torch.log(student_unconfident_probs)  # don't forget that we need log probs of x_hat for KL 
        # FOR THE TEACHER DISTRIBUTION, WE USE THE COMPRESSED-K PROBS IN THE AE LATENT SPACE.
        teacher_probs_unconfident = compressedk_probs[~M] # [N-, K] the teacher distribution for the unconfident positions is the compressed distribution in the AE latent space
        assert teacher_probs_unconfident.shape == student_unconfident_logprobs.shape, "Shape mismatch between student and teacher distributions for unconfident positions"
        k = topk_ids.size(-1)
        kl_unconfident_tensor = F.kl_div(
            student_unconfident_logprobs.view(-1, k),
            teacher_probs_unconfident.view(-1, k),
            reduction='none'
        ) 
        kl_unconfident_tensor = kl_unconfident_tensor.sum(dim=-1).view(*shift_labels[~M].shape) #shape of shift_labels[~M] = [1, N-] 
        valid_mask = (shift_labels[~M] != -100)
        if valid_mask.any():
            kl_unconfident = kl_unconfident_tensor[valid_mask].mean() * (temperature ** 2)
        else:
            kl_unconfident = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    else:
        kl_unconfident = 0.0

    # (6) compute the final loss as a weighted average of the confident vs. unconfident KL losses, using alpha to weight against the CE loss as well
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels[..., 1:].contiguous().view(-1))
    loss = alpha * ce_loss + (1 - alpha) * (kl_confident + kl_unconfident) 
    return loss, ce_loss, (kl_confident + kl_unconfident)
            
                               
def compute_cached_topk_kd_loss(compressedk_probs, ae_model, student_logits, topk_teacher_probs, topk_indices, labels, temperature=1.0, alpha=0.5):  
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