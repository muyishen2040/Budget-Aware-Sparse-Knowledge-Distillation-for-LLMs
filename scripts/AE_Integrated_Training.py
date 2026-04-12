import os
from dataclasses import dataclass
import time
from typing import Literal, Optional, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

import argparse
from src.models import load_teacher
from src.data import get_dataloaders
import json
from huggingface_hub import login, HfApi
import os
from datasets import Dataset
#import shutil
import subprocess
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class CacheConfig:
    # which caches to create: "topk", "sampling", or "both"
    # ADD A MODE TO CACHE FULL LOGITS --> SET FULL LOGITS TO THE DEFAULT MODE
    #==============================================================================================
    mode: Literal["topk", "sampling", "both", "full_logits"] = "full_logits"
    #==============================================================================================

    # data
    seq_len: int = 256
    batch_size: int = 8 #4 #32
    num_train_samples: Optional[int] = 2000  # set None for full train split

    # output
    cache_dir: str = "teacher_cache"
    save_per_split_single_file: bool = False   # if False, save shards instead
    shard_size_batches: int = 50 #100             # used only when save_per_split_single_file=False

    # top-k settings
    topk_k: int = 16

    # random sampling KD settings
    sampling_num_draws: int = 50
    temperature: float = 1.0  # unified temperature

    # dtype to store probabilities
    probs_dtype: torch.dtype = torch.float32


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def teacher_forward(
    teacher: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
    return logits


# ADD A METHOD: build_all_logits_softlabels() that just returns the full teacher distribution as probs (after temperature scaling) 
# for each token, without top-k truncation or sampling. This would be used for the "full_logits" mode of caching, and the corresponding 
# loss would be standard KL divergence over the full vocabulary.
#==============================================================================================
def build_full_logits_softlabels(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    probs_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Build full teacher soft labels (no truncation or sampling).

    Returns:
        {
            "probs": [B, T, V],  full teacher probabilities after temperature scaling
        }
    """
    # find full output prob dist over all batches, seq length, and vocab tokens
    probs = F.softmax(logits / temperature, dim=-1).to(dtype=probs_dtype)
    
    # find the k most probable tokens at each position for logging/debugging (not used in loss)
    #topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1) # shape [B, T, K]
    
    return {
        "probs": probs.cpu(),  # [B, T, V]
    }


def init_storage(mode: str) -> Dict[str, Dict[str, list]]:
    storage: Dict[str, Dict[str, list]] = {}

    if mode == "full_logits":
        storage["full_logits"] = {"probs": []}
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return storage





def cache_split(
    teacher: torch.nn.Module,
    dataloader,
    split_name: str,
    config: CacheConfig,
) -> None:
    """
    Similar to caching, but instead accumulates training data in batches without ever saving to disk or pushing to hub.
    Instead, the training data is immediately used to integrated training and then deleted from RAM to free up space.
    """
    teacher.eval()
    device = get_model_device(teacher)
    #ensure_dir(config.cache_dir)

    print(f"\nCaching split: {split_name}")
    print(f"Mode: {config.mode}")

    # For large sampling budgets, always shard to avoid OOM
    force_shard_sampling = (
        config.mode in ("sampling", "both", "full_logits")  # full_logits can also be large if seq_len and vocab are large
        and config.sampling_num_draws >= 16
    ) # --> True
    
    # TURN ON FORCED SHARDING 
    #==============================================================================================
    assert force_shard_sampling == True, "Sharding is needed for integrated AE training."
    #==============================================================================================
    storage = init_storage(config.mode)
    shard_idx = 0
    batch_counter = 0
    sampling_shard_paths: list = []  # track shard paths for later merge

    for batch in tqdm(dataloader, desc=f"Caching {split_name}"):
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
#        labels = batch["labels"].to(device)
        logits = teacher_forward(teacher, input_ids, attention_mask)

        if config.mode == "full_logits":
            full_logits_out = build_full_logits_softlabels(
                logits=logits,
                k=config.topk_k,  # pass k for logging/debugging but it won't affect the loss since we return the full distribution
                temperature=config.temperature,
                probs_dtype=config.probs_dtype,
            )
            storage["full_logits"]["probs"].append(full_logits_out["probs"])  #append probs tensor to running list for current shard
         
        else:
            raise NotImplementedError("AE Integrated Training is only supported for full_logit mode!")
        batch_counter += 1

        should_shard = (
            (not config.save_per_split_single_file) 
            or force_shard_sampling                 # --> True
        ) and (batch_counter % config.shard_size_batches == 0)
        
        if batch_counter % config.shard_size_batches == 0:
            assert should_shard == True, "Sharding should be forced on for sampling mode with num_draws >= 16 to avoid OOM, and save_per_split_single_file is ignored in this case. Please check the logic for should_shard if you see this assertion error."
        if should_shard:
            if force_shard_sampling and "full_logits" in storage:
                
                full_logits_payload = torch.stack(storage["full_logits"]['probs'])  # list of [B, T, V] tensors
                
                dataset = TensorDataset(full_logits_payload, full_logits_payload) # AE dataset where the labels are the same as the inputs since it's an autoencoder
                dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
                
                print(f"Starting integrated training on shard {shard_idx} with {full_logits_payload.shape[0]} samples...")
                for batch_features, batch_labels in dataloader:
                    print(f"Batch features shape: {batch_features.shape} | Batch labels shape: {batch_labels.shape}")
                    # Here you would add your training code that takes batch_features as input and tries to reconstruct
               
               # after the training loop finishes, we can delete the full_logits from storage to free up RAM before processing the next shard
                try:
                    del dataloader
                    del dataset
                    del storage["full_logits"]
                    print(f"Deleted full_logits from storage to free up RAM after training on shard {shard_idx}.")
                except Exception as e:
                    print(f"Error deleting full_logits from storage: {e}. You may need to free up RAM manually.")
                    
                print("EARLY EXIT FOR TESTING - REMOVE THIS AFTER INTEGRATING TRAINING CODE")
                return
    
    
    





def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Cache teacher soft-labels for sparse KD.",
        epilog="""
Dataset keys (for --dataset):
  wikitext            WikiText-103 (default)
  github-code         GitHub Code, all languages
  github-code-python  GitHub Code, Python only
  pubmed              PubMed abstracts
"""
    )
    # add "full_logits" to the mode choices and set it as the default
    parser.add_argument("--mode", type=str, default="full_logits", choices=["topk", "sampling", "both", "full_logits"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--cache_dir", type=str, default="teacher_cache")
    parser.add_argument("--topk_k", type=int, default=8)
    parser.add_argument("--sampling_num_draws", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        help="Dataset key: 'wikitext', 'github-code', 'github-code-python', 'pubmed'"
    )
    args = parser.parse_args()

    config = CacheConfig(
        mode=args.mode, # "full_logits" by default, can be set to "topk", "sampling", or "both" for different caches
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_train_samples=args.num_train_samples if args.num_train_samples > 0 else None,
        cache_dir=args.cache_dir,
        save_per_split_single_file=True,
        shard_size_batches=10, #40, #50, #100
        topk_k=args.topk_k,
        sampling_num_draws=args.sampling_num_draws,
        temperature=args.temperature,
        probs_dtype=torch.float32,
    )

    teacher, tokenizer = load_teacher()
    teacher.eval()

    train_loader, val_loader = get_dataloaders(
        tokenizer,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_train_samples=config.num_train_samples,
        train_dataset_name=args.dataset,
        val_dataset_name=args.dataset,
    )

    cache_split(teacher, train_loader, "train", config)
   # cache_split(teacher, val_loader, "val", config)


if __name__ == "__main__":
    main()