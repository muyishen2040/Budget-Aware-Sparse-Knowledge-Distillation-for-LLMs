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
from huggingface_hub import login
import os
from datasets import Dataset

# HF API instance for uploading cached shards to Hugging Face Hub
#HF_api = HfApi()

@dataclass
class CacheConfig:
    # which caches to create: "topk", "sampling", or "both"
    # ADD A MODE TO CACHE FULL LOGITS --> SET FULL LOGITS TO THE DEFAULT MODE
    #==============================================================================================
    mode: Literal["topk", "sampling", "both", "full_logits"] = "full_logits"
    #==============================================================================================

    # data
    seq_len: int = 256
    batch_size: int = 4 #32
    num_train_samples: Optional[int] = 2000  # set None for full train split

    # output
    cache_dir: str = "teacher_cache"
    save_per_split_single_file: bool = False   # if False, save shards instead
    shard_size_batches: int = 10 #50 #100             # used only when save_per_split_single_file=False

    # top-k settings
    topk_k: int = 8

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
    topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
    
    return {
        "probs": probs.cpu(),  # [B, T, V]
        "topk_ids": topk_ids.cpu(),  # [B, T, K]
        "topk_probs": topk_probs.cpu(),  # [B, T, K]
    }
#==============================================================================================

def build_topk_softlabels(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    probs_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Build sparse Top-K teacher soft labels.

    Returns:
        {
            "topk_ids":   [B, T, K],
            "topk_probs": [B, T, K],
        }

    We store probabilities, not logits, because Top-K KD in the paper is framed
    as keeping the teacher probabilities on the top-k tokens and zeroing out the rest,
    which is a biased truncation-based estimator. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    """
    probs = F.softmax(logits / temperature, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)

    return {
        "topk_ids": topk_ids.cpu(),
        "topk_probs": topk_probs.to(dtype=probs_dtype).cpu(),
    }


def build_sampling_softlabels(
    logits: torch.Tensor,
    num_draws: int,
    temperature: float = 1.0,
    probs_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Build sparse random-sampling teacher soft labels to match the paper.

    Paper-consistent behavior:
    - sample from teacher distribution q = t_full^tau
    - for tau = 1, sample directly from teacher probs
    - sample WITH replacement
    - final sparse target is t_i^s = c_i / N, where c_i is count among N draws
      :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

    Returns:
        {
            "sampled_ids":      [B, T, Umax]   with padded invalid id = -1
            "sampled_counts":   [B, T, Umax]
            "sampled_probs":    [B, T, Umax]   (= counts / num_draws)
        }

    Umax <= num_draws, padded to exactly num_draws for easy batching.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # proposal q ∝ softmax(logits / tau)  — correct temperature scaling
    proposal_probs = F.softmax(logits / temperature, dim=-1)  # [B, T, V]

    B, T, V = proposal_probs.shape
    flat_probs = proposal_probs.reshape(-1, V)  # [B*T, V]

    # Important: WITH replacement, matching the paper
    flat_sampled = torch.multinomial(
        flat_probs,
        num_samples=num_draws,
        replacement=True,
    )  # [B*T, num_draws]

    max_unique = num_draws

    sampled_ids_list = []
    sampled_counts_list = []
    sampled_probs_list = []

    # Move to CPU immediately to avoid spinning across 4096 GPU syncs per batch
    flat_sampled_cpu = flat_sampled.cpu()
    
    # Iterate over batch*seq_len (e.g. 4096). Working with ~K elements per row instead of 50,277 elements!
    for row in flat_sampled_cpu:
        ids, cts = torch.unique(row, return_counts=True)
        prs = cts.to(torch.float32) / float(num_draws)

        pad_len = max_unique - ids.numel()
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=-1)
            cts = F.pad(cts, (0, pad_len), value=0)
            prs = F.pad(prs, (0, pad_len), value=0.0)

        sampled_ids_list.append(ids)
        sampled_counts_list.append(cts)
        sampled_probs_list.append(prs)

    sampled_ids = torch.stack(sampled_ids_list, dim=0).view(B, T, max_unique)
    sampled_counts = torch.stack(sampled_counts_list, dim=0).view(B, T, max_unique)
    sampled_probs = (
        torch.stack(sampled_probs_list, dim=0)
        .to(dtype=probs_dtype)
        .view(B, T, max_unique)
    )

    return {
        "sampled_ids": sampled_ids.cpu(),
        "sampled_counts": sampled_counts.cpu(),
        "sampled_probs": sampled_probs.cpu(),
    }


def init_storage(mode: str) -> Dict[str, Dict[str, list]]:
    storage: Dict[str, Dict[str, list]] = {}

    # ADD A STORAGE KEY FOR FULL LOGITS MODE
    #==============================================================================================
    if mode == "full_logits":
        storage["full_logits"] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "probs": [],  # store full teacher probs (after temperature scaling) here
            "topk_ids": [],
            "topk_probs": [],
        }
    #==============================================================================================
    
    elif mode in ("topk", "both"):
        storage["topk"] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "topk_ids": [],
            "topk_probs": [],
        }

    elif mode in ("sampling", "both"):
        storage["sampling"] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "sampled_ids": [],
            "sampled_counts": [],
            "sampled_probs": [],
        }
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return storage


def append_common_tensors(
    target_dict: Dict[str, list],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    target_dict["input_ids"].append(input_ids.cpu())
    target_dict["attention_mask"].append(attention_mask.cpu())
    target_dict["labels"].append(labels.cpu())


def concat_storage(data_dict: Dict[str, list]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, values in data_dict.items():
        if isinstance(values, list):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values
    return out


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)
    print(f"Saved: {path}")


def push_to_hf_hub(local_path: str, data: Dict[str, Any]) -> None:
    
    if "train" in local_path:
        split = "train"
    elif "val" in local_path:
        split = "val"
    
    # convert the data dictionary to a hf dataset and push to the hub
    dataset = Dataset.from_dict(data)
    
    # push the dataset to the hub under the specified repo and path
    try:
        dataset.push_to_hub(
        repo_id="jmcochrane/Sparse_KD_AE_Training_Data",
        path_in_repo=f"{split}/{os.path.basename(local_path)}",
        private=False,
        )
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
    
  #  HF_api.upload_file(
  #      path_or_fileobj=local_path, # Local file path
  #      path_in_repo=split, # Path in the repository
  #      repo_id="jmcochrane/Sparse_KD_AE_Training_Data", # Repository name
  #      repo_type="dataset" # Type: dataset, model, or space
  #  )

def make_output_paths(config: CacheConfig, split_name: str) -> Dict[str, str]:
    out = {}
    # ADD A PATH FOR FULL LOGITS CACHE
    #==============================================================================================
    if config.mode == "full_logits":
        out["full_logits"] = os.path.join(config.cache_dir, f"full_logits_{split_name}.pt")
    #==============================================================================================
    if config.mode in ("topk", "both"):
        out["topk"] = os.path.join(config.cache_dir, f"topk_{split_name}.pt")
    if config.mode in ("sampling", "both"):
        out["sampling"] = os.path.join(config.cache_dir, f"sampling_{split_name}.pt")
    return out


def save_shard(
    storage: Dict[str, Dict[str, list]],
    config: CacheConfig,
    split_name: str,
    shard_idx: int,
) -> None:
    
    # ADD A BRANCH FOR SAVING FULL LOGITS SHARD
    #==============================================================================================
    if "full_logits" in storage:
        full_logits_payload = concat_storage(storage["full_logits"])
#        full_logits_payload["meta"] = {
#            "split": split_name,
#            "cache_type": "full_logits",
#            "temperature": config.temperature,
#            "seq_len": config.seq_len,
#        }
        save_payload(
            os.path.join(config.cache_dir, f"full_logits_{split_name}_shard{shard_idx:04d}.pt"),
            full_logits_payload,
        )
    #==============================================================================================    
    elif "topk" in storage:
        topk_payload = concat_storage(storage["topk"])
        topk_payload["meta"] = {
            "split": split_name,
            "cache_type": "topk",
            "k": config.topk_k,
            "seq_len": config.seq_len,
            "temperature": config.temperature,
        }
        save_payload(
            os.path.join(config.cache_dir, f"topk_{split_name}_shard{shard_idx:04d}.pt"),
            topk_payload,
        )

    elif "sampling" in storage:
        sampling_payload = concat_storage(storage["sampling"])
        sampling_payload["meta"] = {
            "split": split_name,
            "cache_type": "sampling",
            "num_draws": config.sampling_num_draws,
            "temperature": config.temperature,
            "seq_len": config.seq_len,
        }
        save_payload(
            os.path.join(config.cache_dir, f"sampling_{split_name}_shard{shard_idx:04d}.pt"),
            sampling_payload,
        )
    else:
        raise ValueError("Storage must contain either 'full_logits', 'topk', or 'sampling' key.")

def cache_split(
    teacher: torch.nn.Module,
    dataloader,
    split_name: str,
    config: CacheConfig,
) -> None:
    """
    Cache teacher outputs for a data split.

    For sampling mode with large num_draws (e.g. 40, 50), accumulating ALL
    batches in RAM before saving causes OOM (the sampled_ids tensor alone
    can exceed 20 GB for 200k samples at k=50). We therefore always use
    sharded saves for the sampling cache when num_draws >= 16, regardless of
    save_per_split_single_file, then concatenate shards into one file at the
    end. Top-K still respects save_per_split_single_file as before.
    """
    teacher.eval()
    device = get_model_device(teacher)
    ensure_dir(config.cache_dir)

    print(f"\nCaching split: {split_name}")
    print(f"Mode: {config.mode}")

    # For large sampling budgets, always shard to avoid OOM
    force_shard_sampling = (
        config.mode in ("sampling", "both", "full_logits")  # full_logits can also be large if seq_len and vocab are large
        and config.sampling_num_draws >= 16
    ) # --> True
    
    # TURN ON FORCED SHARDING 
    #==============================================================================================
    assert force_shard_sampling == True, "Sharding is needed for full_logits mode"
    #==============================================================================================
    storage = init_storage(config.mode)
    shard_idx = 0
    batch_counter = 0
    sampling_shard_paths: list = []  # track shard paths for later merge

    for batch in tqdm(dataloader, desc=f"Caching {split_name}"):
        
        # ==============================================================================================
#        if batch_counter >= 250:
#            print(f"Reached batch limit for testing: {batch_counter} batches processed. Stopping early.")
#            break
        # =============================================================================================
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = teacher_forward(teacher, input_ids, attention_mask)
        
        # *** FULL LOGIT CACHE WOULD BE HERE  ***
        #==============================================================================================
        if config.mode == "full_logits":
            full_logits_out = build_full_logits_softlabels(
                logits=logits,
                k=config.topk_k,  # pass k for logging/debugging but it won't affect the loss since we return the full distribution
                temperature=config.temperature,
                probs_dtype=config.probs_dtype,
            )
            append_common_tensors(storage["full_logits"], input_ids, attention_mask, labels)
            storage["full_logits"]["probs"].append(full_logits_out["probs"])
            # we can optionally also store the top-k ids and probs for analysis
            storage["full_logits"]["topk_ids"].append(full_logits_out["topk_ids"])
            storage["full_logits"]["topk_probs"].append(full_logits_out["topk_probs"])
        #==============================================================================================

        if config.mode in ("topk", "both"):
            topk_out = build_topk_softlabels(
                logits=logits,
                k=config.topk_k,
                temperature=config.temperature,
                probs_dtype=config.probs_dtype,
            )
            append_common_tensors(storage["topk"], input_ids, attention_mask, labels)
            storage["topk"]["topk_ids"].append(topk_out["topk_ids"])
            storage["topk"]["topk_probs"].append(topk_out["topk_probs"])

        if config.mode in ("sampling", "both"):
            sampling_out = build_sampling_softlabels(
                logits=logits,
                num_draws=config.sampling_num_draws,
                temperature=config.temperature,
                probs_dtype=config.probs_dtype,
            )
            append_common_tensors(storage["sampling"], input_ids, attention_mask, labels)
            storage["sampling"]["sampled_ids"].append(sampling_out["sampled_ids"])
            storage["sampling"]["sampled_counts"].append(sampling_out["sampled_counts"])
            storage["sampling"]["sampled_probs"].append(sampling_out["sampled_probs"])

        batch_counter += 1

        should_shard = (
            (not config.save_per_split_single_file) 
            or force_shard_sampling                 # --> True
        ) and (batch_counter % config.shard_size_batches == 0)

        # ==============================================================================================
        #assert config.shard_size_batches == 10
        
        if batch_counter % config.shard_size_batches == 0:
            assert should_shard == True, "Sharding should be forced on for sampling mode with num_draws >= 16 to avoid OOM, and save_per_split_single_file is ignored in this case. Please check the logic for should_shard if you see this assertion error."
        # ==============================================================================================
        
        if should_shard:

            # For forced-shard sampling, only flush the sampling storage;
            # topk can still accumulate if save_per_split_single_file is set.
            if force_shard_sampling and "sampling" in storage:
                
                raise Exception("storage branch not used")
                
                shard_path = os.path.join(
                    config.cache_dir,
                    f"sampling_{split_name}_shard{shard_idx:04d}.pt",
                )
                sampling_payload = concat_storage(storage["sampling"])
                sampling_payload["meta"] = {
                    "split": split_name,
                    "cache_type": "sampling",
                    "num_draws": config.sampling_num_draws,
                    "temperature": config.temperature,
                    "seq_len": config.seq_len,
                }
                save_payload(shard_path, sampling_payload)
                sampling_shard_paths.append(shard_path)
                storage["sampling"] = init_storage("sampling")["sampling"]

            # MAKE A BRANCH FOR FULL LOGITS SHARDING IF MODE IS FULL_LOGITS
            # ==============================================================================================
            if force_shard_sampling and "full_logits" in storage:
                shard_path = os.path.join(
                    config.cache_dir,
                    f"full_logits_{split_name}_shard{shard_idx:04d}.pt",
                )
                full_logits_payload = concat_storage(storage["full_logits"])
#                full_logits_payload["meta"] = {
#                    "split": split_name,
#                    "cache_type": "full_logits",
#                    "temperature": config.temperature,
#                    "seq_len": config.seq_len,
#                }
                save_payload(shard_path, full_logits_payload)
                print(f"Flushing full logits shard {shard_idx} to {shard_path} with {full_logits_payload['input_ids'].shape[0]} samples...")
                sampling_shard_paths.append(shard_path)
                time.sleep(20)  # 20 second delay to ensure file is fully written before upload, since these shards can be very large and we want to avoid any risk of trying to upload an incomplete file
                # upload the full_logits_payload to HF hub immediately after saving each shard, since full logits shards can be very large and we want to avoid any risk of trying to upload an incomplete file if we wait until the end when we might have multiple shards ready to upload at once
                print("Authenticating with Hugging Face Hub for upload...")
                hf_api_key = os.getenv("HF_TOKEN")
                login(token=hf_api_key, add_to_git_credential=True)
                print(f"Uploading {shard_path} to Hugging Face Hub...")
                push_to_hf_hub(shard_path, full_logits_payload)
                print(f"Finished uploading {shard_path} to Hugging Face Hub.")
                # delete storage to free up RAM and re-init empty storage for next shard
                del storage["full_logits"]
            #    gc.collect()
                #storage = init_storage("full_logits")
                try:
                    storage["full_logits"] = init_storage("full_logits")["full_logits"]
                except Exception as e:
                    print(f"Error initializing empty full_logits storage: {e}")
            # ==============================================================================================
            
            if not force_shard_sampling:
                
                raise Exception("storage branch not used")
                
                save_shard(storage, config, split_name, shard_idx)
                storage = init_storage(config.mode)

            shard_idx += 1

    # ── Flush any remaining batches ──────────────────────────────────────────

    # SAVE FULL LOGITS CACHE TO OUTPUT FILE
    #==============================================================================================
    if "full_logits" in storage and len(storage["full_logits"]["input_ids"]) > 0:
        if config.save_per_split_single_file:
            
            raise Exception("save_per_split_single_file should be false!")
            out_paths = make_output_paths(config, split_name)
            full_logits_payload = concat_storage(storage["full_logits"])
            full_logits_payload["meta"] = {
                "split": split_name,
                "cache_type": "full_logits",
                "temperature": config.temperature,
                "seq_len": config.seq_len,
            }
            save_payload(out_paths["full_logits"], full_logits_payload)
        else:
            save_shard(storage, config, split_name, shard_idx)
            # upload the final full logits shard to HF hub immediately after saving
            time.sleep(20)  # 20 second delay to ensure file is fully written before upload, since full logits shards can be very large and we want to avoid any risk of trying to upload an incomplete file
            shard_path = os.path.join(
                config.cache_dir,
                f"full_logits_{split_name}_shard{shard_idx:04d}.pt",
            )
            
            print("Authenticating with Hugging Face Hub for upload...")
            hf_api_key = os.getenv("HF_TOKEN")
            login(token=hf_api_key, add_to_git_credential=True)
            print(f"Uploading {shard_path} to Hugging Face Hub...")
            full_logits_payload = concat_storage(storage["full_logits"])
            push_to_hf_hub(shard_path, full_logits_payload)
            print(f"Finished uploading {shard_path} to Hugging Face Hub.")
    # ==============================================================================================
    
    # Remaining topk (always single-file or normal shard)
    elif "topk" in storage and len(storage["topk"]["input_ids"]) > 0:
        if config.save_per_split_single_file:
            out_paths = make_output_paths(config, split_name)
            topk_payload = concat_storage(storage["topk"])
            topk_payload["meta"] = {
                "split": split_name,
                "cache_type": "topk",
                "k": config.topk_k,
                "seq_len": config.seq_len,
                "temperature": config.temperature,
            }
            save_payload(out_paths["topk"], topk_payload)
        else:
            save_shard(storage, config, split_name, shard_idx)

    # Remaining sampling
    elif "sampling" in storage and len(storage["sampling"]["input_ids"]) > 0:
        if force_shard_sampling:
            shard_path = os.path.join(
                config.cache_dir,
                f"sampling_{split_name}_shard{shard_idx:04d}.pt",
            )
            sampling_payload = concat_storage(storage["sampling"])
            sampling_payload["meta"] = {
                "split": split_name,
                "cache_type": "sampling",
                "num_draws": config.sampling_num_draws,
                "temperature": config.temperature,
                "seq_len": config.seq_len,
            }
            save_payload(shard_path, sampling_payload)
            sampling_shard_paths.append(shard_path)
        elif config.save_per_split_single_file:
            out_paths = make_output_paths(config, split_name)
            sampling_payload = concat_storage(storage["sampling"])
            sampling_payload["meta"] = {
                "split": split_name,
                "cache_type": "sampling",
                "num_draws": config.sampling_num_draws,
                "temperature": config.temperature,
                "seq_len": config.seq_len,
            }
            save_payload(out_paths["sampling"], sampling_payload)
        else:
            save_shard(storage, config, split_name, shard_idx)
    else:
        raise ValueError("No data to save for sampling cache, but expected some based on batch processing.")
    
    # ── Shards are left in place; ShardedCachedDataset in data.py loads them lazily ──
    if force_shard_sampling and (sampling_shard_paths):
        print(
            f"Saved {len(sampling_shard_paths)} shards for {split_name} "
            f"in '{config.cache_dir}'. No merge needed — ShardedCachedDataset "
            f"will load them on demand during training."
        )




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
        shard_size_batches=100,
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
    cache_split(teacher, val_loader, "val", config)


if __name__ == "__main__":
    main()