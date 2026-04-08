from datasets import load_dataset
import torch
import os
import glob

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Maps a short dataset key -> (hf_repo, hf_config, text_column, train_split, val_split)
# hf_config can be None if the dataset has no config.
# val_split can be None — the loader will carve val from the tail of train.
_DATASET_REGISTRY = {
    # tuple: (hf_repo, hf_config, text_col, train_split, val_split)

    # WikiText-103
    "wikitext": (
        "wikitext", "wikitext-103-raw-v1", "text", "train", "validation"
    ),
    "wikitext-103-raw-v1": (
        "wikitext", "wikitext-103-raw-v1", "text", "train", "validation"
    ),
    # codeparrot/codeparrot-clean — Python code (~5M files, ~13GB).
    # Downloads fully on first use, then cached by HuggingFace.
    "github-code": (
        "codeparrot/codeparrot-clean", None, "content", "train", None
    ),
    "github-code-python": (
        "codeparrot/codeparrot-clean", None, "content", "train", None
    ),
    # PubMed abstracts
    "pubmed": (
        "scientific_papers", "pubmed", "abstract", "train", "validation"
    ),
}


def _resolve_dataset(name_or_key: str):
    """Return (hf_repo, hf_config, text_col, train_split, val_split)."""
    if name_or_key not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset key '{name_or_key}'. "
            f"Known keys: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name_or_key]


def get_dataloaders(
    tokenizer,
    seq_len: int = 256,
    batch_size: int = 4,
    num_train_samples: int = 10000,
    num_val_samples: int = 1000,
    train_dataset_name: str = "wikitext",
    train_dataset_config: str = None,   # ignored when using registry key
    val_dataset_name: str = None,       # defaults to same as train
    val_dataset_config: str = None,     # ignored when using registry key
):
    """Return (train_dataloader, val_dataloader).

    Supported dataset keys:
        'wikitext' / 'wikitext-103-raw-v1'  — WikiText-103 (default)
        'github-code'                        — GitHub Code, all languages
        'github-code-python'                 — GitHub Code, Python only
        'pubmed'                             — PubMed abstracts

    All datasets are downloaded fully on first use and cached by HuggingFace.
    Subsequent runs load from the local cache instantly.
    """
    # ---- resolve train dataset ----
    train_key = train_dataset_config if train_dataset_config and train_dataset_config in _DATASET_REGISTRY else train_dataset_name
    hf_repo, hf_config, text_col, train_split_name, val_split_name = _resolve_dataset(train_key)

    # ---- resolve val dataset (default: same as train) ----
    if val_dataset_name is None and val_dataset_config is None:
        val_key = train_key
    else:
        val_key = val_dataset_config if val_dataset_config and val_dataset_config in _DATASET_REGISTRY else (val_dataset_name or train_key)
    val_hf_repo, val_hf_config, val_text_col, _, val_split_name_resolved = _resolve_dataset(val_key)

    if val_split_name_resolved is not None:
        # Dedicated val split exists — load train and val independently.
        print(f"[data] Loading train split ({num_train_samples} samples)...")
        train_dataset = load_dataset(hf_repo, hf_config, split=f"{train_split_name}[:{num_train_samples}]", trust_remote_code=True)
        print(f"[data] Loading val split ({num_val_samples} samples)...")
        val_dataset = load_dataset(val_hf_repo, val_hf_config, split=f"{val_split_name_resolved}[:{num_val_samples}]", trust_remote_code=True)
    else:
        # No dedicated val split — carve non-overlapping slices from train.
        # val = first num_val_samples rows, train = next num_train_samples rows.
        print(f"[data] Loading full train split (will carve train + val)...")
        val_dataset = load_dataset(val_hf_repo, val_hf_config, split=f"train[:{num_val_samples}]", trust_remote_code=True)
        train_dataset = load_dataset(hf_repo, hf_config, split=f"train[{num_val_samples}:{num_val_samples + num_train_samples}]", trust_remote_code=True)
        print(f"[data] Carved {len(train_dataset)} train + {len(val_dataset)} val samples from train split.")

    # ---- tokenize & chunk ----
    def _make_tokenize_fn(col):
        def tokenize_and_chunk(batch):
            tokenized = tokenizer(batch[col])
            # sum(list, []) is O(N^2) in Python; use standard list flatten for O(N)
            concatenated = {
                k: [item for sublist in tokenized[k] for item in sublist]
                for k in tokenized.keys()
            }
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // seq_len) * seq_len
            result = {
                k: [v[i: i + seq_len] for i in range(0, total_length, seq_len)]
                for k, v in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        return tokenize_and_chunk

    num_workers = os.cpu_count() or 1
    print(f"[data] Mapping datasets with {num_workers} processes...")

    train_tokenized = train_dataset.map(
        _make_tokenize_fn(text_col), 
        batched=True, 
        remove_columns=train_dataset.column_names,
        num_proc=num_workers
    )
    train_tokenized.set_format("torch")

    val_tokenized = val_dataset.map(
        _make_tokenize_fn(val_text_col), 
        batched=True, 
        remove_columns=val_dataset.column_names,
        num_proc=num_workers
    )
    val_tokenized.set_format("torch")

    train_dataloader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size)

    return train_dataloader, val_dataloader

class CachedDataset(torch.utils.data.Dataset):
    """Loads a single pre-merged .pt file into memory."""
    def __init__(self, data_dict):
        self.data = {k: v for k, v in data_dict.items() if k != "meta"}
        self.meta = data_dict.get("meta", {})
        self.length = len(self.data["input_ids"])
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class ShardedCachedDataset(torch.utils.data.Dataset):
    """
    Lazily loads sharded cache files one shard at a time.

    Shards are expected to be named like:
        <cache_fmt>_<split>_shard0000.pt
        <cache_fmt>_<split>_shard0001.pt
        ...

    Only one shard is kept in RAM at a time, so memory usage is bounded
    by the size of a single shard regardless of K or dataset size.
    """

    def __init__(self, shard_paths: list, shuffle_within_shard: bool = True, seed: int = 42):
        if not shard_paths:
            raise ValueError("shard_paths must be non-empty")

        self.shard_paths = sorted(shard_paths)
        self._shuffle_within_shard = shuffle_within_shard
        self._seed = seed

        # Build a cumulative index so we can map global idx -> (shard, local_idx)
        self.shard_lengths = []
        for path in self.shard_paths:
            shard = torch.load(path, weights_only=False)
            self.shard_lengths.append(len(shard["input_ids"]))

        self.cumulative = []
        total = 0
        for ln in self.shard_lengths:
            total += ln
            self.cumulative.append(total)
        self.total_length = total

        # Lazy-load cache
        self._current_shard_idx = -1
        self._current_shard_data = None
        self._shard_perm = None  # optional intra-shard index permutation

    def __len__(self):
        return self.total_length

    def _load_shard(self, shard_idx: int):
        if self._current_shard_idx != shard_idx:
            shard = torch.load(self.shard_paths[shard_idx], weights_only=False)
            self._current_shard_data = {k: v for k, v in shard.items() if k != "meta"}
            self._current_shard_idx = shard_idx
            if self._shuffle_within_shard:
                n = self.shard_lengths[shard_idx]
                g = torch.Generator().manual_seed(self._seed + shard_idx)
                self._shard_perm = torch.randperm(n, generator=g)
            else:
                self._shard_perm = None

    def __getitem__(self, idx):
        # Binary-search to find which shard owns this global index
        lo, hi = 0, len(self.cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo
        local_idx = idx if shard_idx == 0 else idx - self.cumulative[shard_idx - 1]

        self._load_shard(shard_idx)
        if self._shard_perm is not None:
            local_idx = self._shard_perm[local_idx].item()
        return {k: v[local_idx] for k, v in self._current_shard_data.items()}


def get_cached_dataloaders(cache_fmt="topk", cache_dir="teacher_cache", batch_size=4):
    """
    Auto-detects whether the cache is sharded or a single merged file.

    Sharded layout (preferred for large K)::

        <cache_dir>/<cache_fmt>_train_shard0000.pt
        <cache_dir>/<cache_fmt>_train_shard0001.pt
        ...

    Single-file layout::

        <cache_dir>/<cache_fmt>_train.pt
        <cache_dir>/<cache_fmt>_val.pt
    """
    def _make_dataset(split):
        # Check for shards first
        pattern = os.path.join(cache_dir, f"{cache_fmt}_{split}_shard*.pt")
        shard_paths = sorted(glob.glob(pattern))
        if shard_paths:
            # ==============================================================
            raise NotImplementedError("Sharded cache loading is not yet implemented in this snippet. Detected shards:\n" + "\n".join(shard_paths))
            # ==============================================================
            print(f"[data] Found {len(shard_paths)} shards for {cache_fmt}/{split} — using ShardedCachedDataset")
            return ShardedCachedDataset(shard_paths)

        # Fall back to single merged file
        single_path = os.path.join(cache_dir, f"{cache_fmt}_{split}.pt")
        if os.path.exists(single_path):
            print(f"[data] Loading single-file cache: {single_path}")
            data = torch.load(single_path, weights_only=False)
            return CachedDataset(data)

        raise FileNotFoundError(
            f"No cache found for fmt='{cache_fmt}' split='{split}' in '{cache_dir}'.\n"
            f"  Looked for shards : {pattern}\n"
            f"  Looked for single : {single_path}"
        )

    train_dataset = _make_dataset("train")
    val_dataset   = _make_dataset("val")

    # NOTE: shuffle=False for ShardedCachedDataset — shuffling across shards
    # causes thrashing (each random access potentially loads a different shard).
    # For training, consider setting shuffle=False and shuffling at the shard level
    # during caching, or accept the sequential order.
    shuffle_train = isinstance(train_dataset, CachedDataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0
    )

    return train_loader, val_loader
