from datasets import load_dataset
import torch

def get_dataloaders(tokenizer, seq_len=256, batch_size=4, num_train_samples=10000, num_val_samples=1000):
    # We load a small subset for quick local testing and development
    train_dataset = load_dataset("openwebtext", trust_remote_code=True, split=f"train[:{num_train_samples}]")
    val_dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True, split=f"validation[:{num_val_samples}]")
    
    def tokenize_and_chunk(batch):
        tokenized = tokenizer(batch["text"])
        concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // seq_len) * seq_len
        result = {
            k: [v[i: i + seq_len] for i in range(0, total_length, seq_len)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    train_tokenized = train_dataset.map(tokenize_and_chunk, batched=True, remove_columns=train_dataset.column_names)
    train_tokenized.set_format("torch")
    
    val_tokenized = val_dataset.map(tokenize_and_chunk, batched=True, remove_columns=val_dataset.column_names)
    val_tokenized.set_format("torch")
    
    train_dataloader = torch.utils.data.DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size)
    
    return train_dataloader, val_dataloader

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data = {k: v for k, v in data_dict.items() if k != "meta"}
        self.meta = data_dict.get("meta", {})
        self.length = len(self.data["input_ids"])
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def get_cached_dataloaders(cache_fmt="topk", cache_dir="teacher_cache", batch_size=4):
    import os
    train_path = os.path.join(cache_dir, f"{cache_fmt}_train.pt")
    val_path = os.path.join(cache_dir, f"{cache_fmt}_val.pt")
    
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    
    train_dataset = CachedDataset(train_data)
    val_dataset = CachedDataset(val_data)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
