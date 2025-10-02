from datasets import load_dataset
from transformers import AutoTokenizer
import os

def preprocess_data(model_name="distilbert-base-uncased", max_length=128, cache_dir="./data"):
    """
    Load SST-2 dataset, tokenize, and cache for training.
    
    Args:
        model_name: Pretrained model name for tokenizer
        max_length: Maximum sequence length
        cache_dir: Directory to cache processed data
    
    Returns:
        tokenized_datasets: Tokenized train and validation splits
        tokenizer: The tokenizer instance
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")
    
    def tokenize_function(examples):
        """Tokenize the sentences."""
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing"
    )
    
    # Remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "tokenized_sst2")
    print(f"Saving tokenized dataset to {cache_path}")
    tokenized_datasets.save_to_disk(cache_path)
    
    print(f"✓ Preprocessing complete!")
    print(f"  Train samples: {len(tokenized_datasets['train'])}")
    print(f"  Validation samples: {len(tokenized_datasets['validation'])}")
    
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    tokenized_datasets, tokenizer = preprocess_data()
    print("\nSample tokenized example:")
    print(tokenized_datasets['train'][0])