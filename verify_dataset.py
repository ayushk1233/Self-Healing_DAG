from datasets import load_dataset

# Load SST-2 dataset
print("Loading SST-2 dataset...")
ds = load_dataset("stanfordnlp/sst2")

# Print dataset structure
print("\nDataset structure:")
print(ds)

# Inspect sample from train split
print("\n=== Sample from training set ===")
print(f"Number of training examples: {len(ds['train'])}")
print(f"\nFirst 3 examples:")
for i in range(3):
    example = ds['train'][i]
    print(f"\nExample {i+1}:")
    print(f"  Sentence: {example['sentence']}")
    print(f"  Label: {example['label']} ({'positive' if example['label'] == 1 else 'negative'})")

# Inspect validation split
print(f"\n=== Validation set ===")
print(f"Number of validation examples: {len(ds['validation'])}")
print(f"\nFirst example from validation:")
val_example = ds['validation'][0]
print(f"  Sentence: {val_example['sentence']}")
print(f"  Label: {val_example['label']}")

print("\n✓ Dataset loaded successfully!")