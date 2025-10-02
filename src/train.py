import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(
    model_name="distilbert-base-uncased",
    data_dir="./data/tokenized_sst2",
    output_dir="./model",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    """
    Fine-tune DistilBERT on SST-2 dataset.
    """
    print(f"Loading tokenized dataset from {data_dir}")
    tokenized_datasets = load_from_disk(data_dir)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    # Train
    trainer.train()
    
    # Save final model and tokenizer
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print("\nFinal Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✓ Training complete! Model saved to {output_dir}")
    
    return trainer, eval_results

if __name__ == "__main__":
    trainer, results = train_model()