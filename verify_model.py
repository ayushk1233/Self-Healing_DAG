from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def verify_model(model_dir="./model"):
    """
    Load the fine-tuned model and run a sample inference.
    """
    print(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print("✓ Model and tokenizer loaded successfully!\n")
    
    # Test samples
    test_sentences = [
        "This movie is absolutely fantastic and amazing!",
        "I hated this film, it was terrible and boring.",
        "The acting was okay but the plot was confusing."
    ]
    
    print("Running sample inferences:\n")
    print("="*60)
    
    for sentence in test_sentences:
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        label = "positive" if predicted_class == 1 else "negative"
        
        print(f"\nSentence: {sentence}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: [neg={probabilities[0][0]:.4f}, pos={probabilities[0][1]:.4f}]")
        print("-"*60)
    
    print("\n✓ Model verification complete!")

if __name__ == "__main__":
    verify_model()