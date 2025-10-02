from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentPredictor:
    """Wrapper for fine-tuned sentiment classification model."""
    
    def __init__(self, model_path="./model"):
        """
        Initialize the predictor with a fine-tuned model.
        
        Args:
            model_path: Path to the saved model directory
        """
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        print("✓ Model loaded successfully")
    
    def predict(self, text):
        """
        Predict sentiment for a given text.
        
        Args:
            text: Input text string
            
        Returns:
            dict: {
                'label': str ('positive' or 'negative'),
                'confidence': float (0-1),
                'probabilities': dict {'negative': float, 'positive': float}
            }
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Extract prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        label = "positive" if predicted_class == 1 else "negative"
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'positive': probabilities[0][1].item()
            }
        }

# Convenience function
def predict(text, model_path="./model"):
    """
    Simple function to predict sentiment.
    
    Args:
        text: Input text string
        model_path: Path to saved model
        
    Returns:
        dict: Prediction results with label and confidence
    """
    predictor = SentimentPredictor(model_path)
    return predictor.predict(text)

# Test the module
if __name__ == "__main__":
    predictor = SentimentPredictor()
    
    test_cases = [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "It's okay, nothing special."
    ]
    
    print("\n" + "="*60)
    print("Testing inference module:")
    print("="*60)
    
    for text in test_cases:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: neg={result['probabilities']['negative']:.4f}, "
              f"pos={result['probabilities']['positive']:.4f}")