import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class BackupClassifier:
    """Zero-shot classification as backup for low-confidence predictions."""
    
    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initialize backup zero-shot classifier.
        
        Args:
            model_name: Zero-shot model name (default: BART MNLI)
        """
        logger.info(f"Loading backup model: {model_name}")
        print(f"Loading backup zero-shot model: {model_name}...")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1  # Use CPU
        )
        
        self.candidate_labels = ["negative", "positive"]
        logger.info("Backup model loaded successfully")
        print("✓ Backup model loaded")
    
    def predict(self, text):
        """
        Predict sentiment using zero-shot classification.
        
        Args:
            text: Input text
            
        Returns:
            dict: {
                'label': str ('positive' or 'negative'),
                'confidence': float,
                'probabilities': dict
            }
        """
        logger.info(f"Running backup model on: '{text[:50]}...'")
        
        # Run zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=self.candidate_labels,
            multi_label=False
        )
        
        # Extract results
        predicted_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # Build probability dict
        probabilities = {
            'negative': result['scores'][1] if result['labels'][0] == 'positive' else result['scores'][0],
            'positive': result['scores'][0] if result['labels'][0] == 'positive' else result['scores'][1]
        }
        
        logger.info(f"Backup prediction: {predicted_label}, confidence: {confidence:.4f}")
        
        return {
            'label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities
        }


# Test backup model
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    backup = BackupClassifier()
    
    test_cases = [
        "This movie is absolutely fantastic!",
        "I hated this film, it was terrible.",
        "It was okay, nothing special."
    ]
    
    print("\n" + "="*60)
    print("Testing Backup Zero-Shot Classifier")
    print("="*60)
    
    for text in test_cases:
        result = backup.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
        print("-"*60)