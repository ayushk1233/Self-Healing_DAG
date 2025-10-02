import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nodes import create_nodes
import logging

# Setup logger
logger = logging.getLogger(__name__)

class SelfHealingDAG:
    """Self-healing sentiment classification pipeline with confidence-based fallback."""
    
    def __init__(self, model_path="./model", confidence_threshold=0.85, 
                 fallback_mode="user", backup_model=None):
        """
        Initialize the DAG pipeline.
        
        Args:
            model_path: Path to fine-tuned model
            confidence_threshold: Minimum confidence for accepting predictions
            fallback_mode: 'user' or 'backup'
            backup_model: BackupClassifier instance (if using backup mode)
        """
        logger.info("Initializing Self-Healing DAG")
        self.nodes = create_nodes(model_path, confidence_threshold, fallback_mode, backup_model)
        self.confidence_threshold = confidence_threshold
        self.fallback_mode = fallback_mode
        logger.info(f"DAG initialized with confidence threshold={confidence_threshold}, "
                   f"fallback_mode={fallback_mode}")
    
    def run_pipeline(self, text):
        """
        Run the complete pipeline on input text.
        
        Args:
            text: Input text for sentiment classification
            
        Returns:
            dict: Pipeline state with results and metadata
        """
        logger.info("="*60)
        logger.info(f"Starting pipeline for text: '{text[:50]}...'")
        
        # Initialize state
        state = {
            'text': text,
            'prediction': None,
            'confidence': None,
            'probabilities': None,
            'needs_fallback': False,
            'fallback_triggered': False,
            'awaiting_user_input': False,
            'final_label': None
        }
        
        # Step 1: Run inference
        logger.info("Step 1: Running InferenceNode")
        state = self.nodes['inference'].run(state)
        
        # Step 2: Check confidence
        logger.info("Step 2: Running ConfidenceCheckNode")
        state = self.nodes['confidence_check'].run(state)
        
        # Step 3: Handle fallback if needed
        logger.info("Step 3: Running FallbackNode")
        state = self.nodes['fallback'].run(state)
        
        # If no fallback needed, set final label
        if not state['fallback_triggered']:
            state['final_label'] = state['prediction']
        
        logger.info(f"Pipeline complete. Final label: {state.get('final_label', 'PENDING')}")
        logger.info("="*60)
        
        return state
    
    def apply_user_feedback(self, state, user_label):
        """
        Apply user feedback for fallback cases.
        
        Args:
            state: Current pipeline state
            user_label: User-provided label
            
        Returns:
            dict: Updated state with final decision
        """
        logger.info(f"Applying user feedback: {user_label}")
        state = self.nodes['fallback'].apply_user_decision(state, user_label)
        return state


# Convenience function
def run_pipeline(text, model_path="./model", confidence_threshold=0.85):
    """
    Run the self-healing pipeline on input text.
    
    Args:
        text: Input text for classification
        model_path: Path to fine-tuned model
        confidence_threshold: Confidence threshold for fallback
        
    Returns:
        dict: Pipeline results
    """
    dag = SelfHealingDAG(model_path, confidence_threshold)
    return dag.run_pipeline(text)


# Test the DAG
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    dag = SelfHealingDAG(confidence_threshold=0.90)
    
    test_cases = [
        "This movie is absolutely fantastic!",  # High confidence
        "It was okay, somewhat average.",       # Potentially low confidence
        "Terrible and awful experience!"        # High confidence
    ]
    
    print("\n" + "="*60)
    print("Testing Self-Healing DAG Pipeline")
    print("="*60 + "\n")
    
    for text in test_cases:
        result = dag.run_pipeline(text)
        print(f"\nInput: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Needs Fallback: {result['needs_fallback']}")
        print(f"Final Label: {result.get('final_label', 'PENDING')}")
        print("-"*60)