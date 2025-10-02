import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import SentimentPredictor
import logging

# Setup logger
logger = logging.getLogger(__name__)

class InferenceNode:
    """Node that performs sentiment inference using the fine-tuned model."""
    
    def __init__(self, model_path="./model"):
        """
        Initialize the inference node.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.predictor = SentimentPredictor(model_path)
        logger.info("InferenceNode initialized")
    
    def run(self, state):
        """
        Run inference on the input text.
        
        Args:
            state: dict with 'text' key
            
        Returns:
            dict: Updated state with prediction results
        """
        text = state.get('text', '')
        logger.info(f"InferenceNode processing: '{text[:50]}...'")
        
        prediction = self.predictor.predict(text)
        
        state['prediction'] = prediction['label']
        state['confidence'] = prediction['confidence']
        state['probabilities'] = prediction['probabilities']
        
        logger.info(f"Prediction: {prediction['label']}, Confidence: {prediction['confidence']:.4f}")
        
        return state


class ConfidenceCheckNode:
    """Node that checks if prediction confidence meets threshold."""
    
    def __init__(self, threshold=0.85):
        """
        Initialize the confidence check node.
        
        Args:
            threshold: Minimum confidence threshold (0-1)
        """
        self.threshold = threshold
        logger.info(f"ConfidenceCheckNode initialized with threshold={threshold}")
    
    def run(self, state):
        """
        Check if confidence meets threshold.
        
        Args:
            state: dict with 'confidence' key
            
        Returns:
            dict: Updated state with 'needs_fallback' flag
        """
        confidence = state.get('confidence', 0.0)
        needs_fallback = confidence < self.threshold
        
        state['needs_fallback'] = needs_fallback
        state['threshold'] = self.threshold
        
        if needs_fallback:
            logger.warning(f"Low confidence detected: {confidence:.4f} < {self.threshold}")
        else:
            logger.info(f"Confidence sufficient: {confidence:.4f} >= {self.threshold}")
        
        return state


class FallbackNode:
    """Node that handles low-confidence predictions via user input or backup model."""
    
    def __init__(self, mode="user", backup_model=None):
        """
        Initialize the fallback node.
        
        Args:
            mode: 'user' for user input, 'backup' for backup model
            backup_model: BackupClassifier instance (required if mode='backup')
        """
        self.mode = mode
        self.backup_model = backup_model
        logger.info(f"FallbackNode initialized with mode={mode}")
        
        if mode == "backup" and backup_model is None:
            logger.warning("Backup mode selected but no backup_model provided")
    
    def run(self, state):
        """
        Handle fallback for low-confidence predictions.
        
        Args:
            state: dict with prediction info
            
        Returns:
            dict: Updated state with fallback decision
        """
        if not state.get('needs_fallback', False):
            logger.info("No fallback needed, using original prediction")
            state['final_label'] = state['prediction']
            state['fallback_triggered'] = False
            return state
        
        logger.info("Fallback triggered")
        state['fallback_triggered'] = True
        
        if self.mode == "user":
            # Request user input (will be handled by CLI)
            state['awaiting_user_input'] = True
            logger.info("Awaiting user input for fallback decision")
        elif self.mode == "backup":
            # Use backup model
            if self.backup_model:
                logger.info("Using backup zero-shot model")
                backup_result = self.backup_model.predict(state['text'])
                state['backup_prediction'] = backup_result['label']
                state['backup_confidence'] = backup_result['confidence']
                state['backup_probabilities'] = backup_result['probabilities']
                state['final_label'] = backup_result['label']
                logger.info(f"Backup model prediction: {backup_result['label']} "
                          f"(confidence: {backup_result['confidence']:.4f})")
            else:
                logger.warning("Backup model not available, using original prediction")
                state['final_label'] = state['prediction']
        
        return state
    
    def apply_user_decision(self, state, user_label):
        """
        Apply user's fallback decision.
        
        Args:
            state: Current state dict
            user_label: User-provided label ('positive' or 'negative')
            
        Returns:
            dict: Updated state with final label
        """
        state['final_label'] = user_label
        state['user_override'] = True
        state['awaiting_user_input'] = False
        logger.info(f"User override applied: {user_label}")
        return state


# Helper function to create default nodes
def create_nodes(model_path="./model", confidence_threshold=0.85, fallback_mode="user", backup_model=None):
    """
    Create and return all pipeline nodes.
    
    Args:
        model_path: Path to fine-tuned model
        confidence_threshold: Confidence threshold for fallback
        fallback_mode: 'user' or 'backup'
        backup_model: BackupClassifier instance (if using backup mode)
        
    Returns:
        dict: Dictionary of initialized nodes
    """
    return {
        'inference': InferenceNode(model_path),
        'confidence_check': ConfidenceCheckNode(confidence_threshold),
        'fallback': FallbackNode(mode=fallback_mode, backup_model=backup_model)
    }