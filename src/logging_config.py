import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path

def setup_logging(log_file="logs/pipeline.log", level=logging.INFO):
    """
    Configure structured logging for the pipeline.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - simple logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup
    # Log startup
    logging.info("="*80)
    logging.info("Logging system initialized")


class PipelineLogger:
    """Structured logging for the sentiment analysis pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type, **data):
        """Log a structured event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_prediction(self, text, label, confidence, probabilities):
        """Log model prediction."""
        self.log_event(
            "PREDICTION",
            text=text[:200],  # Truncate long texts
            predicted_label=label,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def log_confidence_check(self, confidence, threshold, needs_fallback):
        """Log confidence check results."""
        self.log_event(
            "CONFIDENCE_CHECK",
            confidence=confidence,
            threshold=threshold,
            needs_fallback=needs_fallback
        )
    
    def log_fallback(self, reason, original_prediction, confidence):
        """Log fallback trigger."""
        self.log_event(
            "FALLBACK_TRIGGERED",
            reason=reason,
            original_prediction=original_prediction,
            confidence=confidence
        )
    
    def log_user_override(self, original_label, new_label, confidence):
        """Log user feedback."""
        self.log_event(
            "USER_OVERRIDE",
            original_label=original_label,
            new_label=new_label,
            confidence=confidence
        )
    
    def log_final_decision(self, text, final_label, fallback_triggered, user_override):
        """Log final classification decision."""
        self.log_event(
            "FINAL_DECISION",
            text=text[:200],  # Truncate long texts
            final_label=final_label,
            fallback_triggered=fallback_triggered,
            user_override=user_override
        )
    # Startup message
    logging.info("Logging initialized")
class PipelineLogger:
    """Structured logger for pipeline events."""
    
    def __init__(self, name="pipeline"):
        """Initialize pipeline logger."""
        self.logger = logging.getLogger(name)
    
    def log_prediction(self, text, prediction, confidence, probabilities):
        """
        Log a prediction event.
        
        Args:
            text: Input text
            prediction: Predicted label
            confidence: Confidence score
            probabilities: Probability distribution
        """
        self.logger.info(
            f"PREDICTION | Text: '{text[:50]}...' | "
            f"Label: {prediction} | Confidence: {confidence:.4f} | "
            f"Probs: [neg={probabilities['negative']:.4f}, pos={probabilities['positive']:.4f}]"
        )
    
    def log_confidence_check(self, confidence, threshold, needs_fallback):
        """
        Log a confidence check event.
        
        Args:
            confidence: Confidence score
            threshold: Threshold value
            needs_fallback: Whether fallback is needed
        """
        status = "FALLBACK_NEEDED" if needs_fallback else "ACCEPTED"
        self.logger.info(
            f"CONFIDENCE_CHECK | Confidence: {confidence:.4f} | "
            f"Threshold: {threshold:.4f} | Status: {status}"
        )
    
    def log_fallback(self, trigger_reason, original_prediction, confidence):
        """
        Log a fallback trigger event.
        
        Args:
            trigger_reason: Reason for fallback
            original_prediction: Original model prediction
            confidence: Confidence score
        """
        self.logger.warning(
            f"FALLBACK_TRIGGERED | Reason: {trigger_reason} | "
            f"Original: {original_prediction} | Confidence: {confidence:.4f}"
        )
    
    def log_user_override(self, original_prediction, user_label, confidence):
        """
        Log a user override event.
        
        Args:
            original_prediction: Model's prediction
            user_label: User-provided label
            confidence: Original confidence
        """
        self.logger.info(
            f"USER_OVERRIDE | Original: {original_prediction} | "
            f"User_Label: {user_label} | Original_Confidence: {confidence:.4f}"
        )
    
    def log_final_decision(self, text, final_label, had_fallback, user_override=False):
        """
        Log the final decision event.
        
        Args:
            text: Input text
            final_label: Final classification label
            had_fallback: Whether fallback was triggered
            user_override: Whether user override was applied
        """
        source = "USER_OVERRIDE" if user_override else ("MODEL" if not had_fallback else "FALLBACK")
        self.logger.info(
            f"FINAL_DECISION | Text: '{text[:50]}...' | "
            f"Label: {final_label} | Source: {source}"
        )
    
    def log_error(self, error_msg, exception=None):
        """
        Log an error event.
        
        Args:
            error_msg: Error message
            exception: Exception object (optional)
        """
        if exception:
            self.logger.error(f"ERROR | {error_msg}", exc_info=exception)
        else:
            self.logger.error(f"ERROR | {error_msg}")