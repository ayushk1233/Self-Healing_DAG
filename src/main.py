import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.dag import SelfHealingDAG
from src.logging_config import setup_logging, PipelineLogger

# Setup structured logging
setup_logging()
logger = PipelineLogger()

class SentimentCLI:
    """Interactive CLI for sentiment classification with self-healing."""
    
    def __init__(self, model_path="./model", confidence_threshold=0.85,
                 fallback_mode="user", backup_model=None):
        """Initialize the CLI."""
        print("="*60)
        print("Self-Healing Sentiment Classification System")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Fallback Mode: {fallback_mode}")
        print("="*60 + "\n")
        
        self.dag = SelfHealingDAG(model_path, confidence_threshold, fallback_mode, backup_model)
        self.fallback_mode = fallback_mode
    
    def process_text(self, text):
        """Process a single text input through the pipeline."""
        print(f"\n📝 Input: {text}")
        print("-"*60)
        
        # Run pipeline
        state = self.dag.run_pipeline(text)
        
        # Log prediction
        logger.log_prediction(
            text, 
            state['prediction'], 
            state['confidence'],
            state['probabilities']
        )
        
        # Log confidence check
        logger.log_confidence_check(
            state['confidence'],
            state['threshold'],
            state['needs_fallback']
        )
        
        # Display results
        print(f"🤖 Prediction: {state['prediction']}")
        print(f"📊 Confidence: {state['confidence']:.4f}")
        print(f"   Probabilities: neg={state['probabilities']['negative']:.4f}, "
              f"pos={state['probabilities']['positive']:.4f}")
        
        # Handle fallback if needed
        if state.get('awaiting_user_input', False):
            logger.log_fallback(
                "Low confidence",
                state['prediction'],
                state['confidence']
            )
            
            print(f"\n⚠️  Low confidence detected ({state['confidence']:.4f} < {state['threshold']})")
            print("    Fallback mechanism activated.")
            
            final_label = self.get_user_feedback(state)
            state = self.dag.apply_user_feedback(state, final_label)
            
            logger.log_user_override(
                state['prediction'],
                final_label,
                state['confidence']
            )
        elif state.get('backup_prediction'):
            # Backup model was used
            logger.log_fallback(
                "Using backup model",
                state['prediction'],
                state['confidence']
            )
            print(f"\n🔄 Backup model used:")
            print(f"   Backup Prediction: {state['backup_prediction']}")
            print(f"   Backup Confidence: {state['backup_confidence']:.4f}")
            
            logger.log_final_decision(
                state['text'],
                state['backup_prediction'],
                True,
                False
            )
        
        # Log final decision
        logger.log_final_decision(
            text,
            state['final_label'],
            state['fallback_triggered'],
            state.get('user_override', False)
        )
        
        # Display final decision
        print(f"\n✅ Final Decision: {state['final_label'].upper()}")
        
        if state.get('user_override', False):
            print("   (User override applied)")
        
        print("="*60)
        
        return state
    
    def get_user_feedback(self, state):
        """
        Prompt user for feedback on low-confidence predictions.
        
        Args:
            state: Current pipeline state
            
        Returns:
            str: User-provided label ('positive' or 'negative')
        """
        print("\n🔄 Please provide the correct sentiment:")
        print(f"   Current prediction: {state['prediction']}")
        
        while True:
            user_input = input("   Enter 'positive' or 'negative' (or 'p'/'n'): ").strip().lower()
            
            if user_input in ['positive', 'p']:
                return 'positive'
            elif user_input in ['negative', 'n']:
                return 'negative'
            else:
                print("   ❌ Invalid input. Please enter 'positive', 'negative', 'p', or 'n'.")
    
    def run(self):
        """Run the interactive CLI loop."""
        print("Welcome! Enter text to classify sentiment.")
        print("Commands: 'quit' or 'exit' to stop, 'help' for help\n")
        
        while True:
            try:
                # Get user input
                user_input = input("Enter text (or command): ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    print("⚠️  Please enter some text.")
                    continue
                
                # Process the text
                self.process_text(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                self.logger.log_event("ERROR", error=str(e), traceback=str(sys.exc_info()))
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")
    
    def show_help(self):
        """Display help information."""
        print("\n" + "="*60)
        print("HELP - Self-Healing Sentiment Classification")
        print("="*60)
        print("• Enter any text to get sentiment prediction")
        print("• System predicts: POSITIVE or NEGATIVE")
        print("• If confidence is low, you'll be asked to provide feedback")
        print("• Commands:")
        print("    - 'quit' or 'exit': Exit the application")
        print("    - 'help': Show this help message")
        print("="*60 + "\n")


def main():
    """Main entry point for the CLI."""
    # Parse command-line arguments
    confidence_threshold = 0.85
    fallback_mode = "user"
    backup_model = None
    
    if len(sys.argv) > 1:
        try:
            confidence_threshold = float(sys.argv[1])
            print(f"Using custom confidence threshold: {confidence_threshold}")
        except ValueError:
            print(f"Invalid threshold value. Using default: {confidence_threshold}")
    
    if len(sys.argv) > 2 and sys.argv[2] == "--backup":
        print("\n🔄 Initializing backup zero-shot model...")
        from src.backup import BackupClassifier
        backup_model = BackupClassifier()
        fallback_mode = "backup"
        print("✓ Backup mode enabled\n")
    
    # Run CLI
    cli = SentimentCLI(
        confidence_threshold=confidence_threshold,
        fallback_mode=fallback_mode,
        backup_model=backup_model
    )
    cli.run()


if __name__ == "__main__":
    main()