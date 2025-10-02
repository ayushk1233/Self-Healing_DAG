# Self-Healing Classification DAG with Fine-Tuned Model

A self-healing sentiment ## Fallback Mechanisms

The system provides two fallback mechanisms for handling low-confidence predictions:

### 1. User-in-the-Loop## Model Details

### Primary Model

- **Base Model**: `distilbert-base-uncased`
- **Dataset**: Stanford SST-2 (67k train, 872 validation)
- **Task**: Binary sentiment classification (positive/negative)
- **Training Configuration**:
  - Epochs: 3
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Expected Accuracy: ~90%+

### Backup Model

- **Base Model**: `facebook/bart-large-mnli`
- **Type**: Zero-shot classifier
- **Task**: Natural Language Inference
- **Features**:
  - No task-specific training required
  - Flexible classification schema
  - Independent sentiment assessment
  - Complementary to primary model
    When the model's confidence falls below the threshold, the system prompts for user feedback:
- Shows the current prediction and confidence score
- Allows users to provide the correct sentiment
- Logs the override for future analysis

### 2. Zero-Shot Backup Model

Alternatively, you can enable the automated backup classifier:

- Uses BART-large-MNLI for zero-shot classification
- Provides independent sentiment assessment
- Automatically resolves low-confidence cases
- Useful for automated/batch processing

To enable the backup model:

```bash
python src/main.py [threshold] --backup
```

## Usage

### Running the Interactive CLI

```bash
# Run with default mode (user feedback) and threshold (0.85)
python src/main.py

# Run with custom threshold
python src/main.py 0.90

# Run with backup model
python src/main.py 0.90 --backuption pipeline that uses a fine-tuned DistilBERT model with confidence-based fallback mechanism. When predictions have low confidence, the system triggers a fallback to request user feedback, creating a robust and adaptive classification system.

## 🌟 Key Features

- ✨ **Self-Healing Pipeline**: Automatically detects and handles low-confidence predictions
- 🤝 **Dual Fallback System**: Choose between user feedback or zero-shot backup model
- 🤖 **Zero-Shot Backup**: BART-based model for automated fallback decisions
- 📊 **Structured Logging**: Comprehensive event tracking and monitoring
- 💻 **Interactive CLI**: User-friendly command-line interface
- 🔧 **Modular Design**: Easily extensible node-based architecture
- 🎯 **High Accuracy**: ~90%+ accuracy on SST-2 dataset

## 🏗️ Project Structure

```

atg-self-healing-dag/
├── src/
│ ├── preprocess.py # Dataset preprocessing and tokenization
│ ├── train.py # Model training script
│ ├── inference.py # Inference wrapper for predictions
│ ├── nodes.py # DAG nodes (Inference, ConfidenceCheck, Fallback)
│ ├── dag.py # Pipeline orchestration
│ ├── main.py # Interactive CLI application
│ ├── backup.py # Zero-shot backup classifier
│ └── logging_config.py # Structured logging configuration
├── data/
│ └── tokenized_sst2/ # Cached preprocessed dataset
├── model/ # Fine-tuned model artifacts
│ ├── config.json
│ ├── model.safetensors
│ ├── tokenizer_config.json
│ └── vocab.txt
├── logs/
│ └── pipeline.log # Structured pipeline logs
├── requirements.txt # Python dependencies
├── verify_dataset.py # Dataset verification script
├── verify_model.py # Model verification script
└── README.md # Documentation

````

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 2GB+ disk space for model and dataset
- CUDA-compatible GPU (optional, for faster training)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

### 2. Dataset Preparation

```bash
# Preprocess and cache the SST-2 dataset
python src/preprocess.py

# Verify dataset
python verify_dataset.py
```

### 3. Model Training

```bash
# Train the model (requires ~10-20 minutes)
python src/train.py

# Verify model performance
python verify_model.py
```

## 🎮 Usage

### Running the Interactive CLI

```bash
# Run with default confidence threshold (0.85)
python src/main.py

# Run with custom threshold
python src/main.py 0.90
```

### Example Session

```
Enter text (or command): This movie is absolutely fantastic!
------------------------------------------------------------
🤖 Prediction: positive
📊 Confidence: 0.9993
   Probabilities: neg=0.0007, pos=0.9993

✅ Final Decision: POSITIVE
============================================================

Enter text (or command): It was okay, nothing special.
------------------------------------------------------------
🤖 Prediction: negative
📊 Confidence: 0.8234
   Probabilities: neg=0.8234, pos=0.1766

⚠️  Low confidence detected (0.8234 < 0.9000)
    Fallback mechanism activated.

🔄 Please provide the correct sentiment:
   Enter 'positive' or 'negative' (or 'p'/'n'): p

✅ Final Decision: POSITIVE
   (User override applied)
```

### CLI Commands

- Enter any text to get sentiment prediction
- `help` - Show help information
- `quit` or `exit` - Exit the application

## 🔄 Pipeline Architecture

## Pipeline Architecture

### DAG Flow

````
Input Text
    ↓
[Inference Node]
    ↓
Model Prediction + Confidence Score
    ↓
[Confidence Check Node]
    ↓
Confidence >= Threshold?
    ↓                ↓
   YES              NO
    ↓                ↓
Accept          [Fallback Node]
    ↓                ↓
    ↓          Mode Selection
    ↓          ↙           ↘
    ↓     User Input    Backup Model
    ↓          ↓           ↓
    ↓     Manual       Zero-Shot
    ↓     Override     Prediction
    ↓          ↓           ↓
    └─────────┴───────────┘
              ↓
       Final Decision

### Node Descriptions

- **InferenceNode**: Runs sentiment prediction using fine-tuned DistilBERT
- **ConfidenceCheckNode**: Evaluates prediction confidence against threshold
- **FallbackNode**: Handles low-confidence cases via user feedback

## 📊 Logging

All pipeline events are logged to `logs/pipeline.log` with structured JSON format:

- `PREDICTION`: Model predictions with confidence scores
- `CONFIDENCE_CHECK`: Threshold comparisons
- `FALLBACK_TRIGGERED`: Low confidence warnings
- `USER_OVERRIDE`: User feedback events
- `FINAL_DECISION`: Final classification with source attribution

View logs:

```bash
cat logs/pipeline.log
````

## 🤖 Model Details

- **Base Model**: `distilbert-base-uncased`
- **Dataset**: Stanford SST-2 (67k train, 872 validation)
- **Task**: Binary sentiment classification (positive/negative)
- **Training Configuration**:
  - Epochs: 3
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Expected Accuracy: ~90%+

## 🔜 Future Enhancements

1. Backup zero-shot model for automated fallback
2. Fine-tuning on user feedback for continuous improvement
3. REST API endpoint for production deployment
4. Batch processing mode for multiple inputs
5. Confidence threshold optimization
6. Model performance monitoring and drift detection
7. Active learning integration for selective user feedback

## 📋 Requirements

- **Python**: 3.8+
- **Key Libraries**:
  - PyTorch 2.0+
  - Transformers 4.30+
  - Datasets
  - tqdm
  - numpy
  - pandas

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
