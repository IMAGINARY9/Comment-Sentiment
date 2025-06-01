# Comment Sentiment Analysis

A comprehensive sentiment analysis system designed for social media comments and informal text, featuring advanced preprocessing, multiple model architectures, and ensemble methods for robust sentiment classification.

## Overview

- **Task**: Multi-class sentiment classification (Negative, Neutral, Positive) for social media content
- **Focus**: Handling informal language, emojis, hashtags, mentions, slang, and abbreviations
- **Approach**: Transformer models (Twitter RoBERTa) + BiLSTM with attention + Lexicon-based methods + Ensemble strategies
- **Performance**: Achieving 97.6% accuracy on Twitter sentiment datasets

## Project Architecture

```
Comment-Sentiment/
├── data/                           # Datasets and preprocessing pipeline
│   ├── airline_sentiment/          # US Airline Twitter Sentiment (14.6K samples)
│   ├── apple_sentiment/           # Apple Twitter Sentiment (1.6K samples)
│   ├── social_media_sentiment/    # Reddit (36.8K) + Twitter Social (163K) 
│   ├── twitter_sentiment/         # Twitter Combined Dataset (70K samples)
│   └── preprocessed/              # Cleaned datasets ready for training
├── src/                           # Core implementation
│   ├── models.py                  # Neural architectures (RoBERTa, BiLSTM, Ensemble)
│   ├── preprocessing.py           # Text cleaning & social media preprocessing
│   ├── training.py               # Training loops and optimization
│   ├── evaluation.py             # Comprehensive evaluation & visualization
│   ├── data_utils.py             # Data loading and configuration management
│   └── bilstm_trainer.py         # Specialized BiLSTM training logic
├── configs/                       # YAML configurations for different models
│   ├── twitter_roberta.yaml      # Twitter RoBERTa fine-tuning config
│   ├── bilstm_glove.yaml         # BiLSTM with GloVe embeddings
│   ├── distilbert.yaml           # DistilBERT configuration
│   └── ensemble.yaml             # Ensemble model configuration
├── scripts/                       # Training and evaluation scripts
│   ├── train.py                  # Main training script with logging
│   ├── evaluate.py               # Model evaluation with visualizations
│   └── predict.py                # Inference on new text samples
├── notebooks/                     # Analysis and experimentation
│   ├── exploration.ipynb         # Dataset exploration and insights
│   └── cleaning&preprocessing.ipynb  # Data preprocessing pipeline
├── models/                        # Trained model checkpoints
├── reports/                       # Evaluation results and visualizations
├── tests/                         # Unit tests for all components
└── references/                    # Reference implementations and papers
```

## Key Features & Capabilities

### Advanced Text Preprocessing
- **Social Media Specific**: Handles @mentions, #hashtags, URLs, emojis
- **Text Normalization**: Contractions expansion, spelling correction, case normalization
- **Emoji Processing**: Convert to text descriptions or remove/keep as configured
- **Multilingual Support**: Unicode handling and emoji standardization

### Multiple Model Architectures
- **Transformer Models**: Fine-tuned Twitter RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **BiLSTM with Attention**: Bidirectional LSTM with GloVe embeddings and attention mechanism
- **Lexicon-Based**: VADER Sentiment and TextBlob integration
- **Ensemble Methods**: Weighted combination of neural and lexicon approaches

### Comprehensive Evaluation & Analysis
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrices, ROC curves, confidence distributions
- **Error Analysis**: Length-based analysis, misclassification patterns
- **Social Media Analytics**: Emoji sentiment patterns, hashtag analysis, text feature correlation

### Production-Ready Features
- **Configurable Pipeline**: YAML-based configuration for different use cases
- **Batch Processing**: Efficient inference on large datasets
- **Model Persistence**: Save/load trained models with tokenizers and vocabularies
- **Extensible Design**: Easy addition of new models and preprocessing steps

## Setup

### Automated Setup (Recommended)

Use the provided setup scripts to create a virtual environment and install all dependencies:

**Windows:**
```bash
# Run the setup script
.\setup.bat

# Activate the environment
.\activate.bat
```

**PowerShell:**
```powershell
# Run the setup script
.\setup.ps1

# Activate the environment
.\activate.bat
```

**Unix/Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### Manual Setup

If you prefer manual setup:
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Quick Start

### 1. Environment Setup
```powershell
# Windows (PowerShell) - Automated setup
.\setup.ps1
.\activate.bat

# Windows (Command Prompt)
.\setup.bat
.\activate.bat

# Unix/Linux/macOS
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Training Models

**Train Twitter RoBERTa (Recommended)**:
```powershell
python scripts/train.py --config configs/twitter_roberta.yaml --platform twitter --evaluate
```

**Train BiLSTM with GloVe**:
```powershell
python scripts/train.py --config configs/bilstm_glove.yaml --platform twitter --evaluate
```

**Train on specific platforms**:
```powershell
# Airline sentiment
python scripts/train.py --config configs/twitter_roberta.yaml --platform airline

# Reddit sentiment  
python scripts/train.py --config configs/twitter_roberta.yaml --platform reddit

# Apple sentiment
python scripts/train.py --config configs/twitter_roberta.yaml --platform apple
```

### 3. Model Evaluation

**Evaluate trained model**:
```powershell
python scripts/evaluate.py --model_path models/comment_sentiment_twitter_transformer_cardiffnlp_twitter-roberta-base-sentiment-latest_20250601.pt --config configs/twitter_roberta.yaml --platform twitter
```

**Generate comprehensive evaluation report**:
```powershell
python scripts/evaluate.py --model_path <model_path> --config <config> --output reports/custom_eval
```

### 4. Inference & Prediction

**Single text prediction**:
```powershell
python scripts/predict.py --model_path <model_path> --config <config> --text "I love this new update! 😍 #amazing"
```

**Batch prediction from file**:
```powershell
python scripts/predict.py --model_path <model_path> --config <config> --input_file data/new_comments.csv --output_file predictions.csv
```

### 5. Configuration Examples

**Training Configuration (configs/twitter_roberta.yaml)**:
```yaml
model:
  name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  num_labels: 3
  max_length: 280
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 5
  warmup_steps: 100

preprocessing:
  handle_emojis: "convert"
  expand_contractions: true
  normalize_case: true
```

## Advanced Usage

### Custom Dataset Integration
```python
from src.preprocessing import CommentPreprocessor
from src.data_utils import load_data_from_config

# Configure custom dataset
config = {
    'data': {
        'platform': 'custom',
        'text_column': 'text',
        'label_column': 'sentiment'
    }
}

# Load and preprocess
texts, labels = load_data_from_config(config, data_dir='path/to/custom/data')
```

### Ensemble Model Usage
```python
from src.models import EnsembleModel
from transformers import AutoTokenizer, AutoModel

# Create ensemble with custom weights
ensemble = EnsembleModel(
    transformer_model=roberta_model,
    transformer_weight=0.6,
    vader_weight=0.25,
    textblob_weight=0.15
)
```

### Custom Preprocessing Pipeline
```python
from src.preprocessing import CommentPreprocessor

preprocessor = CommentPreprocessor({
    'handle_emojis': 'convert',
    'expand_contractions': True,
    'handle_mentions': True,
    'handle_hashtags': True,
    'normalize_case': True
})

cleaned_text = preprocessor.clean_text("Love this! 😍 @user #awesome")
```

## Datasets

The project includes comprehensive datasets from multiple social media platforms:

| Dataset | Samples | Description | Sentiment Distribution |
|---------|---------|-------------|----------------------|
| **Twitter Combined** | 69,974 | General Twitter sentiment data | Negative: 30.4%, Neutral: 42.2%, Positive: 27.4% |
| **Twitter Social** | 162,969 | Social media sentiment dataset | Balanced distribution |
| **Reddit Sentiment** | 36,799 | Reddit comments sentiment | Negative: 22.4%, Neutral: 34.7%, Positive: 42.9% |
| **US Airline Sentiment** | 14,427 | Airline customer feedback | Negative: 62.9%, Neutral: 21.2%, Positive: 15.9% |
| **Apple Twitter Sentiment** | 1,624 | Apple product sentiment | Negative: 49.1%, Neutral: 50.9%, Positive: 0.0% |

**Total Training Data**: ~292K preprocessed samples across platforms

### Data Preprocessing Pipeline
- **Automated cleaning**: Removes duplicates, standardizes labels, handles missing data
- **Platform-specific handling**: Adapts to different text formats and labeling schemes
- **Quality assurance**: Text length analysis, preprocessing impact validation
- **Export format**: Standardized CSV with `sentiment` (0=Negative, 1=Neutral, 2=Positive) and `clean_text` columns

## Model Architectures & Performance

### Twitter RoBERTa (Primary Model)
- **Base Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Architecture**: Transformer with additional classification layers
- **Performance**: 97.6% accuracy, 97.7% F1-score on Twitter dataset
- **Features**: Twitter-specific pre-training, 280 character max length, dropout regularization

### BiLSTM with Attention
- **Architecture**: Bidirectional LSTM with GloVe embeddings (300D) and attention mechanism
- **Features**: 
  - Attention-based sequence modeling for variable length texts
  - Pre-trained GloVe embeddings with optional fine-tuning
  - Dropout and layer normalization for regularization
- **Use Case**: Memory-efficient alternative for resource-constrained environments

### Ensemble Model
- **Strategy**: Weighted combination of neural and lexicon-based approaches
- **Components**:
  - Transformer model (50% weight)
  - VADER Sentiment Analyzer (30% weight) 
  - TextBlob sentiment (20% weight)
- **Benefits**: Improved robustness and reduced overfitting to training distribution

### Evaluation Metrics
- **Overall Performance**: 97.6% accuracy with 99.7% ROC-AUC
- **Per-class metrics**: Balanced precision/recall across sentiment classes
- **Error Analysis**: Length-based performance analysis, confidence calibration
- **Visualization**: Confusion matrices, ROC curves, confidence distributions

## Technical Details

### Dependencies & Requirements
- **Python**: 3.8+ (tested with 3.11)
- **PyTorch**: 2.0+ with CUDA support (optional)
- **Transformers**: 4.30+ (Hugging Face)
- **Core ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Text Processing**: NLTK, spaCy, emoji, contractions
- **Sentiment Libraries**: VADER, TextBlob
- **Monitoring**: Weights & Biases, TensorBoard

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended for large datasets
- **GPU**: Optional but recommended for transformer training
- **Storage**: 2GB for dependencies, 1GB for datasets, 500MB for models

### Performance Benchmarks

| Model | Platform | Accuracy | F1-Score | Training Time | Inference Speed |
|-------|----------|----------|----------|---------------|-----------------|
| Twitter RoBERTa | Twitter | 97.6% | 97.7% | ~30 min (GPU) | ~1000 samples/sec |
| BiLSTM + Attention | Twitter | 94.2% | 94.1% | ~10 min (GPU) | ~2000 samples/sec |
| Ensemble | Twitter | 98.1% | 98.0% | N/A (combination) | ~500 samples/sec |

### Model Interpretability
- **Attention Visualization**: BiLSTM attention weights highlight important words
- **Error Analysis**: Detailed misclassification patterns and confidence analysis
- **Feature Analysis**: Text length, emoji usage, social media features correlation
- **Social Media Analytics**: Hashtag sentiment, mention patterns, emoji sentiment mapping

## Research & Methodology

### Problem Statement
Social media sentiment analysis faces unique challenges:
- **Informal Language**: Slang, abbreviations, intentional misspellings
- **Context Dependency**: Sarcasm, cultural references, implicit meaning
- **Evolving Language**: New expressions, hashtags, emoji combinations
- **Platform Differences**: Character limits, user demographics, content types

### Solution Approach
1. **Multi-level Preprocessing**: Platform-specific text normalization
2. **Transfer Learning**: Leveraging Twitter-pretrained models
3. **Ensemble Methods**: Combining neural and lexicon approaches
4. **Attention Mechanisms**: Focus on semantically important words
5. **Comprehensive Evaluation**: Beyond accuracy metrics to practical applicability

### Validation Strategy
- **Cross-platform Validation**: Models trained on one platform tested on others
- **Temporal Robustness**: Training on historical data, testing on recent samples
- **Error Analysis**: Systematic study of failure modes and edge cases
- **Human Evaluation**: Qualitative assessment of difficult cases

## Directory Structure Details

```
Comment-Sentiment/
├── cache/                  # Cached embeddings and preprocessing results
├── logs/                   # Training logs and experiment tracking
├── lexicons/              # Sentiment lexicons and word lists
├── outputs/               # Generated outputs and temporary files
├── visualizations/        # Generated plots and analysis charts
├── activate.bat           # Windows environment activation
├── setup.bat             # Windows automated setup script
├── setup.ps1             # PowerShell setup script
├── setup.py              # Python package configuration
└── requirements.txt       # Python dependencies specification
```

## Extending the Framework

### Adding New Models
```python
# src/models.py
class CustomSentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implementation
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass
        return {'logits': logits, 'loss': loss}

# Register in create_model function
def create_model(config):
    if config['type'] == 'custom':
        return CustomSentimentModel(config)
```
