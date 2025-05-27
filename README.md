# Comment Sentiment Analysis

This project implements sentiment analysis for comments, focusing on handling informal language, slang, emojis, and social media text.

## Overview

- **Task**: Classify sentiment of comments from social media, forums, reviews
- **Challenges**: Informal language, slang, misspellings, emojis, abbreviations
- **Approach**: Fine-tuned transformers + lexicon-based methods + traditional ML

## Project Structure

```
comment-sentiment/
├── data/                    # Comment datasets (Twitter, Reddit, etc.)
├── src/                     # Source code
├── models/                  # Trained models and checkpoints
├── notebooks/               # Jupyter notebooks for experiments
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation scripts
├── references/              # Reference notebooks and papers
├── logs/                    # Training logs
├── reports/                 # Analysis reports and results
├── lexicons/                # Sentiment lexicons (VADER, etc.)
└── tests/                   # Unit tests
```

## Key Features

1. **Informal Text Handling**: Slang expansion, emoji processing, spelling correction
2. **Multiple Approaches**: Transformers, LSTM, lexicon-based (VADER, TextBlob)
3. **Ensemble Methods**: Combining multiple models for better performance
4. **Social Media Focus**: Twitter, Reddit, forum comment analysis

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

1. **Setup environment**: Use `setup.bat` (Windows) or `setup.sh` (Unix/Linux)
2. **Activate environment**: Run `activate.bat` or `source venv/bin/activate`
3. **Download lexicons**: `python scripts/download_lexicons.py`
4. **Prepare data**: `python scripts/prepare_data.py`
5. **Train model**: `python scripts/train.py --config configs/twitter_roberta.yaml`
6. **Evaluate**: `python scripts/evaluate.py --ensemble`

## Datasets

- Twitter Sentiment datasets (Twitter_Data.csv, Twitter Sentiment Analysis)
- Reddit comments (Reddit_Data.csv)
- Twitter US Airline Sentiment
- Apple Twitter Sentiment
- NLTK twitter_samples

## Models

- Fine-tuned RoBERTa for social media
- Bi-LSTM with GloVe embeddings
- VADER sentiment analyzer
- TextBlob sentiment
- Ensemble approaches
