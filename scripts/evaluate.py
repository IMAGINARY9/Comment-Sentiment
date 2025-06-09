"""
Script to evaluate a trained sentiment analysis model.

Usage:
    python evaluate.py --model_dir <path_to_model_dir> --data <data_csv> [--platform <platform>] [--output <output_dir>]
"""
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from evaluation import CommentEvaluator, create_evaluation_report
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_file_in_dir(directory, patterns):
    """Find the first file in directory matching any of the patterns."""
    for pattern in patterns:
        for file in Path(directory).glob(pattern):
            return str(file)
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained sentiment analysis model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model.pt, config.yaml, and vocab files')
    parser.add_argument('--data', type=str, required=False, help='Path to evaluation data CSV (optional, will use config/platform defaults if not provided)')
    parser.add_argument('--platform', type=str, default=None, help='Platform override (twitter, airline, apple, etc)')
    parser.add_argument('--output', type=str, default=None, help='Directory to save evaluation report and plots')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find model.pt, config.yaml, and vocab files
    model_path = find_file_in_dir(model_dir, ["model.pt", "*.pt"])
    config_path = find_file_in_dir(model_dir, ["config.yaml", "*.yaml", "*.yml"])
    word2idx_path_json = model_dir / 'word2idx.json'
    word2idx_path_pkl = model_dir / 'word2idx.pkl'

    if not model_path or not config_path:
        raise FileNotFoundError(f"Could not find model.pt or config.yaml in {model_dir}")

    config = load_config(config_path)
    platform = args.platform or config['data'].get('platform', 'twitter')

    # Data loading logic (same as train.py)
    if args.data:
        data_path = args.data
    else:
        data_dir = Path(config['data'].get('data_dir', 'data/twitter_sentiment'))
        preprocessed_dir = Path('data/preprocessed')
        cleaned_file_map = {
            'twitter': preprocessed_dir / 'twitter_combined_sentiment_cleaned.csv',
            'airline': preprocessed_dir / 'airline_sentiment_cleaned.csv',
            'apple': preprocessed_dir / 'apple_sentiment_cleaned.csv',
            'reddit': preprocessed_dir / 'reddit_sentiment_cleaned.csv',
            'twitter_social': preprocessed_dir / 'twitter_social_sentiment_cleaned.csv',
        }
        cleaned_file = cleaned_file_map.get(platform)
        if cleaned_file and cleaned_file.exists():
            data_path = cleaned_file
            print(f"Using cleaned data file: {data_path}")
        else:
            if platform == 'twitter':
                data_path = data_dir / 'twitter_combined.csv'
            elif platform == 'airline':
                data_path = Path('data/airline_sentiment/Tweets.csv')
            elif platform == 'apple':
                data_path = Path('data/apple_sentiment/apple-twitter-sentiment-texts.csv')
            elif platform == 'reddit':
                data_path = Path('data/social_media_sentiment/Reddit_Data.csv')
            elif platform == 'twitter_social':
                data_path = Path('data/social_media_sentiment/Twitter_Data.csv')
            else:
                raise ValueError(f"Unknown platform: {platform}")
            print(f"Using original data file: {data_path}")
    df = pd.read_csv(data_path)
    if 'clean_text' in df.columns:
        texts = df['clean_text'].astype(str).tolist()
    else:
        text_col = [c for c in ['text', 'comment', 'clean_comment', 'tweet'] if c in df.columns]
        texts = df[text_col[0]].astype(str).tolist()
    label_col = [c for c in ['sentiment', 'category', 'label', 'target'] if c in df.columns]
    labels_raw = df[label_col[0]].tolist()
    if not all(isinstance(l, (int, np.integer)) for l in labels_raw):
        unique_labels = sorted(set(labels_raw))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Label mapping: {label2id}")
        labels = [label2id[l] for l in labels_raw]
    else:
        labels = labels_raw

    # Model loading logic
    model_config = config['model']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    word2idx = None
    if model_config.get('type', 'transformer') != 'transformer' and 'roberta' not in model_config.get('name', ''):
        checkpoint = torch.load(model_path, map_location=device)
        word2idx = checkpoint.get('word2idx', None)
        # Try to load word2idx from a separate file if not found in checkpoint
        if word2idx is None:
            if word2idx_path_json.exists():
                with open(word2idx_path_json, 'r', encoding='utf-8') as f:
                    word2idx = json.load(f)
                print(f"Loaded word2idx from {word2idx_path_json}")
            elif word2idx_path_pkl.exists():
                import pickle
                with open(word2idx_path_pkl, 'rb') as f:
                    word2idx = pickle.load(f)
                print(f"Loaded word2idx from {word2idx_path_pkl}")
            else:
                print("Warning: word2idx not found in checkpoint or as a separate file. BiLSTM evaluation will fail if vocabulary is required.")
        if word2idx is not None:
            config['data']['vocab_size'] = len(word2idx) + 1
        model = create_model(model_config)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = create_model(model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Tokenizer (for transformer models)
    tokenizer = None
    if model_config.get('type', 'transformer') == 'transformer' or 'roberta' in model_config.get('name', ''):
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])

    # Evaluate and create plots
    if model_config.get('type', 'transformer') != 'transformer' and 'roberta' not in model_config.get('name', ''):
        evaluator = CommentEvaluator(model, tokenizer, word2idx=word2idx)
    else:
        evaluator = CommentEvaluator(model, tokenizer)
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = args.output or f"reports/eval_{platform}_{date_str}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Evaluating and creating plots in {output_dir} ...")
    create_evaluation_report(evaluator, texts, labels, output_dir)
    print(f"Evaluation report and plots saved to {output_dir}")

if __name__ == "__main__":
    main()
