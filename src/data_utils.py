"""
Data loading utilities for comment sentiment analysis project.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def load_data_from_config(config, platform_override=None, data_dir_override=None):
    data_dir = Path(config['data'].get('data_dir', 'data/twitter_sentiment'))
    platform = platform_override or config['data'].get('platform', 'twitter')
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
        data_file = cleaned_file
        print(f"Using cleaned data file: {data_file}")
    else:
        if platform == 'twitter':
            data_file = data_dir / 'twitter_combined.csv'
        elif platform == 'airline':
            data_file = Path('data/airline_sentiment/Tweets.csv')
        elif platform == 'apple':
            data_file = Path('data/apple_sentiment/apple-twitter-sentiment-texts.csv')
        elif platform == 'reddit':
            data_file = Path('data/social_media_sentiment/Reddit_Data.csv')
        elif platform == 'twitter_social':
            data_file = Path('data/social_media_sentiment/Twitter_Data.csv')
        else:
            raise ValueError(f"Unknown platform: {platform}")
        print(f"Using original data file: {data_file}")
    df = pd.read_csv(data_file)
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
    return texts, labels
