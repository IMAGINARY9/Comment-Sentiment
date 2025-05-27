"""
Script to prepare and organize comment sentiment data.

This script processes various comment datasets and prepares them for training.
"""

import pandas as pd
import os
import sys
from pathlib import Path
import nltk

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def download_nltk_data():
    """Download required NLTK data."""
    nltk.download('twitter_samples', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

def prepare_twitter_data():
    """Prepare Twitter sentiment datasets."""
    data_dirs = [
        "./data/twitter_sentiment",
        "./data/airline_sentiment", 
        "./data/apple_sentiment",
        "./data/social_media_sentiment"
    ]
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            continue
            
        print(f"Processing {data_dir}")
        
        # Process CSV files in the directory
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                print(f"  Processing {csv_file.name}: {len(df)} rows")
                
                # Standardize column names
                if 'tweet' in df.columns:
                    df['text'] = df['tweet']
                elif 'content' in df.columns:
                    df['text'] = df['content']
                
                # Standardize sentiment labels
                if 'airline_sentiment' in df.columns:
                    df['sentiment'] = df['airline_sentiment']
                elif 'target' in df.columns:
                    df['sentiment'] = df['target']
                
                # Save standardized data
                output_path = data_path / f"standardized_{csv_file.name}"
                df.to_csv(output_path, index=False)
                print(f"    Saved to {output_path}")
                
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")

def prepare_nltk_twitter_samples():
    """Prepare NLTK Twitter samples."""
    try:
        from nltk.corpus import twitter_samples
        
        # Get positive and negative tweets
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')
        
        # Create DataFrame
        data = []
        for tweet in positive_tweets:
            data.append({'text': tweet, 'sentiment': 'positive'})
        for tweet in negative_tweets:
            data.append({'text': tweet, 'sentiment': 'negative'})
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = Path("./data/nltk_twitter_samples.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved NLTK Twitter samples to {output_path}: {len(df)} tweets")
        
    except Exception as e:
        print(f"Error preparing NLTK Twitter samples: {e}")

def create_train_val_test_splits():
    """Create train/validation/test splits."""
    data_dir = Path("./data")
    
    for csv_file in data_dir.glob("standardized_*.csv"):
        df = pd.read_csv(csv_file)
        
        # Shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Split data
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        # Save splits
        base_name = csv_file.stem.replace("standardized_", "")
        train_df.to_csv(data_dir / f"{base_name}_train.csv", index=False)
        val_df.to_csv(data_dir / f"{base_name}_val.csv", index=False)
        test_df.to_csv(data_dir / f"{base_name}_test.csv", index=False)
        
        print(f"Split {csv_file.name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    print("Preparing comment sentiment data...")
    
    # Create output directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./lexicons", exist_ok=True)
    
    # Download NLTK data
    print("Downloading NLTK data...")
    download_nltk_data()
    
    # Prepare Twitter data
    print("Preparing Twitter datasets...")
    prepare_twitter_data()
    
    # Prepare NLTK samples
    print("Preparing NLTK Twitter samples...")
    prepare_nltk_twitter_samples()
    
    # Create splits
    print("Creating train/val/test splits...")
    create_train_val_test_splits()
    
    print("Data preparation complete!")
