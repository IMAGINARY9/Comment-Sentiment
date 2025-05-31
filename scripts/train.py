"""
Training script for comment sentiment analysis models.

Usage:
    python train.py --config configs/twitter_roberta.yaml
    python train.py --config configs/bilstm_glove.yaml --data data/twitter_sentiment/
"""

import argparse
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from training import CommentTrainer
from evaluation import CommentEvaluator, create_evaluation_report
from transformers import AutoTokenizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train comment sentiment analysis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Data directory override')
    parser.add_argument('--platform', type=str, default=None, help='Platform override (twitter, airline, apple)')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.data:
        config['data']['data_dir'] = args.data
    if args.platform:
        config['data']['platform'] = args.platform
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="comment-sentiment",
            config=config,
            name=f"comment_sentiment_{config['model'].get('name', 'model').replace('/', '_')}"
        )
    
    print(f"Training comment sentiment model with config: {args.config}")
    print(f"Platform: {config['data'].get('platform', 'generic')}")
    
    # Load data
    print("Loading data...")
    data_dir = Path(config['data'].get('data_dir', 'data/twitter_sentiment'))
    platform = config['data'].get('platform', 'twitter')
    
    # Expect pre-cleaned data from notebook
    # User should run the data_cleaning_preprocessing notebook before training
    # Load pre-cleaned data
    data_file = None
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
    
    df = pd.read_csv(data_file)
    if 'clean_text' in df.columns:
        texts = df['clean_text'].astype(str).tolist()
    else:
        # Fallback to text/comment column
        text_col = [c for c in ['text', 'comment', 'clean_comment', 'tweet'] if c in df.columns]
        texts = df[text_col[0]].astype(str).tolist()
    label_col = [c for c in ['sentiment', 'category', 'label', 'target'] if c in df.columns]
    labels = df[label_col[0]].tolist()
    
    # Debug mode - use smaller dataset
    if args.debug:
        texts = texts[:1000]
        labels = labels[:1000]
        print(f"Debug mode: using {len(texts)} samples")
    
    # Check model type
    model_config = config['model']
    model_type = model_config.get('type', 'transformer')
    
    if model_type == 'transformer' or 'roberta' in model_config.get('name', ''):
        # Transformer model training
        print("Training transformer model...")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        
        # Create model
        model = create_model(model_config)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        
        # Train model
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
        
        # Save model
        model_save_path = f"models/comment_sentiment_{platform}_transformer.pt"
        trainer.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Evaluation
        if args.evaluate:
            print("Running evaluation...")
            evaluator = CommentEvaluator(model, tokenizer)
            create_evaluation_report(evaluator, val_texts, val_labels, f"reports/{platform}_transformer_eval")
    
    elif model_type == 'bilstm':
        # BiLSTM model training
        print("Training BiLSTM model...")
        # Expect pre-tokenized/cleaned data from notebook
        # Convert texts to sequences using a simple tokenizer (split on whitespace)
        max_length = model_config.get('max_length', 128)
        sequences = [t.split()[:max_length] for t in texts]
        # Map tokens to integer ids (simple word index)
        word2idx = {}
        idx = 1
        for seq in sequences:
            for token in seq:
                if token not in word2idx:
                    word2idx[token] = idx
                    idx += 1
        sequences = [[word2idx[token] for token in seq] for seq in sequences]
        # Split data
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training samples: {len(train_seqs)}")
        print(f"Validation samples: {len(val_seqs)}")
        print(f"Vocabulary size: {len(word2idx)}")
        # Create dummy datasets for BiLSTM (sequences instead of texts)
        class SequenceDataset:
            def __init__(self, sequences, labels):
                self.sequences = sequences
                self.labels = labels
            def __len__(self):
                return len(self.sequences)
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
                    'label': torch.tensor(self.labels[idx], dtype=torch.long)
                }
        from torch.utils.data import DataLoader
        train_dataset = SequenceDataset(train_seqs, train_labels)
        val_dataset = SequenceDataset(val_seqs, val_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        print("Starting BiLSTM training...")
        for epoch in range(config['training']['num_epochs']):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, labels)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Loss: {avg_loss:.4f}")
        model_save_path = f"models/comment_sentiment_{platform}_bilstm.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Training completed successfully!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
