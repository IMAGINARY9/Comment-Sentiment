"""
Main training script for comment sentiment analysis models.
"""
import argparse
import yaml
import wandb
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_utils import load_data_from_config
from bilstm_trainer import train_bilstm
from models import create_model
from training import CommentTrainer
from evaluation import CommentEvaluator, create_evaluation_report
from transformers import AutoTokenizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train comment sentiment analysis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Data directory override')
    parser.add_argument('--platform', type=str, default=None, help='Platform override (twitter, airline, apple)')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training and create plots')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data:
        config['data']['data_dir'] = args.data
    if args.platform:
        config['data']['platform'] = args.platform

    if args.wandb:
        wandb.init(
            project="comment-sentiment",
            config=config,
            name=f"comment_sentiment_{config['model'].get('name', 'model').replace('/', '_')}"
        )

    print(f"Training comment sentiment model with config: {args.config}")
    print(f"Platform: {config['data'].get('platform', 'generic')}")

    texts, labels = load_data_from_config(config, args.platform, args.data)
    if args.debug:
        texts = texts[:1000]
        labels = labels[:1000]
        print(f"Debug mode: using {len(texts)} samples")

    model_config = config['model']
    model_type = model_config.get('type', 'transformer')

    if model_type == 'transformer' or 'roberta' in model_config.get('name', ''):
        print("Training transformer model...")
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        model = create_model(model_config)
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Label distribution: {dict(zip(*__import__('numpy').unique(train_labels, return_counts=True)))}")
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
        model_save_path = f"models/comment_sentiment_{config['data'].get('platform', 'generic')}_transformer_{model_config['name'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.pt"
        trainer.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")
        if args.evaluate:
            print("Running evaluation...")
            evaluator = CommentEvaluator(model, tokenizer)
            eval_report_dir = f"reports/{config['data'].get('platform', 'generic')}_transformer_eval"
            create_evaluation_report(evaluator, val_texts, val_labels, eval_report_dir)
            print(f"Evaluation report and plots saved to {eval_report_dir}")
    elif model_type == 'bilstm':
        print("Training BiLSTM model...")
        from bilstm_trainer import train_bilstm
        # train_bilstm returns (model, val_sequences, val_labels, word2idx) if evaluate=True
        result = train_bilstm(texts, labels, model_config, config['training'], evaluate=args.evaluate)
        if args.evaluate and result is not None:
            model, val_seqs, val_labels, word2idx = result
            # For plots, reconstruct val_texts from val_seqs and word2idx
            inv_word2idx = {v: k for k, v in word2idx.items()}
            val_texts = [" ".join([inv_word2idx.get(idx, '') for idx in seq if idx in inv_word2idx]) for seq in val_seqs]
            evaluator = CommentEvaluator(model, word2idx=word2idx)
            eval_report_dir = f"reports/{config['data'].get('platform', 'generic')}_bilstm_eval"
            create_evaluation_report(evaluator, val_texts, val_labels, eval_report_dir)
            print(f"Evaluation report and plots saved to {eval_report_dir}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("Training completed successfully!")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
