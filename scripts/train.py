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
from models import create_model
from training import CommentTrainer, build_vocabulary_for_bilstm
from evaluation import CommentEvaluator, create_evaluation_report
from transformers import AutoTokenizer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

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
        # Compute class weights
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        print(f"Class weights: {class_weights}")
        config['training']['class_weights'] = class_weights.tolist()
        trainer = CommentTrainer(model, config)
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
        # Create a timestamped folder for saving model and config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        platform = config['data'].get('platform', 'generic')
        model_name = model_config['name'].replace('/', '_')
        save_dir = Path(f"models/{platform}_{model_name}_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        trainer.save_model(str(save_dir), original_config=config)

        print(f"Model saved to {save_dir}")
        if args.evaluate:
            print("Running evaluation...")
            evaluator = CommentEvaluator(model, tokenizer)
            eval_report_dir = f"reports/{config['data'].get('platform', 'generic')}_transformer_eval"            
            create_evaluation_report(evaluator, val_texts, val_labels, eval_report_dir)
            print(f"Evaluation report and plots saved to {eval_report_dir}")
    elif model_type == 'bilstm':
        print("Training BiLSTM model...")
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
          # Print class distribution for all splits
        def print_class_dist(labels, name):
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"{name} class distribution: {dist}")
        
        print_class_dist(train_labels, "Train")
        print_class_dist(val_labels, "Validation")
        
        print(f"Label distribution: {dict(zip(*__import__('numpy').unique(train_labels, return_counts=True)))}")
        # Compute class weights
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        print(f"Class weights: {class_weights}")
        config['training']['class_weights'] = class_weights.tolist()
          # Build vocabulary first for BiLSTM
        vocabulary, vocab_size = build_vocabulary_for_bilstm(train_texts, config)
        
        # Update model config with correct vocab size (+1 for index 0 padding/unknown)
        model_config['vocab_size'] = vocab_size + 1
        print(f"Updated model config with vocab_size: {vocab_size + 1} (vocabulary words: {vocab_size}, +1 for padding/unknown)")
        
        model = create_model(model_config)
        trainer = CommentTrainer(model, config)
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, vocabulary=vocabulary)
        # Create a timestamped folder for saving model and config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        platform = config['data'].get('platform', 'generic')
        model_name = model_config.get('name', 'bilstm')        
        save_dir = Path(f"models/{platform}_{model_name}_{timestamp}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        trainer.save_model(str(save_dir), original_config=config)
        
        print(f"Model saved to {save_dir}")
        if args.evaluate:
            print("Running evaluation...")
            
            # Verification Step 1: Confirm word2idx vocabulary
            word2idx = getattr(trainer, 'word2idx', None)
            if word2idx is None:
                print("[WARNING] No word2idx vocabulary found in trainer!")
            else:
                print(f"[INFO] Vocabulary size: {len(word2idx)}")
                # Show sample vocabulary entries
                sample_words = list(word2idx.keys())[:10]
                print(f"[INFO] Sample vocabulary: {sample_words}")
            
            # Verification Step 2: Confirm best model is loaded
            print(f"[INFO] Best validation F1 during training: {trainer.best_val_f1:.4f}")
            
            # Verification Step 3: Get calibration parameters
            calibration_params = getattr(trainer, 'calibration_params', None)
            if calibration_params is None:
                print("[WARNING] No calibration parameters found!")
            else:
                print(f"[INFO] Calibration parameters available for {len(calibration_params['coef'])} classes")
            
            # Create evaluator with all available parameters
            evaluator = CommentEvaluator(
                model, 
                word2idx=word2idx, 
                calibration_params=calibration_params
            )
            
            # Verification Step 4: Test model prediction on a small sample
            print("[INFO] Testing model prediction on a sample...")
            sample_texts = val_texts[:5] if len(val_texts) >= 5 else val_texts
            sample_labels = val_labels[:5] if len(val_labels) >= 5 else val_labels
            
            try:
                sample_predictions, sample_probs = evaluator.predict_batch(sample_texts)
                print(f"[INFO] Sample predictions: {sample_predictions}")
                print(f"[INFO] Sample true labels: {sample_labels}")
                print(f"[INFO] Sample max probabilities: {np.max(sample_probs, axis=1)}")
                
                # Check if all predictions are the same
                unique_preds = np.unique(sample_predictions)
                if len(unique_preds) == 1:
                    print(f"[WARNING] All sample predictions are the same class: {evaluator.label_names[unique_preds[0]]}")
                else:
                    print(f"[INFO] Sample shows {len(unique_preds)} different predicted classes")
                    
            except Exception as e:
                print(f"[ERROR] Failed to make sample predictions: {e}")
                return
            
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
