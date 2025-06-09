"""
Script to predict sentiment using a trained model for custom input (string, array, or file).

Usage:
    python predict.py --model_dir <path_to_model_dir> --input <text|file|array> [--platform <platform>] [--visualize]

Examples:
    python scripts/predict.py --model_dir models/comment_sentiment_bilstm_20250601 --input "I love this product!"
    python scripts/predict.py --model_dir models/comment_sentiment_bilstm_20250601 --input_file my_texts.txt
    python scripts/predict.py --model_dir models/comment_sentiment_bilstm_20250601 --input_array "['I love it', 'I hate it']"
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import ast
import os
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
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

def load_input(args):
    if args.input:
        return [args.input]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif args.input_array:
        return ast.literal_eval(args.input_array)
    else:
        raise ValueError("No input provided. Use --input, --input_file, or --input_array.")

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for custom input using a trained model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model.pt, config.yaml, and vocab files')
    parser.add_argument('--input', type=str, help='Input string to predict')
    parser.add_argument('--input_file', type=str, help='Path to file with one text per line')
    parser.add_argument('--input_array', type=str, help='Python list as string, e.g. "[\"text1\", \"text2\"]"')
    parser.add_argument('--platform', type=str, default=None, help='Platform override (twitter, airline, apple, etc)')
    parser.add_argument('--visualize', action='store_true', help='If set, generate and save prediction explanation plots')
    parser.add_argument('--baseline_text', type=str, default=None, help='Custom baseline text if baseline_type is "text"')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find model.pt, config.yaml, and vocab files
    model_path = find_file_in_dir(model_dir, ["model.pt", "*.pt"])
    
    # Look for the main config file, excluding calibration_params.yaml
    config_path = None
    for pattern in ["*_config.yaml", "config.yaml", "*.yaml", "*.yml"]:
        for file in model_dir.glob(pattern):
            if "calibration_params" not in file.name:
                config_path = str(file)
                break
        if config_path:
            break
    
    word2idx_path_json = model_dir / 'word2idx.json'
    word2idx_path_pkl = model_dir / 'word2idx.pkl'

    if not model_path or not config_path:
        raise FileNotFoundError(f"Could not find model.pt or config.yaml in {model_dir}")

    config = load_config(config_path)
    model_config = config['model']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Visualization imports and setup
    if args.visualize:
        from src.visualization import explain_and_plot_transformer, explain_and_plot_lstm
        vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    # Load model and word2idx if BiLSTM
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
                print("Warning: word2idx not found in checkpoint or as a separate file. BiLSTM prediction will fail if vocabulary is required.")
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

    # Tokenizer for transformer
    tokenizer = None
    if model_config.get('type', 'transformer') == 'transformer' or 'roberta' in model_config.get('name', ''):
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])

    # Label mapping (default)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    label_names = [label_map[i] for i in range(len(label_map))]

    # Load input
    texts = load_input(args)
    print(f"Predicting sentiment for {len(texts)} input(s)...")

    # Prediction logic
    results = []
    if tokenizer:
        # Transformer
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=model_config.get('max_length', 280),
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': int(preds[i]),
                'confidence': float(probs[i][preds[i]]),
                'probabilities': probs[i].cpu().numpy().tolist()
            })
            print(f"Text: {text}")
            print(f"Predicted: {label_map.get(int(preds[i]), int(preds[i]))} (confidence: {probs[i][preds[i]]:.3f})")
            print(f"Probabilities: {probs[i].cpu().numpy().tolist()}")
            print('-' * 60)
            if args.visualize:
                save_path = os.path.join(vis_dir, f"transformer_explain_{i}.png")
                try:
                    explain_and_plot_transformer(model, tokenizer, text, label_names, save_path)
                    print(f"Visualization saved to {save_path}")
                except Exception as e:
                    print(f"Visualization failed: {e}")
    elif word2idx:
        # BiLSTM
        max_length = model_config.get('max_length', 128)
        sequences = []
        for t in texts:
            tokens = t.split()[:max_length]
            seq = [word2idx.get(token, 0) for token in tokens]
            sequences.append(seq)
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=0).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': int(preds[i]),
                'confidence': float(probs[i][preds[i]]),
                'probabilities': probs[i].cpu().numpy().tolist()
            })
            print(f"Text: {text}")
            print(f"Predicted: {label_map.get(int(preds[i]), int(preds[i]))} (confidence: {probs[i][preds[i]]:.3f})")
            print(f"Probabilities: {probs[i].cpu().numpy().tolist()}")
            print('-' * 60)
            if args.visualize:
                # For LSTM, need a vocab_builder-like object. We'll pass word2idx and a dummy for compatibility.
                class DummyVocabBuilder:
                    def __init__(self, word2idx, max_len):
                        self.word2idx = word2idx
                        self.max_len = max_len
                        self.pad_token = '<pad>' if '<pad>' in word2idx else None
                    def preprocess(self, texts):
                        return [t.split()[:self.max_len] for t in texts]
                    def encode(self, tokenized, max_len=None):
                        max_len = max_len or self.max_len
                        return [[self.word2idx.get(tok, 0) for tok in toks] + [0]*(max_len-len(toks)) for toks in tokenized]
                vocab_builder = DummyVocabBuilder(word2idx, max_length)
                save_path = os.path.join(vis_dir, f"lstm_explain_{i}.png")
                try:
                    explain_and_plot_lstm(
                        model, vocab_builder, text, label_names, save_path,
                        pred_probs=probs[i].cpu().numpy(), pred_label=int(preds[i]),
                        attribution_method="integrated_gradients",
                        baseline_type="text" if args.baseline_text else "pad",
                        baseline_text=args.baseline_text or None
                    )
                    print(f"Visualization saved to {save_path}")
                except Exception as e:
                    print(f"Visualization failed: {e}")
    else:
        raise NotImplementedError("Prediction for this model type is not implemented.")

if __name__ == "__main__":
    main()
