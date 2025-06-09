"""
Evaluation utilities for comment sentiment analysis.

This module provides comprehensive evaluation tools including
metrics calculation, visualization, and error analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict, List, Optional, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class CommentEvaluator:
    """
    Comprehensive evaluator for comment sentiment analysis models.
    
    This class provides evaluation metrics, visualizations, and analysis
    tools for assessing model performance on social media text.
    """
    
    def __init__(self, model: nn.Module, tokenizer=None, device: str = 'auto', word2idx=None, calibration_params=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            tokenizer: Tokenizer for transformer models
            device: Device to run evaluation on
            word2idx: Word-to-index mapping for BiLSTM models
            calibration_params: Platt scaling calibration parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.word2idx = word2idx
        self.calibration_params = calibration_params
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.label_names = ['Negative', 'Neutral', 'Positive']
        self.label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling calibration to logits.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Calibrated probabilities
        """
        if self.calibration_params is None:
            return torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        num_classes = logits.shape[1]
        calibrated_probs = np.zeros_like(logits)
        
        for i in range(num_classes):
            # Apply one-vs-rest calibration
            coef = self.calibration_params['coef'][i]
            intercept = self.calibration_params['intercept'][i]
            calibrated_logits = coef * logits[:, i] + intercept
            calibrated_probs[:, i] = 1 / (1 + np.exp(-calibrated_logits))  # Sigmoid
        
        # Normalize to ensure probabilities sum to 1
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
        
        return calibrated_probs

    def texts_to_sequences(self, texts, max_length=128):
        if not self.word2idx:
            raise ValueError("word2idx must be provided for BiLSTM evaluation.")
        sequences = []
        for t in texts:
            tokens = t.split()[:max_length]
            seq = [self.word2idx.get(token, 0) for token in tokens]
            sequences.append(seq)
        return sequences

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            if self.tokenizer:
                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=280,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    if hasattr(self.model, 'forward') and 'texts' in self.model.forward.__code__.co_varnames:
                        # Ensemble model
                        outputs = self.model(input_ids, attention_mask, batch_texts)
                    else:
                        outputs = self.model(input_ids, attention_mask)                    
                    logits = outputs['logits']
                    
                    # Apply calibration if available
                    if self.calibration_params is not None:
                        probabilities = self._apply_calibration(logits.cpu().numpy())
                        probabilities = torch.tensor(probabilities)
                    else:
                        probabilities = torch.softmax(logits, dim=-1)
                    
                    predictions = torch.argmax(probabilities, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
            elif self.word2idx:
                # For BiLSTM: convert texts to sequences
                sequences = self.texts_to_sequences(batch_texts)
                from torch.utils.data import DataLoader
                from torch.nn.utils.rnn import pad_sequence
                class SequenceDataset:
                    def __init__(self, sequences):
                        self.sequences = sequences
                    def __len__(self):
                        return len(self.sequences)
                    def __getitem__(self, idx):
                        return torch.tensor(self.sequences[idx], dtype=torch.long)
                def collate_fn(batch):
                    input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
                    return input_ids
                dataset = SequenceDataset(sequences)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)                
                for input_ids in dataloader:
                    input_ids = input_ids.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs['logits']
                        
                        # Apply calibration if available
                        if self.calibration_params is not None:
                            probabilities = self._apply_calibration(logits.cpu().numpy())
                            probabilities = torch.tensor(probabilities)
                        else:
                            probabilities = torch.softmax(logits, dim=-1)
                        
                        predictions = torch.argmax(probabilities, dim=-1)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())
            else:
                # For traditional models, need to convert texts to sequences
                # This would require vocabulary - simplified for now
                raise NotImplementedError("Evaluation for non-transformer models needs vocabulary")
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate_dataset(self, texts: List[str], true_labels: List[int]) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            texts: List of input texts
            true_labels: List of ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions,
            target_names=self.label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Print confusion matrix and warn if only one class is predicted
        unique_preds = np.unique(predictions)
        if len(unique_preds) == 1:
            print(f"[WARNING] Model predicted only one class: {self.label_names[unique_preds[0]]} for all samples.")
        print("Confusion Matrix:")
        print(cm)
        
        # ROC AUC (for multiclass)
        try:
            roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr')
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class,
                'support': support
            },
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            annot_kws={"size": 12},
            cbar_kws={"shrink": 0.7}
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curves(self, true_labels: List[int], probabilities: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for each class.
        
        Args:
            true_labels: Ground truth labels
            probabilities: Prediction probabilities
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_bin = label_binarize(true_labels, classes=[0, 1, 2])
        
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.label_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
            auc = roc_auc_score(y_bin[:, i], probabilities[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (All Classes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_prediction_confidence(self, probabilities: np.ndarray, predictions: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot prediction confidence distribution.
        
        Args:
            probabilities: Prediction probabilities
            predictions: Predicted labels
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall confidence distribution
        axes[0].hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Confidence Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by predicted class
        for i, class_name in enumerate(self.label_names):
            class_mask = predictions == i
            class_confidences = confidence_scores[class_mask]
            axes[1].hist(class_confidences, bins=20, alpha=0.6, label=class_name, edgecolor='black')
        
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution by Class')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_errors(self, texts: List[str], true_labels: List[int],
                      predictions: np.ndarray, probabilities: np.ndarray) -> Dict:
        """
        Analyze prediction errors.
        
        Args:
            texts: Input texts
            true_labels: Ground truth labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Error analysis results
        """
        # Find misclassified samples
        misclassified_mask = predictions != np.array(true_labels)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(texts),
            'errors_by_true_class': {},
            'errors_by_predicted_class': {},
            'low_confidence_errors': [],
            'high_confidence_errors': [],
            'common_error_patterns': {}
        }
        
        # Analyze errors by true class
        for i, class_name in enumerate(self.label_names):
            true_class_mask = np.array(true_labels) == i
            class_errors = misclassified_mask & true_class_mask
            error_analysis['errors_by_true_class'][class_name] = {
                'count': np.sum(class_errors),
                'rate': np.sum(class_errors) / np.sum(true_class_mask) if np.sum(true_class_mask) > 0 else 0
            }
        
        # Analyze errors by predicted class
        for i, class_name in enumerate(self.label_names):
            pred_class_mask = predictions == i
            class_errors = misclassified_mask & pred_class_mask
            error_analysis['errors_by_predicted_class'][class_name] = {
                'count': np.sum(class_errors),
                'rate': np.sum(class_errors) / np.sum(pred_class_mask) if np.sum(pred_class_mask) > 0 else 0
            }
        
        # Analyze confidence of errors
        if len(misclassified_indices) > 0:
            error_confidences = np.max(probabilities[misclassified_indices], axis=1)
            
            # Low confidence errors (bottom 25%)
            low_conf_threshold = np.percentile(error_confidences, 25)
            low_conf_mask = error_confidences <= low_conf_threshold
            low_conf_indices = misclassified_indices[low_conf_mask]
            
            # High confidence errors (top 25%)
            high_conf_threshold = np.percentile(error_confidences, 75)
            high_conf_mask = error_confidences >= high_conf_threshold
            high_conf_indices = misclassified_indices[high_conf_mask]
            
            # Sample errors for analysis
            for idx in low_conf_indices[:5]:  # Top 5 low confidence errors
                error_analysis['low_confidence_errors'].append({
                    'text': texts[idx],
                    'true_label': self.label_mapping[true_labels[idx]],
                    'predicted_label': self.label_mapping[predictions[idx]],
                    'confidence': float(np.max(probabilities[idx]))
                })
            
            for idx in high_conf_indices[:5]:  # Top 5 high confidence errors
                error_analysis['high_confidence_errors'].append({
                    'text': texts[idx],
                    'true_label': self.label_mapping[true_labels[idx]],
                    'predicted_label': self.label_mapping[predictions[idx]],
                    'confidence': float(np.max(probabilities[idx]))
                })
        
        return error_analysis
    
    def length_based_analysis(self, texts: List[str], true_labels: List[int],
                            predictions: np.ndarray) -> Dict:
        """
        Analyze performance based on text length.
        
        Args:
            texts: Input texts
            true_labels: Ground truth labels
            predictions: Predicted labels
            
        Returns:
            Length-based analysis results
        """
        text_lengths = [len(text.split()) for text in texts]
        
        # Define length bins
        length_bins = [
            (0, 10, 'Very Short (≤10 words)'),
            (11, 20, 'Short (11-20 words)'),
            (21, 50, 'Medium (21-50 words)'),
            (51, float('inf'), 'Long (>50 words)')
        ]
        
        analysis = {}
        
        for min_len, max_len, bin_name in length_bins:
            if max_len == float('inf'):
                mask = np.array(text_lengths) >= min_len
            else:
                mask = (np.array(text_lengths) >= min_len) & (np.array(text_lengths) <= max_len)
            
            if np.sum(mask) > 0:
                bin_true = np.array(true_labels)[mask]
                bin_pred = predictions[mask]
                
                accuracy = accuracy_score(bin_true, bin_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    bin_true, bin_pred, average='weighted'
                )
                
                analysis[bin_name] = {
                    'count': np.sum(mask),
                    'avg_length': np.mean(np.array(text_lengths)[mask]),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        return analysis
    
    def platform_comparison(self, results_dict: Dict[str, Dict]) -> None:
        """
        Compare performance across different platforms.
        
        Args:
            results_dict: Dictionary with platform names as keys and evaluation results as values
        """
        platforms = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison DataFrame
        comparison_data = []
        for platform in platforms:
            for metric in metrics:
                comparison_data.append({
                    'Platform': platform,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': results_dict[platform][metric]
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig = px.bar(
            df, x='Platform', y='Score', color='Metric',
            title='Performance Comparison Across Platforms',
            barmode='group'
        )
        fig.show()
        
        # Print summary table
        print("\nPlatform Performance Summary:")
        print("=" * 50)
        for platform in platforms:
            print(f"\n{platform}:")
            for metric in metrics:
                score = results_dict[platform][metric]
                print(f"  {metric.replace('_', ' ').title()}: {score:.4f}")

class SocialMediaAnalyzer:
    """
    Specialized analyzer for social media sentiment patterns.
    
    This class provides analysis tools specific to social media content
    including emoji analysis, hashtag sentiment, and temporal patterns.
    """
    
    def __init__(self, evaluator: CommentEvaluator):
        """
        Initialize analyzer.
        
        Args:
            evaluator: CommentEvaluator instance
        """
        self.evaluator = evaluator
    
    def analyze_emoji_sentiment(self, texts: List[str], predictions: np.ndarray) -> Dict:
        """
        Analyze sentiment patterns related to emoji usage.
        
        Args:
            texts: Input texts
            predictions: Predicted labels
            
        Returns:
            Emoji sentiment analysis results
        """
        import emoji
        
        emoji_sentiment = {label: [] for label in self.evaluator.label_names}
        emoji_counts = {label: Counter() for label in self.evaluator.label_names}
        
        for text, pred in zip(texts, predictions):
            label_name = self.evaluator.label_mapping[pred]
            
            # Extract emojis
            emojis = [char for char in text if char in emoji.EMOJI_DATA]
            emoji_sentiment[label_name].extend(emojis)
            emoji_counts[label_name].update(emojis)
        
        # Find most common emojis per sentiment
        results = {}
        for label in self.evaluator.label_names:
            most_common = emoji_counts[label].most_common(10)
            results[label] = {
                'total_emojis': len(emoji_sentiment[label]),
                'unique_emojis': len(set(emoji_sentiment[label])),
                'most_common': most_common
            }
        
        return results
    
    def analyze_hashtag_sentiment(self, texts: List[str], predictions: np.ndarray) -> Dict:
        """
        Analyze sentiment patterns related to hashtag usage.
        
        Args:
            texts: Input texts
            predictions: Predicted labels
            
        Returns:
            Hashtag sentiment analysis results
        """
        import re
        
        hashtag_sentiment = {label: [] for label in self.evaluator.label_names}
        hashtag_counts = {label: Counter() for label in self.evaluator.label_names}
        
        for text, pred in zip(texts, predictions):
            label_name = self.evaluator.label_mapping[pred]
            
            # Extract hashtags
            hashtags = re.findall(r'#\w+', text.lower())
            hashtag_sentiment[label_name].extend(hashtags)
            hashtag_counts[label_name].update(hashtags)
        
        # Find most common hashtags per sentiment
        results = {}
        for label in self.evaluator.label_names:
            most_common = hashtag_counts[label].most_common(10)
            results[label] = {
                'total_hashtags': len(hashtag_sentiment[label]),
                'unique_hashtags': len(set(hashtag_sentiment[label])),
                'most_common': most_common
            }
        
        return results
    
    def analyze_text_features(self, texts: List[str], predictions: np.ndarray) -> Dict:
        """
        Analyze various text features and their correlation with sentiment.
        
        Args:
            texts: Input texts
            predictions: Predicted labels
            
        Returns:
            Text feature analysis results
        """
        import re
        
        features = {
            'length': [],
            'exclamation_count': [],
            'question_count': [],
            'caps_ratio': [],
            'url_count': [],
            'mention_count': [],
            'sentiment': []
        }
        
        for text, pred in zip(texts, predictions):
            features['length'].append(len(text.split()))
            features['exclamation_count'].append(text.count('!'))
            features['question_count'].append(text.count('?'))
            features['caps_ratio'].append(sum(1 for c in text if c.isupper()) / len(text) if text else 0)
            features['url_count'].append(len(re.findall(r'http[s]?://\S+', text)))
            features['mention_count'].append(len(re.findall(r'@\w+', text)))
            features['sentiment'].append(self.evaluator.label_mapping[pred])
        
        df = pd.DataFrame(features)
        
        # Calculate correlations and statistics
        results = {}
        for feature in ['length', 'exclamation_count', 'question_count', 'caps_ratio', 'url_count', 'mention_count']:
            results[feature] = {}
            for sentiment in self.evaluator.label_names:
                sentiment_data = df[df['sentiment'] == sentiment][feature]
                results[feature][sentiment] = {
                    'mean': float(sentiment_data.mean()),
                    'std': float(sentiment_data.std()),
                    'median': float(sentiment_data.median())
                }
        
        return results
    
    def create_sentiment_dashboard(self, texts: List[str], predictions: np.ndarray,
                                 probabilities: np.ndarray, save_path: Optional[str] = None):
        """
        Create an interactive dashboard for sentiment analysis results.
        
        Args:
            texts: Input texts
            predictions: Predicted labels
            probabilities: Prediction probabilities
            save_path: Path to save dashboard HTML
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sentiment Distribution', 'Confidence Distribution',
                          'Text Length vs Sentiment', 'Prediction Confidence'],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Sentiment distribution
        sentiment_counts = Counter([self.evaluator.label_mapping[p] for p in predictions])
        fig.add_trace(
            go.Bar(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()),
                   name='Sentiment Distribution'),
            row=1, col=1
        )
        
        # Confidence distribution
        confidences = np.max(probabilities, axis=1)
        fig.add_trace(
            go.Histogram(x=confidences, name='Confidence Distribution'),
            row=1, col=2
        )
        
        # Text length vs sentiment
        text_lengths = [len(text.split()) for text in texts]
        for i, sentiment in enumerate(self.evaluator.label_names):
            mask = predictions == i
            fig.add_trace(
                go.Box(y=np.array(text_lengths)[mask], name=f'{sentiment} Length'),
                row=2, col=1
            )
        
        # Prediction confidence scatter
        fig.add_trace(
            go.Scatter(
                x=text_lengths,
                y=confidences,
                mode='markers',
                marker=dict(color=predictions, colorscale='viridis'),
                name='Length vs Confidence'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Sentiment Analysis Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()

def create_evaluation_report(evaluator: CommentEvaluator, texts: List[str], 
                           true_labels: List[int], save_dir: str) -> None:
    """
    Create a comprehensive evaluation report.
    
    Args:
        evaluator: CommentEvaluator instance
        texts: Input texts
        true_labels: Ground truth labels
        save_dir: Directory to save report files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating evaluation report...")
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(texts, true_labels)
    
    # Generate plots
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    evaluator.plot_roc_curves(
        true_labels, results['probabilities'],
        os.path.join(save_dir, 'roc_curves.png')
    )
    
    evaluator.plot_prediction_confidence(
        results['probabilities'], results['predictions'],
        os.path.join(save_dir, 'confidence_distribution.png')
    )
    
    # Error analysis
    error_analysis = evaluator.analyze_errors(
        texts, true_labels, results['predictions'], results['probabilities']
    )
    
    # Length-based analysis
    length_analysis = evaluator.length_based_analysis(
        texts, true_labels, results['predictions']
    )
    
    # Save results to JSON
    import json
    
    # Prepare results for JSON serialization
    def to_serializable(val):
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    json_results = {
        'overall_metrics': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'roc_auc': float(results['roc_auc']) if results['roc_auc'] else None
        },
        'per_class_metrics': {
            'precision': [float(x) for x in results['per_class_metrics']['precision']],
            'recall': [float(x) for x in results['per_class_metrics']['recall']],
            'f1': [float(x) for x in results['per_class_metrics']['f1']],
            'support': [int(x) for x in results['per_class_metrics']['support']]
        },
        'error_analysis': json.loads(json.dumps(error_analysis, default=to_serializable)),
        'length_analysis': json.loads(json.dumps(length_analysis, default=to_serializable))
    }
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Evaluation report saved to {save_dir}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Overall F1-Score: {results['f1_score']:.4f}")
