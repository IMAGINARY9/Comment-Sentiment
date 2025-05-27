"""
Comment sentiment analysis models.

This module provides various neural network models for sentiment analysis
of social media comments and short text, including transformer-based models
and traditional neural networks with word embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class CommentDataset(Dataset):
    """Dataset class for comment sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # For transformer models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For traditional models, return text and label
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }

class TwitterRoBERTaModel(nn.Module):
    """
    RoBERTa-based model fine-tuned for Twitter sentiment analysis.
    
    This model uses a pre-trained Twitter RoBERTa model and adds
    additional layers for social media-specific sentiment classification.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Additional layers for fine-tuning
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
            
        return result
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions

class BiLSTMGloVeModel(nn.Module):
    """
    Bidirectional LSTM model with GloVe embeddings for comment sentiment analysis.
    
    This model combines pre-trained word embeddings with bidirectional LSTM
    for capturing sequential patterns in social media text.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 128,
                 num_layers: int = 2, num_labels: int = 3, dropout: float = 0.3,
                 embedding_matrix: Optional[np.ndarray] = None, trainable_embeddings: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = trainable_embeddings
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        x = self.dropout(attended)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss
            
        return result
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        with torch.no_grad():
            outputs = self.forward(input_ids)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions

class EnsembleModel(nn.Module):
    """
    Ensemble model combining transformer model with lexicon-based approaches.
    
    This model combines predictions from a neural model with traditional
    sentiment analysis methods like VADER and TextBlob for robust predictions.
    """
    
    def __init__(self, transformer_model: nn.Module, 
                 transformer_weight: float = 0.5,
                 vader_weight: float = 0.3,
                 textblob_weight: float = 0.2):
        super().__init__()
        
        self.transformer_model = transformer_model
        self.transformer_weight = transformer_weight
        self.vader_weight = vader_weight
        self.textblob_weight = textblob_weight
        
        # Import lexicon analyzers
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader_analyzer = None
            print("Warning: VADER not available")
        
        try:
            from textblob import TextBlob
            self.textblob_available = True
        except ImportError:
            self.textblob_available = False
            print("Warning: TextBlob not available")
    
    def _get_vader_scores(self, texts: List[str]) -> torch.Tensor:
        """Get VADER sentiment scores."""
        if self.vader_analyzer is None:
            return torch.zeros(len(texts), 3)
        
        scores = []
        for text in texts:
            sentiment = self.vader_analyzer.polarity_scores(text)
            # Convert to negative, neutral, positive probabilities
            compound = sentiment['compound']
            if compound >= 0.05:
                score = [0.0, 0.0, 1.0]  # positive
            elif compound <= -0.05:
                score = [1.0, 0.0, 0.0]  # negative
            else:
                score = [0.0, 1.0, 0.0]  # neutral
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def _get_textblob_scores(self, texts: List[str]) -> torch.Tensor:
        """Get TextBlob sentiment scores."""
        if not self.textblob_available:
            return torch.zeros(len(texts), 3)
        
        from textblob import TextBlob
        
        scores = []
        for text in texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to class probabilities
            if polarity > 0.1:
                score = [0.0, 0.0, 1.0]  # positive
            elif polarity < -0.1:
                score = [1.0, 0.0, 0.0]  # negative
            else:
                score = [0.0, 1.0, 0.0]  # neutral
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                texts: List[str], labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble model.
        
        Args:
            input_ids: Token IDs for transformer
            attention_mask: Attention mask for transformer
            texts: Original text for lexicon analysis
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing ensemble logits and optionally loss
        """
        # Get transformer predictions
        transformer_outputs = self.transformer_model(input_ids, attention_mask)
        transformer_probs = F.softmax(transformer_outputs['logits'], dim=-1)
        
        # Get lexicon scores
        vader_probs = self._get_vader_scores(texts).to(transformer_probs.device)
        textblob_probs = self._get_textblob_scores(texts).to(transformer_probs.device)
        
        # Combine predictions
        ensemble_probs = (
            self.transformer_weight * transformer_probs +
            self.vader_weight * vader_probs +
            self.textblob_weight * textblob_probs
        )
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        result = {'logits': ensemble_logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(ensemble_logits, labels)
            result['loss'] = loss
            
        return result
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                texts: List[str]) -> torch.Tensor:
        """Make ensemble predictions."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, texts)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions

def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create sentiment analysis models.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_type = config.get('type', 'transformer')
    
    if model_type == 'transformer' or 'roberta' in config.get('name', ''):
        return TwitterRoBERTaModel(
            model_name=config.get('name', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            num_labels=config.get('num_labels', 3),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'bilstm':
        return BiLSTMGloVeModel(
            vocab_size=config.get('vocab_size', 20000),
            embedding_dim=config.get('embedding_dim', 300),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_labels=config.get('num_labels', 3),
            dropout=config.get('dropout', 0.3)
        )
    
    elif model_type == 'ensemble':
        # Create base transformer model first
        transformer_model = TwitterRoBERTaModel(
            model_name=config.get('name', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            num_labels=config.get('num_labels', 3),
            dropout=config.get('dropout', 0.1)
        )
        
        return EnsembleModel(
            transformer_model=transformer_model,
            transformer_weight=config.get('transformer_weight', 0.5),
            vader_weight=config.get('vader_weight', 0.3),
            textblob_weight=config.get('textblob_weight', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class CommentSentimentPredictor:
    """
    High-level predictor class for comment sentiment analysis.
    
    This class provides a simple interface for loading models and making
    predictions on social media comments and short text.
    """
    
    def __init__(self, model_path: str, tokenizer_name: str, device: str = 'auto'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model
            tokenizer_name: Name or path of tokenizer
            device: Device to run inference on
        """
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        if tokenizer_name:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        
        # Label mapping
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def predict_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.tokenizer:
            encoding = self.tokenizer(
                text, truncation=True, padding=True, 
                max_length=280, return_tensors='pt'
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = F.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
        else:
            # For models without tokenizer (like BiLSTM), need vocabulary mapping
            # This would require additional preprocessing steps
            raise NotImplementedError("Text prediction for non-transformer models not implemented")
        
        return {
            'sentiment': self.label_mapping[prediction],
            'confidence': confidence,
            'probabilities': {
                'negative': probs[0][0].item(),
                'neutral': probs[0][1].item(),
                'positive': probs[0][2].item()
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict_text(text) for text in texts]
