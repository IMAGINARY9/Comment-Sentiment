"""
Training utilities for comment sentiment analysis.

This module provides comprehensive training infrastructure for both
transformer-based and traditional neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import time
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models import CommentDataset, TwitterRoBERTaModel, BiLSTMGloVeModel, EnsembleModel
from preprocessing import CommentPreprocessor, VocabularyBuilder, EmbeddingLoader

class CommentTrainer:
    """
    Comprehensive trainer for comment sentiment analysis models.
    
    This class handles training for various model types including
    transformer models, BiLSTM models, and ensemble approaches.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'auto'):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 5)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.patience = config.get('patience', 3)
        
        # Logging
        self.save_steps = config.get('save_steps', 500)
        self.eval_steps = config.get('eval_steps', 250)
        self.logging_steps = config.get('logging_steps', 50)
        
        # Initialize optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0
        self.best_model_state = None
        self.patience_counter = 0
    
    def _setup_optimizer_and_scheduler(self, train_dataloader: DataLoader) -> None:
        """Setup optimizer and learning rate scheduler."""
        
        # Check if model is transformer-based
        is_transformer = isinstance(self.model, (TwitterRoBERTaModel, EnsembleModel))
        
        if is_transformer:
            # Use AdamW for transformer models
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            # Use Adam for traditional models
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.num_epochs
        
        if is_transformer:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Use ReduceLROnPlateau for traditional models
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True
            )
    
    def _create_dataloaders(self, train_texts: List[str], train_labels: List[int],
                           val_texts: List[str], val_labels: List[int],
                           tokenizer=None, vocabulary=None) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        
        # Determine max_length based on model type
        max_length = self.config.get('max_length', 280 if tokenizer else 128)
        
        # Create datasets
        train_dataset = CommentDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = CommentDataset(val_texts, val_labels, tokenizer, max_length)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_dataloader, val_dataloader
    
    def _train_epoch(self, train_dataloader: DataLoader, epoch: int, tokenizer=None) -> Tuple[float, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            if tokenizer:
                # Transformer model
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, EnsembleModel):
                    # Ensemble model needs original texts
                    texts = [train_dataloader.dataset.texts[i] for i in range(len(labels))]
                    outputs = self.model(input_ids, attention_mask, texts, labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
            else:
                # Traditional model
                input_ids = batch['input_ids'].to(self.device)  # Token sequences
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Update scheduler for transformer models
            if isinstance(self.model, (TwitterRoBERTaModel, EnsembleModel)):
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
            
            # Logging
            if step % self.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if wandb.run:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/accuracy': current_acc,
                        'train/learning_rate': current_lr,
                        'train/step': epoch * len(train_dataloader) + step
                    })
        
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, val_dataloader: DataLoader, tokenizer=None) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                # Move batch to device
                if tokenizer:
                    # Transformer model
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    if isinstance(self.model, EnsembleModel):
                        # Ensemble model needs original texts
                        texts = [val_dataloader.dataset.texts[i] for i in range(len(labels))]
                        outputs = self.model(input_ids, attention_mask, texts, labels)
                    else:
                        outputs = self.model(input_ids, attention_mask, labels)
                else:
                    # Traditional model
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, labels)
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              tokenizer=None, vocabulary=None) -> Dict:
        """
        Train the model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            tokenizer: Tokenizer for transformer models
            vocabulary: Vocabulary for traditional models
            
        Returns:
            Training history dictionary
        """
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Create dataloaders
        train_dataloader, val_dataloader = self._create_dataloaders(
            train_texts, train_labels, val_texts, val_labels, tokenizer, vocabulary
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(train_dataloader)
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self._train_epoch(train_dataloader, epoch, tokenizer)
            
            # Validate
            val_loss, val_acc, val_f1 = self._validate_epoch(val_dataloader, tokenizer)
            
            # Update scheduler for traditional models
            if not isinstance(self.model, (TwitterRoBERTaModel, EnsembleModel)):
                self.scheduler.step(val_f1)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'val/epoch_loss': val_loss,
                    'val/epoch_accuracy': val_acc,
                    'val/epoch_f1': val_f1
                })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"New best model! F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model with F1: {self.best_val_f1:.4f}")
        
        return self.history
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_f1': self.best_val_f1
        }
        
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.best_val_f1 = checkpoint.get('best_val_f1', 0)
        
        print(f"Model loaded from {load_path}")

class MultiPlatformTrainer:
    """
    Trainer for multi-platform sentiment analysis.
    
    This class handles training models on data from multiple social media
    platforms and can perform domain adaptation.
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize multi-platform trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.trainers = {}
        
        # Platform-specific configurations
        self.platform_configs = {
            'twitter': {'max_length': 280, 'learning_rate': 2e-5},
            'airline': {'max_length': 300, 'learning_rate': 1e-5},
            'apple': {'max_length': 200, 'learning_rate': 2e-5},
            'general': {'max_length': 256, 'learning_rate': 2e-5}
        }
    
    def train_platform_specific(self, platform: str, train_data: Dict, val_data: Dict,
                               tokenizer=None) -> Dict:
        """
        Train model for specific platform.
        
        Args:
            platform: Platform name
            train_data: Training data dictionary
            val_data: Validation data dictionary
            tokenizer: Tokenizer for transformer models
            
        Returns:
            Training history
        """
        # Get platform-specific config
        platform_config = self.config.copy()
        platform_config.update(self.platform_configs.get(platform, {}))
        
        # Create trainer
        trainer = CommentTrainer(self.model, platform_config)
        
        # Train
        history = trainer.train(
            train_data['texts'], train_data['labels'],
            val_data['texts'], val_data['labels'],
            tokenizer
        )
        
        self.trainers[platform] = trainer
        
        return history
    
    def domain_adaptation(self, source_platform: str, target_platform: str,
                         source_data: Dict, target_data: Dict, tokenizer=None) -> Dict:
        """
        Perform domain adaptation between platforms.
        
        Args:
            source_platform: Source platform name
            target_platform: Target platform name
            source_data: Source platform data
            target_data: Target platform data
            tokenizer: Tokenizer
            
        Returns:
            Adaptation history
        """
        # First train on source domain
        print(f"Training on source domain: {source_platform}")
        source_history = self.train_platform_specific(
            source_platform, source_data['train'], source_data['val'], tokenizer
        )
        
        # Fine-tune on target domain with lower learning rate
        print(f"Fine-tuning on target domain: {target_platform}")
        adaptation_config = self.config.copy()
        adaptation_config.update(self.platform_configs.get(target_platform, {}))
        adaptation_config['learning_rate'] *= 0.1  # Lower learning rate for fine-tuning
        adaptation_config['num_epochs'] = max(3, adaptation_config['num_epochs'] // 2)
        
        adapter = CommentTrainer(self.model, adaptation_config)
        adaptation_history = adapter.train(
            target_data['train']['texts'], target_data['train']['labels'],
            target_data['val']['texts'], target_data['val']['labels'],
            tokenizer
        )
        
        return {
            'source_history': source_history,
            'adaptation_history': adaptation_history
        }

def train_with_config(config_path: str, data_paths: Dict[str, str], 
                     model_save_path: str, use_wandb: bool = False) -> None:
    """
    Train model using configuration file.
    
    Args:
        config_path: Path to configuration file
        data_paths: Dictionary mapping data types to paths
        model_save_path: Path to save trained model
        use_wandb: Whether to use Weights & Biases logging
    """
    import yaml
    from transformers import AutoTokenizer
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Training with config: {config_path}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="comment-sentiment",
            config=config,
            name=f"comment_sentiment_{config['model']['name'].replace('/', '_')}"
        )
    
    # Initialize components based on model type
    model_config = config['model']
    
    if 'roberta' in model_config.get('name', ''):
        # Transformer model
        from models import create_model
        
        model = create_model(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        
        # Initialize preprocessor
        preprocessor = CommentPreprocessor(config.get('preprocessing', {}))
        
        # Load and preprocess data
        from preprocessing import load_social_media_data
        
        platform = config['data'].get('platform', 'twitter')
        texts, labels = load_social_media_data(data_paths['train'], platform)
        
        # Preprocess texts
        texts = preprocessor.preprocess_batch(texts)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(train_texts, train_labels, val_texts, val_labels, tokenizer)
        
        # Save model
        trainer.save_model(model_save_path)
        
    else:
        # Traditional model (BiLSTM)
        from models import create_model
        from preprocessing import VocabularyBuilder, EmbeddingLoader
        
        # Initialize preprocessor
        preprocessor = CommentPreprocessor(config.get('preprocessing', {}))
        
        # Load and preprocess data
        from preprocessing import load_social_media_data
        
        platform = config['data'].get('platform', 'twitter')
        texts, labels = load_social_media_data(data_paths['train'], platform)
        
        # Preprocess texts
        texts = preprocessor.preprocess_batch(texts)
        
        # Build vocabulary
        vocabulary = VocabularyBuilder(
            max_vocab_size=config['data']['vocab_size'],
            min_freq=config['data']['min_freq']
        )
        vocabulary.build_vocabulary(texts)
        
        # Load embeddings if specified
        embedding_matrix = None
        if 'embeddings' in config:
            embedding_config = config['embeddings']
            embedding_loader = EmbeddingLoader(
                embedding_config['path'],
                embedding_config['type']
            )
            embedding_matrix = embedding_loader.create_embedding_matrix(vocabulary)
        
        # Create model with embedding matrix
        model_config['vocab_size'] = len(vocabulary.word2idx)
        model_config['embedding_matrix'] = embedding_matrix
        model = create_model(model_config)
        
        # Convert texts to sequences
        train_sequences = vocabulary.texts_to_sequences(texts, model_config['max_length'])
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(
            train_sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(train_seqs, train_labels, val_seqs, val_labels, vocabulary=vocabulary)
        
        # Save model and vocabulary
        trainer.save_model(model_save_path)
        vocabulary.save_vocabulary(model_save_path.replace('.pt', '_vocab.json'))
    
    print("Training completed successfully!")
    
    if use_wandb:
        wandb.finish()
