"""
Training utilities for comment sentiment analysis.

This module provides comprehensive training infrastructure for both
transformer-based and traditional neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import time
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV

from models import CommentDataset, TwitterRoBERTaModel, BiLSTMGloVeModel, EnsembleModel, create_model

# --- BiLSTM utilities (from bilstm_trainer.py) ---
def build_word2idx(sequences):
    word2idx = {}
    idx = 1
    for seq in sequences:
        for token in seq:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
    return word2idx

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

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {'input_ids': input_ids_padded, 'label': labels}

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
        import torch
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.batch_size = config['training'].get('batch_size', 32)
        self.learning_rate = float(config['training'].get('learning_rate', 2e-5))
        self.num_epochs = config['training'].get('num_epochs', 5)
        self.warmup_steps = config['training'].get('warmup_steps', 100)
        self.weight_decay = float(config['training'].get('weight_decay', 0.01))
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.patience = config['training'].get('patience', 3)
        
        # Logging
        self.save_steps = config['training'].get('save_steps', 500)
        self.eval_steps = config['training'].get('eval_steps', 250)
        self.logging_steps = config['training'].get('logging_steps', 50)
        
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
        
        # Class weights
        self.class_weights = None
        if 'class_weights' in config['training']:
            import torch
            self.class_weights = torch.tensor(config['training']['class_weights'], dtype=torch.float32).to(self.device)
    
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
                patience=2
            )
    
    def _create_dataloaders(self, train_texts: List[str], train_labels: List[int],
                           val_texts: List[str], val_labels: List[int],
                           tokenizer=None, vocabulary=None) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        # If using BiLSTM, use SequenceDataset and collate_fn
        if isinstance(self.model, BiLSTMGloVeModel):
            max_length = self.config['training'].get('max_length', 128) # Use configured max_length

            # Tokenize texts into words
            train_tokenized_texts = [text.split() for text in train_texts]
            val_tokenized_texts = [text.split() for text in val_texts]

            if vocabulary is None:
                # Build word2idx from actual word tokens from the training set
                word2idx = build_word2idx(train_tokenized_texts)
            else:
                word2idx = vocabulary
            self.word2idx = word2idx # Store the word-based vocabulary

            # Convert word tokens to indices, truncating at the word level
            train_sequences = [[self.word2idx.get(token, 0) for token in t_text[:max_length]] for t_text in train_tokenized_texts]
            val_sequences = [[self.word2idx.get(token, 0) for token in t_text[:max_length]] for t_text in val_tokenized_texts]
            
            train_dataset = SequenceDataset(train_sequences, train_labels)
            val_dataset = SequenceDataset(val_sequences, val_labels)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            return train_dataloader, val_dataloader
        
        # Determine max_length based on model type
        max_length = self.config['training'].get('max_length', 280 if tokenizer else 128)
        
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
        
        # Define criterion with class weights if available
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

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
                    outputs = self.model(input_ids, attention_mask, texts) # Pass labels to criterion
                else:
                    outputs = self.model(input_ids, attention_mask) # Pass labels to criterion
                
                logits = outputs['logits']
                loss = criterion(logits, labels) # Calculate loss using criterion

            else:
                # Traditional model
                input_ids = batch['input_ids'].to(self.device)  # Token sequences
                labels = batch['label'].to(self.device)
                logits = self.model(input_ids)
                if isinstance(logits, dict):
                    logits = logits['logits']
                loss = criterion(logits, labels) # Calculate loss using criterion
            
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
        
        # Define criterion with class weights if available
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                # Move batch to device
                if tokenizer:
                    # Transformer model
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    if isinstance(self.model, EnsembleModel):
                        texts = [val_dataloader.dataset.texts[i] for i in range(len(labels))]
                        outputs = self.model(input_ids, attention_mask, texts)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                    
                    logits = outputs['logits']
                    loss = criterion(logits, labels) # Calculate loss using criterion
                else:
                    # Traditional model
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(input_ids)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                    loss = criterion(logits, labels) # Calculate loss using criterion

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1

    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              tokenizer=None, vocabulary=None, output_dir: str = 'model_output'):
        """Main training loop."""
        
        # Create dataloaders
        train_dataloader, val_dataloader = self._create_dataloaders(
            train_texts, train_labels, val_texts, val_labels, tokenizer, vocabulary
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(train_dataloader)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_dataloader, epoch, tokenizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._validate_epoch(val_dataloader, tokenizer)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Time: {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'val/epoch_loss': val_loss,
                    'val/epoch_accuracy': val_acc,
                    'val/epoch_f1': val_f1,
                    'epoch_time': epoch_time
                })
            
            # Update scheduler for traditional models (ReduceLROnPlateau)
            if not isinstance(self.model, (TwitterRoBERTaModel, EnsembleModel)):
                self.scheduler.step(val_f1) # Use val_f1 for ReduceLROnPlateau
            
            # Early stopping and best model saving
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
                # Save best model checkpoint
                self.save_model(os.path.join(output_dir, 'best_model.pt'), epoch=epoch, val_f1=val_f1)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model with Val F1: {self.best_val_f1:.4f}")

        # Platt Scaling for probability calibration
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        print("Performing Platt Scaling for probability calibration...")
        self.model.eval()
        all_logits = []
        all_labels_calib = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Calibration Data Collection'):
                if tokenizer:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    if isinstance(self.model, EnsembleModel):
                        texts = [val_dataloader.dataset.texts[i] for i in range(len(labels))]
                        outputs = self.model(input_ids, attention_mask, texts)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                    logits = outputs['logits']
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(input_ids)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                
                all_logits.append(logits.cpu().numpy())
                all_labels_calib.append(labels.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels_calib = np.concatenate(all_labels_calib, axis=0)
        
        num_classes = all_logits.shape[1]
        self.calibration_params = {'coef': [], 'intercept': []}
        
        for i in range(num_classes):
            # One-vs-rest calibration
            binary_labels = (all_labels_calib == i).astype(int)
            lr = LogisticRegression(solver='liblinear', class_weight='balanced') # Added class_weight
            lr.fit(all_logits[:, i].reshape(-1, 1), binary_labels) # Calibrate on single logit
            self.calibration_params['coef'].append(lr.coef_[0][0])
            self.calibration_params['intercept'].append(lr.intercept_[0])
        
        print("Platt Scaling complete. Calibration parameters learned.")

        print("Training complete.")
        return self.history

    def evaluate(self, test_texts: List[str], test_labels: List[int], tokenizer=None, vocabulary=None) -> Dict[str, float]:
        """Evaluate the model on test data."""
        
        # Create test dataloader
        # Note: Using a dummy list for train_texts/labels as they are not used for test dataloader creation
        _, test_dataloader = self._create_dataloaders(
            [], [], test_texts, test_labels, tokenizer, vocabulary
        )
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        # Define criterion with class weights if available
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Evaluating'):
                if tokenizer:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    if isinstance(self.model, EnsembleModel):
                        texts = [test_dataloader.dataset.texts[i] for i in range(len(labels))]
                        outputs = self.model(input_ids, attention_mask, texts)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                    
                    logits = outputs['logits']
                    loss = criterion(logits, labels) # Calculate loss using criterion
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(input_ids)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                    loss = criterion(logits, labels) # Calculate loss using criterion

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        print(f"Test Loss: {avg_loss:.4f} - Test Acc: {accuracy:.4f} - Test F1: {f1:.4f}")
        print(classification_report(all_labels, all_predictions))
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1': f1, 'report': report}

    def save_model(self, path: str, epoch: Optional[int] = None, val_f1: Optional[float] = None, original_config: Optional[Dict] = None):
        """Save the trained model, config, vocabulary, and training history."""
        import datetime
        import yaml
        import json
        import os
        import pickle

        # Determine the base directory for saving all artifacts
        # If path is a directory, append 'model.pt' to it
        if os.path.isdir(path) or (not os.path.splitext(path)[1]):
            base_dir = path
            model_path = os.path.join(base_dir, 'model.pt')
        else:
            base_dir = os.path.dirname(path)
            model_path = path
        os.makedirs(base_dir, exist_ok=True) # Ensure the directory exists

        # Save model state dict to the specified path
        torch.save(self.model.state_dict(), model_path)
        print(f"Model state_dict saved to {model_path}")

        # Save the original config (with any runtime overrides) as config.yaml in the same directory
        config_to_save = original_config if original_config is not None else self.config
        config_filename = 'config.yaml' # Default config filename
        # If saving a specific checkpoint like 'best_model.pt', name config accordingly
        model_filename_stem = os.path.splitext(os.path.basename(path))[0]
        if model_filename_stem != 'model': # e.g. if path is 'best_model.pt'
            config_filename = f"{model_filename_stem}_config.yaml"
        
        config_path = os.path.join(base_dir, config_filename) 
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_to_save, f)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
            # Fallback: save with a generic name if specific naming failed, though less likely
            fallback_config_path = os.path.join(base_dir, 'config_fallback.yaml')
            try:
                with open(fallback_config_path, 'w') as f:
                    yaml.dump(config_to_save, f)
                print(f"Configuration saved to {fallback_config_path} (fallback)")
            except Exception as e_fb:
                print(f"Error saving config to {fallback_config_path} (fallback): {e_fb}")

        # Save training history as npz
        history_path = os.path.join(base_dir, 'history.npz')
        np.savez(history_path, **self.history)
        print(f"Training history saved to {history_path}")

        # Save best_val_f1 as a text file
        best_f1_path = os.path.join(base_dir, 'best_val_f1.txt')
        with open(best_f1_path, 'w') as f:
            f.write(str(self.best_val_f1))
        print(f"Best validation F1 score ({self.best_val_f1}) saved to {best_f1_path}")

        # Save vocabulary if BiLSTM (word2idx)
        if hasattr(self, 'word2idx') and self.word2idx is not None: # Check self.word2idx directly
            vocab_json_path = os.path.join(base_dir, 'word2idx.json')
            with open(vocab_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
            
            vocab_pkl_path = os.path.join(base_dir, 'word2idx.pkl')
            with open(vocab_pkl_path, 'wb') as f:
                pickle.dump(self.word2idx, f)
            print(f"Vocabulary saved to {vocab_json_path} and {vocab_pkl_path}")

        # Save calibration parameters if present
        if hasattr(self, 'calibration_params') and self.calibration_params is not None:
            calib_path = os.path.join(base_dir, 'calibration_params.yaml')
            with open(calib_path, 'w') as f:
                yaml.dump(self.calibration_params, f)
            print(f"Calibration parameters saved to {calib_path}")
        
        print(f"All artifacts related to {os.path.basename(path)} saved in {base_dir}")
    
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
        
        # Load and preprocess data
        from preprocessing import load_social_media_data
        
        platform = config['data'].get('platform', 'twitter')
        texts, labels = load_social_media_data(data_paths['train'], platform)
        
        # Train model
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(texts, labels, texts, labels, tokenizer, output_dir=model_save_path)
        
        # Save model
        trainer.save_model(model_save_path, original_config=config)
        
    else:
        # Traditional model (BiLSTM)
        from models import create_model
        
        # Load and preprocess data
        from preprocessing import load_social_media_data
        
        platform = config['data'].get('platform', 'twitter')
        texts, labels = load_social_media_data(data_paths['train'], platform)
        
        # Create model
        model = create_model(model_config)
        
        # Convert texts to sequences
        train_sequences = texts  # Already preprocessed
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(
            train_sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        trainer = CommentTrainer(model, config['training'])
        history = trainer.train(train_seqs, train_labels, val_seqs, val_labels, output_dir=model_save_path)
        
        # Save model
        trainer.save_model(model_save_path, original_config=config)
    
    print("Training completed successfully!")
    
    if use_wandb:
        wandb.finish()

def build_vocabulary_for_bilstm(train_texts: List[str], config: Dict) -> Tuple[Dict[str, int], int]:
    """
    Build vocabulary for BiLSTM model from training texts.
    
    Args:
        train_texts: List of training text strings
        config: Configuration dictionary
        
    Returns:
        Tuple of (word2idx dictionary, vocabulary size)
    """
    print("Building vocabulary for BiLSTM model...")
    
    # Tokenize texts into words
    train_tokenized_texts = [text.split() for text in train_texts]
    
    # Build vocabulary
    word2idx = build_word2idx(train_tokenized_texts)
    vocab_size = len(word2idx)
    
    print(f"Vocabulary built with {vocab_size} words")
    print(f"Sample words: {list(word2idx.keys())[:10]}")
    
    return word2idx, vocab_size
