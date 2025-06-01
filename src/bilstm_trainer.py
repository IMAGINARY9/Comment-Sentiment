"""
BiLSTM training logic for comment sentiment analysis.
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time
from models import create_model

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
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {'input_ids': input_ids_padded, 'label': labels}


def train_bilstm(texts, labels, model_config, training_config, evaluate=False):
    max_length = model_config.get('max_length', 128)
    sequences = [t.split()[:max_length] for t in texts]
    word2idx = build_word2idx(sequences)
    sequences = [[word2idx[token] for token in seq] for seq in sequences]
    model_config['vocab_size'] = len(word2idx) + 1
    model = create_model(model_config)
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_dataset = SequenceDataset(train_seqs, train_labels)
    val_dataset = SequenceDataset(val_seqs, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, collate_fn=collate_fn)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    print("Starting BiLSTM training...")
    total_start_time = time.time()
    for epoch in range(training_config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{training_config['num_epochs']}", ncols=120)
        for batch_idx, batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/labels.size(0):.4f}',
                'avg_acc': f'{avg_acc:.4f}'
            })
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_correct / total_samples
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{training_config['num_epochs']}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Time: {epoch_time:.1f}s")
    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time/60:.2f} min ({total_time:.1f} sec)")
    date_str = datetime.now().strftime("%Y%m%d")
    model_save_path = f"models/comment_sentiment_bilstm_{date_str}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    if evaluate:
        print("Running evaluation on validation set...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids)
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1-score: {f1:.4f}")
        print(classification_report(all_labels, all_preds))
        return model, val_seqs, val_labels, word2idx
    return None
