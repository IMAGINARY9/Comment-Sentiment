"""
Preprocessing utilities for comment sentiment analysis.

This module provides text cleaning and preprocessing functions
specifically designed for social media content and short comments.
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import emoji
from transformers import AutoTokenizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class CommentPreprocessor:
    """
    Comprehensive preprocessor for social media comments and short text.
    
    This class handles various aspects of social media text preprocessing
    including emoji handling, URL cleaning, mention processing, and more.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing options
        """
        self.config = config or {}
        
        # Default settings
        self.clean_text = self.config.get('clean_text', True)
        self.handle_mentions = self.config.get('handle_mentions', True)
        self.handle_hashtags = self.config.get('handle_hashtags', True)
        self.handle_urls = self.config.get('handle_urls', True)
        self.handle_emojis = self.config.get('handle_emojis', 'convert')  # convert, remove, keep
        self.expand_contractions = self.config.get('expand_contractions', True)
        self.correct_spelling = self.config.get('correct_spelling', False)
        self.remove_stopwords = self.config.get('remove_stopwords', False)
        self.normalize_case = self.config.get('normalize_case', True)
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self._init_contraction_mapping()
        
        if self.correct_spelling:
            try:
                from textblob import TextBlob
                self.spell_checker = TextBlob
            except ImportError:
                print("Warning: TextBlob not available for spell checking")
                self.correct_spelling = False
    
    def _init_contraction_mapping(self):
        """Initialize contraction expansion mapping."""
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "might've": "might have", "mightn't": "might not",
            "must've": "must have", "mustn't": "must not",
            "needn't": "need not", "o'clock": "of the clock",
            "oughtn't": "ought not", "shan't": "shall not",
            "sha'n't": "shall not", "she'd": "she would",
            "she'll": "she will", "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "that'd": "that would",
            "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who's": "who is",
            "who've": "who have", "won't": "will not", "wouldn't": "would not",
            "you'd": "you would", "you'll": "you will", "you're": "you are",
            "you've": "you have", "'re": " are", "n't": " not", "'ve": " have",
            "'ll": " will", "'d": " would"
        }
    
    def clean_basic_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation that might be meaningful
        text = re.sub(r'[^\w\s!?.,;:\'"@#-]', ' ', text)
        
        return text
    
    def handle_urls(self, text: str) -> str:
        """
        Handle URLs in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs processed
        """
        if not self.handle_urls:
            return text
        
        # Replace URLs with placeholder
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, ' <URL> ', text)
        
        # Handle www links
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(www_pattern, ' <URL> ', text)
        
        return text
    
    def handle_mentions(self, text: str) -> str:
        """
        Handle @mentions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with mentions processed
        """
        if not self.handle_mentions:
            return text
        
        # Replace mentions with placeholder
        text = re.sub(r'@[A-Za-z0-9_]+', ' <USER> ', text)
        
        return text
    
    def handle_hashtags(self, text: str) -> str:
        """
        Handle hashtags in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with hashtags processed
        """
        if not self.handle_hashtags:
            return text
        
        # Extract text from hashtags (remove # but keep the word)
        text = re.sub(r'#([A-Za-z0-9_]+)', r' \1 ', text)
        
        return text
    
    def handle_emojis(self, text: str) -> str:
        """
        Handle emojis based on configuration.
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis processed
        """
        if self.handle_emojis == 'remove':
            # Remove all emojis
            text = emoji.demojize(text)
            text = re.sub(r':[a-z_&+-]+:', '', text)
        elif self.handle_emojis == 'convert':
            # Convert emojis to text descriptions
            text = emoji.demojize(text)
            text = re.sub(r':', ' ', text)
        # If 'keep', do nothing
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with contractions expanded
        """
        if not self.expand_contractions:
            return text
        
        # Convert to lowercase for matching
        lower_text = text.lower()
        
        for contraction, expansion in self.contractions.items():
            lower_text = lower_text.replace(contraction, expansion)
        
        return lower_text
    
    def correct_spelling(self, text: str) -> str:
        """
        Correct spelling errors.
        
        Args:
            text: Input text
            
        Returns:
            Text with corrected spelling
        """
        if not self.correct_spelling:
            return text
        
        try:
            blob = self.spell_checker(text)
            return str(blob.correct())
        except:
            return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        if not self.remove_stopwords:
            return text
        
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(filtered_words)
    
    def normalize_case(self, text: str) -> str:
        """
        Normalize text case.
        
        Args:
            text: Input text
            
        Returns:
            Case-normalized text
        """
        if not self.normalize_case:
            return text
        
        return text.lower()
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Apply preprocessing steps in order
        if self.clean_text:
            text = self.clean_basic_text(text)
        
        text = self.handle_urls(text)
        text = self.handle_mentions(text)
        text = self.handle_hashtags(text)
        text = self.handle_emojis(text)
        
        if self.expand_contractions:
            text = self.expand_contractions(text)
        
        if self.correct_spelling:
            text = self.correct_spelling(text)
        
        if self.normalize_case:
            text = self.normalize_case(text)
        
        if self.remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]

class VocabularyBuilder:
    """
    Build vocabulary for traditional models (BiLSTM, etc.).
    
    This class creates word-to-index mappings and handles
    out-of-vocabulary words for non-transformer models.
    """
    
    def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
        """
        Initialize vocabulary builder.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for word inclusion
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of training texts
        """
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_counts.update(words)
        
        # Filter by frequency and select top words
        filtered_words = [
            word for word, count in self.word_counts.items() 
            if count >= self.min_freq
        ]
        
        # Sort by frequency and take top words
        most_common = self.word_counts.most_common(self.max_vocab_size - len(self.special_tokens))
        vocab_words = [word for word, _ in most_common if word in filtered_words]
        
        # Create mappings
        vocab = self.special_tokens + vocab_words
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Built vocabulary with {len(self.word2idx)} words")
    
    def texts_to_sequences(self, texts: List[str], max_length: int = 128) -> List[List[int]]:
        """
        Convert texts to sequences of token IDs.
        
        Args:
            texts: List of texts
            max_length: Maximum sequence length
            
        Returns:
            List of token ID sequences
        """
        sequences = []
        
        for text in texts:
            words = text.split()
            sequence = [
                self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) 
                for word in words
            ]
            
            # Pad or truncate
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence = sequence + [self.word2idx[self.PAD_TOKEN]] * (max_length - len(sequence))
            
            sequences.append(sequence)
        
        return sequences
    
    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts),
            'config': {
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq
            }
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocabulary(self, path: str) -> None:
        """Load vocabulary from file."""
        import json
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.word_counts = Counter(vocab_data['word_counts'])
        
        config = vocab_data.get('config', {})
        self.max_vocab_size = config.get('max_vocab_size', self.max_vocab_size)
        self.min_freq = config.get('min_freq', self.min_freq)

class EmbeddingLoader:
    """
    Load pre-trained word embeddings (GloVe, Word2Vec, FastText).
    
    This class handles loading various types of pre-trained embeddings
    and creating embedding matrices for neural networks.
    """
    
    def __init__(self, embedding_path: str, embedding_type: str = 'glove'):
        """
        Initialize embedding loader.
        
        Args:
            embedding_path: Path to embedding file
            embedding_type: Type of embeddings ('glove', 'word2vec', 'fasttext')
        """
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type.lower()
        self.embeddings = {}
        self.embedding_dim = None
    
    def load_embeddings(self) -> None:
        """Load embeddings from file."""
        print(f"Loading {self.embedding_type} embeddings from {self.embedding_path}")
        
        if self.embedding_type == 'glove':
            self._load_glove()
        elif self.embedding_type in ['word2vec', 'fasttext']:
            self._load_word2vec_format()
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
        
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embedding_dim}")
    
    def _load_glove(self) -> None:
        """Load GloVe embeddings."""
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                
                if self.embedding_dim is None:
                    self.embedding_dim = len(vector)
                
                self.embeddings[word] = vector
    
    def _load_word2vec_format(self) -> None:
        """Load Word2Vec or FastText format embeddings."""
        try:
            from gensim.models import KeyedVectors
            
            if self.embedding_type == 'word2vec':
                model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
            else:  # fasttext
                model = KeyedVectors.load_word2vec_format(self.embedding_path)
            
            self.embedding_dim = model.vector_size
            self.embeddings = {word: model[word] for word in model.index_to_key}
            
        except ImportError:
            print("Warning: gensim not available, trying manual loading")
            self._load_glove()  # Fallback to GloVe format
    
    def create_embedding_matrix(self, vocabulary: VocabularyBuilder) -> np.ndarray:
        """
        Create embedding matrix for vocabulary.
        
        Args:
            vocabulary: VocabularyBuilder instance
            
        Returns:
            Embedding matrix as numpy array
        """
        if not self.embeddings:
            self.load_embeddings()
        
        vocab_size = len(vocabulary.word2idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        found = 0
        for word, idx in vocabulary.word2idx.items():
            if word in self.embeddings:
                embedding_matrix[idx] = self.embeddings[word]
                found += 1
            else:
                # Initialize with random values for unknown words
                embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)
        
        print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.1f}%)")
        
        return embedding_matrix

def load_social_media_data(data_path: str, platform: str = 'twitter') -> Tuple[List[str], List[int]]:
    """
    Load social media sentiment data.
    
    Args:
        data_path: Path to data directory
        platform: Platform type ('twitter', 'airline', 'apple', etc.)
        
    Returns:
        Tuple of (texts, labels)
    """
    import os
    
    texts, labels = [], []
    
    # Platform-specific loading
    if platform == 'twitter':
        # Try different Twitter data formats
        for filename in ['tweets.csv', 'twitter_sentiment.csv', 'training.csv']:
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                break
        else:
            raise FileNotFoundError(f"No Twitter data found in {data_path}")
            
    elif platform == 'airline':
        filepath = os.path.join(data_path, 'Tweets.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"No airline data found in {data_path}")
            
    elif platform == 'apple':
        filepath = os.path.join(data_path, 'apple_sentiment.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"No Apple data found in {data_path}")
    
    else:
        # Generic loading
        for filename in ['data.csv', 'sentiment.csv', 'comments.csv']:
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                break
        else:
            raise FileNotFoundError(f"No data found in {data_path}")
    
    # Extract texts and labels
    text_columns = ['text', 'tweet', 'comment', 'message', 'content']
    label_columns = ['sentiment', 'label', 'target', 'emotion']
    
    text_col = None
    label_col = None
    
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    for col in label_columns:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text and label columns in {list(df.columns)}")
    
    texts = df[text_col].astype(str).tolist()
    
    # Handle different label formats
    if df[label_col].dtype == 'object':
        # Convert string labels to integers
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = [label_mapping.get(str(label).lower(), 1) for label in df[label_col]]
    else:
        # Assume numeric labels
        labels = df[label_col].tolist()
    
    print(f"Loaded {len(texts)} samples from {platform} data")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return texts, labels
