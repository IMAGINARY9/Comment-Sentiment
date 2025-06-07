"""
Comment Sentiment Analysis Package

This package provides tools for sentiment analysis of comments and social media text,
including preprocessing for informal language, emoji handling, and ensemble methods.
"""

__version__ = "0.1.0"
__author__ = "Sentiment Analysis Team"

# Only import and expose valid symbols
from .models import EnsembleModel
from .training import CommentTrainer
from .evaluation import CommentEvaluator

__all__ = [
    "EnsembleModel",
    "CommentTrainer",
    "CommentEvaluator"
]
