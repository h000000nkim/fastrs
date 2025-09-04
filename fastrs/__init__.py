"""
FastRS - FastText-based text analysis and visualization package

A comprehensive toolkit for text preprocessing, FastText model training,
and interactive embedding visualization.
"""

from .core.object import Fastrs
from .core.preprocessor import Preprocessor
from .core.visualizer import visualize_embeddings
from .core import util
from .core import exceptions

__all__ = [
    'Fastrs',
    'Preprocessor',
    'visualize_embeddings', 
    'util',
    'exceptions'
]

__version__ = '0.1.0'
