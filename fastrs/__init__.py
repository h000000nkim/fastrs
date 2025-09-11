"""
FastRS - FastText-based text analysis and visualization package

A comprehensive toolkit for text preprocessing, FastText model training,
and interactive embedding visualization.
"""

from .core.object import Fastrs
from .core import preprocessor
from .core.visualizer import scatter
from .core import util
from .core import exceptions

__all__ = ["Fastrs", "preprocessor", "scatter", "util", "exceptions"]

__version__ = "0.1.0"
