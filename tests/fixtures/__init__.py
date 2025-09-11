"""
Test fixtures and sample data for FastRS package testing.

This module provides sample data and utilities for testing the FastRS
text similarity analysis functionality.
"""

__all__ = ["load_sample_data", "SAMPLE_DATA_PATH"]

import json
from pathlib import Path

# Path to sample data files
FIXTURE_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = FIXTURE_DIR / "sample_data.json"


def load_sample_data():
    """
    Load sample test data from JSON file.

    Returns
    -------
    dict
        Dictionary containing sample educational assessment data with
        questions, correct answers, and student responses.
    """
    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def create_minimal_data():
    """
    Create minimal test data for basic functionality tests.

    Returns
    -------
    dict
        Minimal data dictionary for simple tests.
    """
    return {
        "item1": {
            "information": "Test question about basic concepts.",
            "answer": ["correct", "answer"],
            "response": ["correct", "answer", "wrong", "response"],
        }
    }
