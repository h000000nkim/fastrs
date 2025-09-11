"""
Unit tests for the core Fastrs class.

Tests the main Fastrs class functionality including initialization,
preprocessing, training, dimensionality reduction, and visualization
methods in isolation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from gensim.models import FastText

from fastrs.core.object import Fastrs, Item
from fastrs.core.exceptions import FastrsError, TrainingError, ReducerError


class TestFastrsInitialization:
    """Test Fastrs class initialization with various input types."""

    def test_init_with_data_dict(self, sample_data):
        """
        Test Fastrs initialization with dictionary data.

        Parameters
        ----------
        sample_data : dict
            Sample educational assessment data from fixtures.
        """
        fastrs = Fastrs(data=sample_data)

        assert fastrs.data == sample_data
        assert hasattr(fastrs, "items")
        assert len(fastrs.items) == len(sample_data)
        assert all(isinstance(item, Item) for item in fastrs.items)

    def test_init_with_arrays(self):
        """Test Fastrs initialization with numpy arrays."""
        answers = np.array([["answer1", "answer2"], ["answer3", "answer4"]])
        responses = np.array([["response1", "response2"], ["response3", "response4"]])
        informations = np.array(["info1", "info2"])

        fastrs = Fastrs(answers=answers, responses=responses, informations=informations)

        assert hasattr(fastrs, "data")
        assert hasattr(fastrs, "items")
        assert len(fastrs.items) == 2

    def test_init_with_model(self, sample_data):
        """Test Fastrs initialization with pre-trained model."""
        mock_model = Mock(spec=FastText)

        fastrs = Fastrs(data=sample_data, model=mock_model)

        assert fastrs.model == mock_model

    def test_init_invalid_data(self):
        """Test Fastrs initialization with invalid data raises appropriate errors."""
        with pytest.raises(TypeError):
            Fastrs(answers="invalid", responses=None, informations=None)


class TestFastrsPreprocessing:
    """Test Fastrs preprocessing methods."""

    def test_preprocess_default(self, fastrs_instance):
        """
        Test default preprocessing pipeline.

        Parameters
        ----------
        fastrs_instance : Fastrs
            Fastrs instance with sample data loaded.
        """
        fastrs_instance.preprocess(option="default")

        assert hasattr(fastrs_instance, "feed")
        assert isinstance(fastrs_instance.feed, list)
        assert all(isinstance(sentence, list) for sentence in fastrs_instance.feed)

    def test_preprocess_custom_not_implemented(self, fastrs_instance):
        """Test that custom preprocessing raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            fastrs_instance.preprocess(option="custom")

    def test_clean_method(self, fastrs_instance):
        """Test the clean method with various parameters."""
        # Test default cleaning
        result = fastrs_instance.clean()
        assert isinstance(result, list)

        # Test with specific targets
        result = fastrs_instance.clean(target=["answer", "response"])
        assert isinstance(result, list)

    def test_tokenize_method(self, sample_data):
        """Test the tokenize method with different options."""
        # Test morphological tokenization with fresh instance
        fastrs_morphs = Fastrs(data=sample_data)
        result = fastrs_morphs.tokenize(option="morphs")
        assert isinstance(result, list)

        # Test noun extraction with fresh instance
        fastrs_nouns = Fastrs(data=sample_data)
        result = fastrs_nouns.tokenize(option="nouns")
        assert isinstance(result, list)

    def test_jamoize_method(self, fastrs_instance):
        """Test the jamoize method for Korean character decomposition."""
        result = fastrs_instance.jamoize()
        assert isinstance(result, list)

    def test_formatize_method(self, fastrs_instance):
        """Test the formatize method for creating training data."""
        # Mock preprocessed data
        fastrs_instance.items[0].answer = ["processed", "answer"]
        fastrs_instance.items[0].response = ["processed", "response"]

        result = fastrs_instance.formatize(
            iterables=["answer", "response"], anchor="information"
        )

        assert isinstance(result, list)
        assert hasattr(fastrs_instance, "feed")


class TestFastrsTraining:
    """Test Fastrs training methods."""

    @patch("fastrs.core.util.get_pretrained_model")
    def test_finetune_with_existing_model(self, mock_get_model, fastrs_with_feed):
        """Test fine-tuning with existing model."""
        mock_model = Mock(spec=FastText)
        mock_get_model.return_value = mock_model

        result = fastrs_with_feed.finetune()

        assert result == mock_model
        mock_model.build_vocab.assert_called_once()
        mock_model.train.assert_called_once()

    def test_train_new_model(self, fastrs_with_feed):
        """Test training a new FastText model."""
        with patch("fastrs.core.object.FastText") as mock_fasttext:
            mock_model = Mock()
            mock_fasttext.return_value = mock_model

            result = fastrs_with_feed.train(vector_size=100, epochs=5)

            assert result == mock_model
            mock_fasttext.assert_called_once()

    def test_train_without_feed_raises_error(self, fastrs_instance):
        """Test that training without preprocessed data raises TrainingError."""
        with pytest.raises(TrainingError):
            fastrs_instance.train()


class TestFastrsDimensionalityReduction:
    """Test Fastrs dimensionality reduction methods."""

    def test_reduce_umap(self, trained_fastrs):
        """Test UMAP dimensionality reduction."""
        result = trained_fastrs.reduce(method="umap", n_components=2)

        assert isinstance(result, pd.DataFrame)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "response" in result.columns
        assert "token" in result.columns
        assert hasattr(trained_fastrs, "coordinates")

    def test_reduce_pca(self, trained_fastrs):
        """Test PCA dimensionality reduction."""
        result = trained_fastrs.reduce(method="pca", n_components=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 4  # x, y, response, token

    def test_reduce_tsne(self, trained_fastrs):
        """Test t-SNE dimensionality reduction."""
        result = trained_fastrs.reduce(method="tsne", n_components=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 4

    def test_reduce_invalid_method(self, trained_fastrs):
        """Test that invalid reduction method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            trained_fastrs.reduce(method="invalid")

    def test_reduce_mapping_error(self, trained_fastrs):
        """Test ReducerError when token mapping fails."""
        # Mock jamodict to cause mapping failure
        trained_fastrs.jamodict = {}

        with pytest.raises(ReducerError):
            trained_fastrs.reduce(method="umap")


class TestFastrsVisualization:
    """Test Fastrs visualization methods."""

    def test_visualize_success(self, reduced_fastrs):
        """Test successful visualization creation."""
        result = reduced_fastrs.visualize()

        assert isinstance(result, list)
        assert hasattr(reduced_fastrs, "plot")
        assert len(result) == len(reduced_fastrs.items)

    def test_visualize_without_coordinates_raises_error(self, fastrs_instance):
        """Test that visualization without coordinates raises FastrsError."""
        with pytest.raises(FastrsError, match="No reduced coordinates found"):
            fastrs_instance.visualize()


# Fixtures specific to this test module


@pytest.fixture
def fastrs_with_feed(fastrs_instance):
    """Provide a Fastrs instance with preprocessed feed data."""
    # Mock the feed data
    fastrs_instance.feed = [["word1", "word2"], ["word3", "word4"]]
    return fastrs_instance


@pytest.fixture
def trained_fastrs(fastrs_with_feed):
    """Provide a Fastrs instance with a trained model."""
    mock_model = Mock(spec=FastText)
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(50, 100)  # 50 > 30 for t-SNE perplexity
    mock_model.wv.key_to_index = {f"word{i}": i for i in range(50)}

    fastrs_with_feed.model = mock_model
    fastrs_with_feed.jamodict = {f"word{i}": f"response{i}" for i in range(50)}

    return fastrs_with_feed


@pytest.fixture
def reduced_fastrs(trained_fastrs):
    """Provide a Fastrs instance with reduced coordinates."""
    # Mock coordinates
    trained_fastrs.coordinates = pd.DataFrame(
        {
            "response": ["response1", "response2"],
            "token": ["word1", "word2"],
            "x": [0.1, 0.2],
            "y": [0.3, 0.4],
        }
    )

    # Mock item coordinates
    for item in trained_fastrs.items:
        item.coordinates = trained_fastrs.coordinates.copy()

    return trained_fastrs
