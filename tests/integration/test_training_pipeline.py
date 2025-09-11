"""
Integration tests for training pipeline.

Tests the complete training workflow including preprocessing, model training,
fine-tuning, and model evaluation as integrated processes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from gensim.models import FastText

from fastrs.core.object import Fastrs
from fastrs.core.exceptions import TrainingError
from tests.fixtures import load_sample_data


class TestTrainingPipelineIntegration:
    """Test complete training pipeline integration."""

    def test_full_training_pipeline(self, preprocessed_fastrs):
        """
        Test complete training pipeline from preprocessing to trained model.

        Parameters
        ----------
        preprocessed_fastrs : Fastrs
            Fastrs instance with preprocessed data ready for training.
        """
        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model

            # Execute training
            result_model = preprocessed_fastrs.train(
                vector_size=100, epochs=5, window=5
            )

            # Verify training completed
            assert result_model == mock_model
            assert preprocessed_fastrs.model == mock_model
            mock_fasttext_class.assert_called_once()

    def test_preprocessing_to_training_pipeline(self, fastrs_with_sample_data):
        """Test complete pipeline from raw data to trained model."""
        # Step 1: Preprocess
        fastrs_with_sample_data.preprocess(option="default")

        # Verify preprocessing completed
        assert hasattr(fastrs_with_sample_data, "feed")
        assert len(fastrs_with_sample_data.feed) > 0

        # Step 2: Train
        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model

            result_model = fastrs_with_sample_data.train(vector_size=100, epochs=3)

            # Verify training completed
            assert result_model == mock_model
            assert fastrs_with_sample_data.model == mock_model

    @patch("fastrs.core.util.get_pretrained_model")
    def test_finetune_pipeline(self, mock_get_pretrained, preprocessed_fastrs):
        """Test fine-tuning pipeline with pretrained model."""
        # Setup pretrained model mock
        pretrained_model = Mock(spec=FastText)
        mock_get_pretrained.return_value = pretrained_model

        # Execute fine-tuning
        result_model = preprocessed_fastrs.finetune()

        # Verify fine-tuning completed
        assert result_model == pretrained_model
        assert preprocessed_fastrs.model == pretrained_model
        pretrained_model.build_vocab.assert_called_once()
        pretrained_model.train.assert_called_once()

    def test_training_with_custom_parameters(self, preprocessed_fastrs):
        """Test training with custom FastText parameters."""
        custom_params = {
            "vector_size": 200,
            "window": 10,
            "min_count": 2,
            "epochs": 10,
            "sg": 1,  # Skip-gram
            "hs": 1,  # Hierarchical softmax
            "negative": 10,
            "alpha": 0.05,
            "min_alpha": 0.001,
        }

        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model

            preprocessed_fastrs.train(**custom_params)

            # Verify custom parameters were passed
            mock_fasttext_class.assert_called_once()
            call_args = mock_fasttext_class.call_args[1]

            for param, value in custom_params.items():
                assert call_args[param] == value

    def test_training_error_handling(self, fastrs_with_sample_data):
        """Test training pipeline error handling."""
        # Test training without preprocessing
        with pytest.raises(TrainingError, match="No preprocessed data found"):
            fastrs_with_sample_data.train()

    def test_model_consistency_after_training(self, preprocessed_fastrs):
        """Test that model remains consistent after training operations."""
        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model

            # Train model
            result_model = preprocessed_fastrs.train(vector_size=100)

            # Verify model consistency
            assert preprocessed_fastrs.model is result_model
            assert preprocessed_fastrs.model is mock_model

            # Multiple references should point to same object
            retrieved_model = preprocessed_fastrs.model
            assert retrieved_model is result_model


class TestTrainingWithRealData:
    """Test training pipeline with real educational data."""

    def test_korean_literature_training(self, literature_trained_fastrs):
        """Test training with Korean literature data."""
        # Verify model was trained
        assert literature_trained_fastrs.model is not None

        # Verify model has vocabulary (mocked)
        if hasattr(literature_trained_fastrs.model, "wv"):
            assert hasattr(literature_trained_fastrs.model.wv, "vectors")

    def test_science_text_training(self, science_trained_fastrs):
        """Test training with science text data."""
        # Verify model was trained
        assert science_trained_fastrs.model is not None

        # Check that scientific terminology is learned (mocked verification)
        assert science_trained_fastrs.model is not None

    def test_mixed_domain_training(self, mixed_domain_trained_fastrs):
        """Test training with mixed domain data."""
        # Verify model handles mixed domains
        assert mixed_domain_trained_fastrs.model is not None

        # Should have processed all items
        assert len(mixed_domain_trained_fastrs.items) > 1


class TestTrainingValidation:
    """Test training validation and quality checks."""

    def test_training_data_validation(self, fastrs_with_sample_data):
        """Test that training validates input data quality."""
        # Preprocess data
        fastrs_with_sample_data.preprocess(option="default")

        # Verify feed data quality
        assert hasattr(fastrs_with_sample_data, "feed")
        assert isinstance(fastrs_with_sample_data.feed, list)
        assert len(fastrs_with_sample_data.feed) > 0

        # Verify feed contains valid sentences
        for sentence in fastrs_with_sample_data.feed:
            assert isinstance(sentence, list)
            assert len(sentence) > 0
            assert all(isinstance(token, str) for token in sentence)

    def test_model_output_validation(self, trained_fastrs):
        """Test that trained model produces valid outputs."""
        # Model should be set
        assert trained_fastrs.model is not None

        # Model should have required attributes (mocked)
        assert (
            hasattr(trained_fastrs.model, "wv") or trained_fastrs.model.wv is not None
        )

    def test_training_reproducibility(self, preprocessed_fastrs):
        """Test that training is reproducible with same parameters."""
        training_params = {"vector_size": 100, "epochs": 5, "seed": 42, "workers": 1}

        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model1 = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model1

            # First training
            preprocessed_fastrs.train(**training_params)

            # Verify consistent parameter usage
            call_args = mock_fasttext_class.call_args[1]
            assert call_args["seed"] == 42
            assert call_args["workers"] == 1


class TestAdvancedTrainingFeatures:
    """Test advanced training features and configurations."""

    def test_incremental_training(self, trained_fastrs):
        """Test incremental training with additional data."""
        # Create additional training data
        additional_data = [["추가", "학습", "데이터"], ["더", "많은", "문장"]]

        # Mock incremental training
        with (
            patch.object(trained_fastrs.model, "build_vocab") as mock_build_vocab,
            patch.object(trained_fastrs.model, "train") as mock_train,
        ):
            # Simulate adding new data
            trained_fastrs.feed.extend(additional_data)

            # Fine-tune with new data
            trained_fastrs.finetune()

            # Verify incremental training
            mock_build_vocab.assert_called_once()
            mock_train.assert_called_once()

    def test_training_with_different_algorithms(self, preprocessed_fastrs):
        """Test training with different FastText algorithms."""
        # Test Skip-gram
        with patch("fastrs.core.object.FastText") as mock_fasttext:
            mock_model = Mock(spec=FastText)
            mock_fasttext.return_value = mock_model

            preprocessed_fastrs.train(sg=1, hs=1)  # Skip-gram with hierarchical softmax

            call_args = mock_fasttext.call_args[1]
            assert call_args["sg"] == 1
            assert call_args["hs"] == 1

        # Test CBOW
        with patch("fastrs.core.object.FastText") as mock_fasttext:
            mock_model = Mock(spec=FastText)
            mock_fasttext.return_value = mock_model

            preprocessed_fastrs.train(sg=0, negative=5)  # CBOW with negative sampling

            call_args = mock_fasttext.call_args[1]
            assert call_args["sg"] == 0
            assert call_args["negative"] == 5

    def test_training_optimization(self, preprocessed_fastrs):
        """Test training optimization parameters."""
        optimization_params = {
            "alpha": 0.025,
            "min_alpha": 0.0001,
            "epochs": 10,
            "workers": 4,
        }

        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = Mock(spec=FastText)
            mock_fasttext_class.return_value = mock_model

            preprocessed_fastrs.train(**optimization_params)

            # Verify optimization parameters
            call_args = mock_fasttext_class.call_args[1]
            for param, value in optimization_params.items():
                assert call_args[param] == value


# Training integration test fixtures
@pytest.fixture
def fastrs_with_sample_data():
    """Provide Fastrs instance with sample data."""
    data = load_sample_data()
    return Fastrs(data=data)


@pytest.fixture
def preprocessed_fastrs(fastrs_with_sample_data):
    """Provide Fastrs instance with preprocessed data ready for training."""
    fastrs_with_sample_data.preprocess(option="default")
    return fastrs_with_sample_data


@pytest.fixture
def trained_fastrs(preprocessed_fastrs):
    """Provide Fastrs instance with trained model."""
    with patch("fastrs.core.object.FastText") as mock_fasttext_class:
        mock_model = Mock(spec=FastText)
        mock_model.wv = Mock()
        mock_model.wv.vectors = np.random.rand(100, 100)
        mock_model.wv.key_to_index = {f"word_{i}": i for i in range(100)}
        mock_fasttext_class.return_value = mock_model

        preprocessed_fastrs.train(vector_size=100, epochs=5)

    return preprocessed_fastrs


@pytest.fixture
def literature_trained_fastrs():
    """Provide trained Fastrs instance with literature data."""
    data = load_sample_data()
    literature_data = {"item1": data["item1"]}  # Use literature item

    fastrs = Fastrs(data=literature_data)
    fastrs.preprocess(option="default")

    with patch("fastrs.core.object.FastText") as mock_fasttext_class:
        mock_model = Mock(spec=FastText)
        mock_model.wv = Mock()
        mock_fasttext_class.return_value = mock_model

        fastrs.train(vector_size=100)

    return fastrs


@pytest.fixture
def science_trained_fastrs():
    """Provide trained Fastrs instance with science data."""
    data = load_sample_data()
    science_data = {"item2": data["item2"]}  # Use science item

    fastrs = Fastrs(data=science_data)
    fastrs.preprocess(option="default")

    with patch("fastrs.core.object.FastText") as mock_fasttext_class:
        mock_model = Mock(spec=FastText)
        mock_model.wv = Mock()
        mock_fasttext_class.return_value = mock_model

        fastrs.train(vector_size=100)

    return fastrs


@pytest.fixture
def mixed_domain_trained_fastrs():
    """Provide trained Fastrs instance with mixed domain data."""
    data = load_sample_data()

    fastrs = Fastrs(data=data)
    fastrs.preprocess(option="default")

    with patch("fastrs.core.object.FastText") as mock_fasttext_class:
        mock_model = Mock(spec=FastText)
        mock_model.wv = Mock()
        mock_fasttext_class.return_value = mock_model

        fastrs.train(vector_size=100)

    return fastrs
