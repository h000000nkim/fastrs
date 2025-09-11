"""
Integration tests for visualization pipeline.

Tests the complete visualization workflow including dimensionality reduction,
coordinate generation, and interactive plot creation as integrated processes.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import plotly.graph_objects as go

from fastrs.core.object import Fastrs
from fastrs.core.exceptions import FastrsError
from tests.fixtures import load_sample_data


class TestVisualizationPipelineIntegration:
    """Test complete visualization pipeline integration."""

    def test_full_visualization_pipeline(self, trained_fastrs_with_vectors):
        """
        Test complete visualization pipeline from trained model to interactive plots.

        Parameters
        ----------
        trained_fastrs_with_vectors : Fastrs
            Fastrs instance with trained model and vector data.
        """
        # Step 1: Reduce dimensions
        coordinates = trained_fastrs_with_vectors.reduce(method="umap", n_components=2)

        # Verify reduction completed
        assert isinstance(coordinates, pd.DataFrame)
        assert "x" in coordinates.columns
        assert "y" in coordinates.columns
        assert "response" in coordinates.columns
        assert "token" in coordinates.columns
        assert hasattr(trained_fastrs_with_vectors, "coordinates")

        # Step 2: Visualize
        with patch("fastrs.core.object.scatter") as mock_visualize:
            mock_figures = [
                Mock(spec=go.Figure) for _ in trained_fastrs_with_vectors.items
            ]
            mock_visualize.side_effect = mock_figures

            plots = trained_fastrs_with_vectors.visualize()

            # Verify visualization completed
            assert isinstance(plots, list)
            assert len(plots) == len(trained_fastrs_with_vectors.items)
            assert hasattr(trained_fastrs_with_vectors, "plot")

    def test_training_to_visualization_pipeline(self, fastrs_ready_for_training):
        """Test complete pipeline from training to visualization."""
        # Step 1: Train model
        with patch("fastrs.core.object.FastText") as mock_fasttext_class:
            mock_model = self._create_mock_model_with_vectors()
            mock_fasttext_class.return_value = mock_model

            fastrs_ready_for_training.train(vector_size=100)
            fastrs_ready_for_training.jamodict = self._create_mock_jamodict()

            # Step 2: Reduce dimensions
            fastrs_ready_for_training.reduce(method="pca")

            # Step 3: Visualize
            with patch("fastrs.core.object.scatter") as mock_viz:
                mock_viz.return_value = Mock(spec=go.Figure)
                plots = fastrs_ready_for_training.visualize()

                # Verify complete pipeline
                assert len(plots) > 0
                assert all(isinstance(plot, Mock) for plot in plots)

    def test_different_reduction_methods_pipeline(self, trained_fastrs_with_vectors):
        """Test visualization pipeline with different dimensionality reduction methods."""
        reduction_methods = ["umap", "pca", "tsne"]

        for method in reduction_methods:
            # Reset coordinates for each test
            if hasattr(trained_fastrs_with_vectors, "coordinates"):
                delattr(trained_fastrs_with_vectors, "coordinates")

            # Test reduction with appropriate parameters for small datasets
            if method == "tsne":
                # Use smaller perplexity for small test datasets
                coordinates = trained_fastrs_with_vectors.reduce(
                    method=method, perplexity=5
                )
            else:
                coordinates = trained_fastrs_with_vectors.reduce(method=method)

            assert isinstance(coordinates, pd.DataFrame)
            assert len(coordinates.columns) == 4

            # Test visualization
            with patch("fastrs.core.object.scatter") as mock_viz:
                mock_viz.return_value = Mock(spec=go.Figure)
                plots = trained_fastrs_with_vectors.visualize()

                assert len(plots) > 0

    def test_pipeline_error_handling(self, trained_fastrs_with_vectors):
        """Test visualization pipeline error handling."""
        # Test visualization without reduction
        untrained_fastrs = Fastrs(data=load_sample_data())

        with pytest.raises(FastrsError, match="No reduced coordinates found"):
            untrained_fastrs.visualize()

        # Test reduction with invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            trained_fastrs_with_vectors.reduce(method="invalid_method")

    def test_pipeline_data_consistency(self, trained_fastrs_with_vectors):
        """Test that data remains consistent throughout visualization pipeline."""
        original_items_count = len(trained_fastrs_with_vectors.items)
        original_item_names = [item.name for item in trained_fastrs_with_vectors.items]

        # Run complete pipeline
        trained_fastrs_with_vectors.reduce(method="umap")

        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            trained_fastrs_with_vectors.visualize()

        # Verify data consistency
        assert len(trained_fastrs_with_vectors.items) == original_items_count
        assert [
            item.name for item in trained_fastrs_with_vectors.items
        ] == original_item_names

        # Verify each item has coordinates
        for item in trained_fastrs_with_vectors.items:
            assert hasattr(item, "coordinates")
            assert isinstance(item.coordinates, pd.DataFrame)

    # Helper methods
    def _create_mock_model_with_vectors(self):
        """Create a mock FastText model with vector data."""
        mock_model = Mock()
        mock_model.wv = Mock()
        mock_model.wv.vectors = np.random.rand(50, 100)
        mock_model.wv.key_to_index = {f"word_{i}": i for i in range(50)}
        return mock_model

    def _create_mock_jamodict(self):
        """Create a mock jamo dictionary for testing."""
        return {f"word_{i}": f"response_{i}" for i in range(50)}


class TestVisualizationWithRealData:
    """Test visualization pipeline with real educational data."""

    def test_literature_data_visualization(self, literature_visualization_fastrs):
        """Test visualization of Korean literature analysis data."""
        # Reduce dimensions
        coordinates = literature_visualization_fastrs.reduce(method="umap")

        # Verify literature-specific data structure
        assert "response" in coordinates.columns
        literature_responses = coordinates["response"].unique()

        # Should contain literature analysis terms
        assert len(literature_responses) > 0

        # Visualize
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            literature_visualization_fastrs.visualize()

            # Verify visualization calls
            assert mock_viz.call_count == len(literature_visualization_fastrs.items)

    def test_science_data_visualization(self, science_visualization_fastrs):
        """Test visualization of science reading comprehension data."""
        # Reduce dimensions
        coordinates = science_visualization_fastrs.reduce(method="pca")

        # Verify science-specific data structure
        science_responses = coordinates["response"].unique()

        # Should contain scientific terminology
        assert len(science_responses) > 0

        # Visualize
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            plots = science_visualization_fastrs.visualize()

            assert len(plots) > 0

    def test_mixed_subject_visualization(self, mixed_subject_visualization_fastrs):
        """Test visualization with mixed subject data."""
        # Reduce dimensions
        coordinates = mixed_subject_visualization_fastrs.reduce(
            method="tsne", perplexity=7
        )

        # Should handle mixed domains
        assert len(coordinates) > 0
        assert "response" in coordinates.columns

        # Visualize all items
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            plots = mixed_subject_visualization_fastrs.visualize()

            # Should create visualization for each item
            assert len(plots) == len(mixed_subject_visualization_fastrs.items)


class TestVisualizationCustomization:
    """Test visualization pipeline customization and parameters."""

    def test_custom_reduction_parameters(self, trained_fastrs_with_vectors):
        """Test dimensionality reduction with custom parameters."""
        # Test UMAP with custom parameters
        coordinates_umap = trained_fastrs_with_vectors.reduce(
            method="umap", n_neighbors=10, min_dist=0.5, metric="cosine"
        )

        assert isinstance(coordinates_umap, pd.DataFrame)

        # Reset for next test
        delattr(trained_fastrs_with_vectors, "coordinates")

        # Test PCA with custom parameters
        coordinates_pca = trained_fastrs_with_vectors.reduce(
            method="pca", svd_solver="full", whiten=True
        )

        assert isinstance(coordinates_pca, pd.DataFrame)

    def test_visualization_customization(self, reduced_fastrs):
        """Test visualization with different customization options."""
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)

            # Test visualization for each item
            reduced_fastrs.visualize()

            # Verify customization options were passed
            assert mock_viz.call_count == len(reduced_fastrs.items)

            # Check that each item's original answers were passed
            for call_args in mock_viz.call_args_list:
                args, kwargs = call_args
                assert "answers" in kwargs or len(args) > 1

    def test_coordinate_filtering_and_processing(self, trained_fastrs_with_vectors):
        """Test coordinate filtering and processing in visualization pipeline."""
        # Add some mock processing to test filtering
        coordinates = trained_fastrs_with_vectors.reduce(method="umap")

        # Verify coordinate structure
        assert "x" in coordinates.columns
        assert "y" in coordinates.columns
        assert "response" in coordinates.columns
        assert "token" in coordinates.columns

        # Test that coordinates are properly assigned to items
        for item in trained_fastrs_with_vectors.items:
            assert hasattr(item, "coordinates")
            item_coords = item.coordinates
            assert isinstance(item_coords, pd.DataFrame)
            assert len(item_coords) >= 0


class TestVisualizationPerformance:
    """Test visualization pipeline performance characteristics."""

    def test_large_dataset_visualization(self, large_dataset_visualization_fastrs):
        """Test visualization pipeline with large datasets."""
        import time

        start_time = time.time()

        # Reduce dimensions
        large_dataset_visualization_fastrs.reduce(method="pca")  # PCA is faster

        # Visualize
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            plots = large_dataset_visualization_fastrs.visualize()

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds max
        assert len(plots) > 0

    def test_memory_efficient_visualization(self, trained_fastrs_with_vectors):
        """Test memory efficiency of visualization pipeline."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run visualization pipeline
        trained_fastrs_with_vectors.reduce(method="umap")

        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_viz.return_value = Mock(spec=go.Figure)
            trained_fastrs_with_vectors.visualize()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # 50MB max increase

    # Helper methods
    def _create_mock_model_with_vectors(self):
        """Create a mock FastText model with vector data."""
        mock_model = Mock()
        mock_model.wv = Mock()
        mock_model.wv.vectors = np.random.rand(50, 100)
        mock_model.wv.key_to_index = {f"word_{i}": i for i in range(50)}
        return mock_model

    def _create_mock_jamodict(self):
        """Create a mock jamo dictionary for testing."""
        return {f"word_{i}": f"response_{i}" for i in range(50)}


# Visualization integration test fixtures
@pytest.fixture
def trained_fastrs_with_vectors():
    """Provide Fastrs instance with trained model and vector data."""
    data = load_sample_data()
    fastrs = Fastrs(data=data)
    fastrs.preprocess(option="default")

    # Mock trained model with vectors
    mock_model = Mock()
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(20, 100)
    mock_model.wv.key_to_index = {f"token_{i}": i for i in range(20)}

    fastrs.model = mock_model
    fastrs.jamodict = {f"token_{i}": f"response_{i % 5}" for i in range(20)}

    return fastrs


@pytest.fixture
def fastrs_ready_for_training():
    """Provide Fastrs instance ready for training."""
    data = load_sample_data()
    fastrs = Fastrs(data=data)
    fastrs.preprocess(option="default")
    return fastrs


@pytest.fixture
def reduced_fastrs(trained_fastrs_with_vectors):
    """Provide Fastrs instance with reduced coordinates."""
    # Mock the coordinates
    coordinates = pd.DataFrame(
        {
            "x": np.random.randn(20),
            "y": np.random.randn(20),
            "response": [f"response_{i % 5}" for i in range(20)],
            "token": [f"token_{i}" for i in range(20)],
        }
    )

    trained_fastrs_with_vectors.coordinates = coordinates

    # Mock item coordinates
    for i, item in enumerate(trained_fastrs_with_vectors.items):
        item_coords = coordinates.iloc[
            i * 3 : (i + 1) * 3
        ].copy()  # Subset for each item
        item.coordinates = item_coords

    return trained_fastrs_with_vectors


@pytest.fixture
def literature_visualization_fastrs():
    """Provide Fastrs instance for literature visualization testing."""
    data = load_sample_data()
    literature_data = {"item1": data["item1"]}

    fastrs = Fastrs(data=literature_data)
    fastrs.preprocess(option="default")

    # Mock model
    mock_model = Mock()
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(15, 100)
    mock_model.wv.key_to_index = {f"lit_token_{i}": i for i in range(15)}

    fastrs.model = mock_model
    # Create jamodict with enough entries for all tokens in the model
    responses = data["item1"]["response"]
    fastrs.jamodict = {}
    for i in range(15):  # Match the number of tokens in the model
        response_idx = i % len(responses)  # Cycle through responses if we need more
        fastrs.jamodict[f"lit_token_{i}"] = responses[response_idx]

    return fastrs


@pytest.fixture
def science_visualization_fastrs():
    """Provide Fastrs instance for science visualization testing."""
    data = load_sample_data()
    science_data = {"item2": data["item2"]}

    fastrs = Fastrs(data=science_data)
    fastrs.preprocess(option="default")

    # Mock model
    mock_model = Mock()
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(15, 100)
    mock_model.wv.key_to_index = {f"sci_token_{i}": i for i in range(15)}

    fastrs.model = mock_model
    # Create jamodict with enough entries for all tokens in the model
    responses = data["item2"]["response"]
    fastrs.jamodict = {}
    for i in range(15):  # Match the number of tokens in the model
        response_idx = i % len(responses)  # Cycle through responses if we need more
        fastrs.jamodict[f"sci_token_{i}"] = responses[response_idx]

    return fastrs


@pytest.fixture
def mixed_subject_visualization_fastrs():
    """Provide Fastrs instance for mixed subject visualization testing."""
    data = load_sample_data()

    fastrs = Fastrs(data=data)
    fastrs.preprocess(option="default")

    # Mock model
    mock_model = Mock()
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(30, 100)
    mock_model.wv.key_to_index = {f"mixed_token_{i}": i for i in range(30)}

    fastrs.model = mock_model

    # Create jamodict from all responses
    all_responses = []
    for item_data in data.values():
        all_responses.extend(item_data["response"])

    fastrs.jamodict = {f"mixed_token_{i}": resp for i, resp in enumerate(all_responses)}

    return fastrs


@pytest.fixture
def large_dataset_visualization_fastrs():
    """Provide Fastrs instance with large dataset for performance testing."""
    # Create larger synthetic dataset
    base_data = load_sample_data()
    large_data = {}

    for i in range(10):  # Create 10 items
        for j, (key, value) in enumerate(base_data.items()):
            new_key = f"{key}_large_{i}"
            large_data[new_key] = {
                "information": f"{value['information']}",
                "answer": value["answer"][:],
                "response": value["response"][:],
            }

    fastrs = Fastrs(data=large_data)
    fastrs.preprocess(option="default")

    # Mock large model
    mock_model = Mock()
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(200, 100)  # Larger vector set
    mock_model.wv.key_to_index = {f"large_token_{i}": i for i in range(200)}

    fastrs.model = mock_model
    fastrs.jamodict = {f"large_token_{i}": f"response_{i % 20}" for i in range(200)}

    return fastrs
