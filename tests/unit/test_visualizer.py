"""
Unit tests for visualization functions.

Tests the visualization functionality including embedding visualization,
plot creation, and interactive figure generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import plotly.graph_objects as go

from fastrs.core.visualizer import scatter


class TestScatter:
    """Test the scatter function."""

    def test_scatter_basic(self):
        """Test basic embedding visualization functionality."""
        # Create sample coordinate data
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "response": ["허무", "공허", "무상", "허무"],
                "token": ["token1", "token2", "token3", "token4"],
            }
        )

        answers = ["허무"]

        result = scatter(coordinates, answers=answers)

        assert isinstance(result, go.Figure)

    def test_scatter_with_title(self):
        """Test visualization with custom title."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["답변1", "답변2"],
                "token": ["tok1", "tok2"],
            }
        )

        custom_title = "Custom Test Title"
        result = scatter(coordinates, answers=["답변1"], title=custom_title)

        assert isinstance(result, go.Figure)
        # Check if title is set (depends on implementation)

    def test_scatter_show_false(self):
        """Test visualization with show=False parameter."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["resp1", "resp2"],
                "token": ["tok1", "tok2"],
            }
        )

        with patch("plotly.graph_objects.Figure.show") as mock_show:
            result = scatter(coordinates, answers=["resp1"], show=False)

            mock_show.assert_not_called()
            assert isinstance(result, go.Figure)

    def test_scatter_show_true(self):
        """Test visualization with show=True parameter."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["resp1", "resp2"],
                "token": ["tok1", "tok2"],
            }
        )

        with patch("plotly.graph_objects.Figure.show") as mock_show:
            result = scatter(coordinates, answers=["resp1"], show=True)

            mock_show.assert_called_once()
            assert isinstance(result, go.Figure)

    def test_scatter_multiple_answers(self):
        """Test visualization with multiple correct answers."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "response": ["허무", "공허", "허무", "무상", "절망"],
                "token": ["tok1", "tok2", "tok3", "tok4", "tok5"],
            }
        )

        answers = ["허무", "공허"]

        result = scatter(coordinates, answers=answers)

        assert isinstance(result, go.Figure)

    def test_scatter_empty_coordinates(self):
        """Test visualization with empty coordinate data."""
        coordinates = pd.DataFrame(columns=["x", "y", "response", "token"])

        with pytest.raises(ValueError, match="Answers list cannot be empty"):
            scatter(coordinates, answers=[])

    def test_scatter_missing_columns(self):
        """Test visualization with missing required columns."""
        # Missing 'y' column
        coordinates = pd.DataFrame(
            {"x": [1.0, 2.0], "response": ["resp1", "resp2"], "token": ["tok1", "tok2"]}
        )

        with pytest.raises(KeyError):
            scatter(coordinates, answers=["resp1"])

    def test_scatter_nan_values(self):
        """Test visualization with NaN values in coordinates."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, np.nan, 3.0],
                "y": [1.0, 2.0, np.nan],
                "response": ["resp1", "resp2", "resp3"],
                "token": ["tok1", "tok2", "tok3"],
            }
        )

        result = scatter(coordinates, answers=["resp1"])

        assert isinstance(result, go.Figure)

    def test_scatter_duplicate_responses(self):
        """Test visualization with duplicate response values."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "response": ["same", "same", "different", "same"],
                "token": ["tok1", "tok2", "tok3", "tok4"],
            }
        )
        result = scatter(coordinates, answers=["same"])
        assert isinstance(result, go.Figure)

    def test_scatter_large_dataset(self):
        """Test visualization with large dataset."""
        n_points = 1000
        coordinates = pd.DataFrame(
            {
                "x": np.random.randn(n_points),
                "y": np.random.randn(n_points),
                "response": [f"resp_{i}" for i in range(n_points)],
                "token": [f"tok_{i}" for i in range(n_points)],
            }
        )

        result = scatter(coordinates, answers=["resp_0", "resp_1"])

        assert isinstance(result, go.Figure)

    def test_scatter_special_characters(self):
        """Test visualization with special characters in responses."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "response": ["!@#$%", "한국어답변", "English_Answer"],
                "token": ["tok1", "tok2", "tok3"],
            }
        )

        result = scatter(coordinates, answers=["!@#$%", "한국어답변"])

        assert isinstance(result, go.Figure)

    def test_scatter_extreme_coordinates(self):
        """Test visualization with extreme coordinate values."""
        coordinates = pd.DataFrame(
            {
                "x": [-1000.0, 0.0, 1000.0],
                "y": [-1000.0, 0.0, 1000.0],
                "response": ["극소", "중간", "극대"],
                "token": ["tok1", "tok2", "tok3"],
            }
        )

        result = scatter(coordinates, answers=["중간"])

        assert isinstance(result, go.Figure)


class TestVisualizationIntegration:
    """Test visualization integration with other components."""

    @patch("fastrs.core.object.scatter")
    def test_integration_with_item_visualize(self, mock_visualize):
        """Test integration with Item.visualize method."""
        from fastrs.core.object import Item

        # Setup mock
        mock_figure = Mock(spec=go.Figure)
        mock_visualize.return_value = mock_figure

        # Create item with coordinates
        item = Item(
            name="test_item",
            answer=["correct"],
            response=["student_resp"],
            information="Test info",
        )

        item.coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["correct", "student_resp"],
                "token": ["tok1", "tok2"],
            }
        )
        item.original_answer = ["correct"]

        result = item.visualize()

        mock_visualize.assert_called_once()
        assert result == mock_figure

    def test_visualization_data_consistency(self):
        """Test that visualization maintains data consistency."""
        # Create coordinates that match realistic FastRS output
        coordinates = pd.DataFrame(
            {
                "x": [0.1, 0.2, 0.15, 0.25],
                "y": [0.3, 0.4, 0.35, 0.45],
                "response": ["허무", "공허", "허무감", "무상"],
                "token": ["허무토큰", "공허토큰", "허무감토큰", "무상토큰"],
            }
        )

        answers = ["허무"]

        result = scatter(coordinates, answers=answers, show=False)

        # Verify the figure was created successfully
        assert isinstance(result, go.Figure)

        # Check that data structure is preserved
        assert len(coordinates) == 4
        assert "x" in coordinates.columns
        assert "y" in coordinates.columns
        assert "response" in coordinates.columns
        assert "token" in coordinates.columns


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    def test_visualize_with_invalid_coordinates_type(self):
        """Test visualization with invalid coordinates type."""
        invalid_coordinates = "not_a_dataframe"

        with pytest.raises(AttributeError):
            scatter(invalid_coordinates, answers=["test"])

    def test_visualize_with_invalid_answers_type(self):
        """Test visualization with invalid answers type."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["resp1", "resp2"],
                "token": ["tok1", "tok2"],
            }
        )

        # This should handle gracefully or raise appropriate error
        try:
            result = scatter(coordinates, answers="not_a_list")
            assert isinstance(result, go.Figure)
        except (TypeError, ValueError):
            pass  # Expected behavior

    def test_visualize_with_none_answers(self):
        """Test visualization with None answers."""
        coordinates = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "response": ["resp1", "resp2"],
                "token": ["tok1", "tok2"],
            }
        )

        with pytest.raises(ValueError, match="Answers cannot be None"):
            scatter(coordinates, answers=None)


# Fixtures for visualization tests
@pytest.fixture
def sample_coordinates():
    """Provide sample coordinate data for testing."""
    return pd.DataFrame(
        {
            "x": np.random.randn(50),
            "y": np.random.randn(50),
            "response": [f"response_{i % 10}" for i in range(50)],
            "token": [f"token_{i}" for i in range(50)],
        }
    )


@pytest.fixture
def korean_literature_coordinates():
    """Provide Korean literature-specific coordinate data."""
    return pd.DataFrame(
        {
            "x": [0.1, 0.15, 0.2, 0.12, 0.18, 0.25, 0.3, 0.22],
            "y": [0.3, 0.35, 0.4, 0.32, 0.38, 0.45, 0.5, 0.42],
            "response": [
                "허무",
                "공허",
                "무상",
                "허무감",
                "초월",
                "절망",
                "체념",
                "냉소",
            ],
            "token": [
                "허무토큰",
                "공허토큰",
                "무상토큰",
                "허무감토큰",
                "초월토큰",
                "절망토큰",
                "체념토큰",
                "냉소토큰",
            ],
        }
    )


@pytest.fixture
def science_coordinates():
    """Provide science reading-specific coordinate data."""
    return pd.DataFrame(
        {
            "x": [0.5, 0.52, 0.48, 0.53, 0.47, 0.55, 0.45],
            "y": [0.6, 0.62, 0.58, 0.63, 0.57, 0.65, 0.55],
            "response": ["광합성", "호흡", "반사", "소화", "순환", "배설", "생식"],
            "token": [
                "광합성토큰",
                "호흡토큰",
                "반사토큰",
                "소화토큰",
                "순환토큰",
                "배설토큰",
                "생식토큰",
            ],
        }
    )
