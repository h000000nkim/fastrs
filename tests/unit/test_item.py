"""
Unit tests for the Item class.

Tests the Item class functionality including initialization,
preprocessing methods, and visualization in isolation.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from fastrs.core.object import Item
from fastrs.core.exceptions import ItemError


class TestItemInitialization:
    """Test Item class initialization."""

    def test_init_valid_data(self):
        """Test Item initialization with valid data."""
        item = Item(
            name="test_item",
            answer=["correct", "answer"],
            response=["student", "response"],
            information="Test question information",
        )

        assert item.name == "test_item"
        assert item.answer == ["correct", "answer"]
        assert item.response == ["student", "response"]
        assert item.information == "Test question information"
        assert item.original_answer == ["correct", "answer"]
        assert item.original_response == ["student", "response"]
        assert item.original_information == "Test question information"

    def test_init_without_information(self):
        """Test Item initialization without information parameter."""
        item = Item(name="test_item", answer=["correct"], response=["response"])

        assert item.information is None
        assert item.original_information is None

    def test_init_invalid_types(self):
        """Test Item initialization with invalid types raises TypeError."""
        with pytest.raises(TypeError):
            Item(name=123, answer=["correct"], response=["response"])

        with pytest.raises(TypeError):
            Item(name="test", answer="not_list", response=["response"])

        with pytest.raises(TypeError):
            Item(name="test", answer=["correct"], response="not_list")


class TestItemPreprocessing:
    """Test Item preprocessing methods."""

    def test_clean_all_targets(self, sample_item):
        """
        Test cleaning all targets.

        Parameters
        ----------
        sample_item : Item
            Sample Item instance for testing.
        """
        result = sample_item.clean(target="all")

        assert hasattr(sample_item, "cleanparams")
        assert isinstance(result, tuple)
        assert len(result) == 3  # answer, response, information

    def test_clean_specific_targets(self, sample_item):
        """Test cleaning specific targets."""
        result = sample_item.clean(target=["answer", "response"])

        assert isinstance(result, tuple)
        assert len(result) == 2  # only answer and response

    def test_clean_with_parameters(self, sample_item):
        """Test cleaning with various parameters."""
        sample_item.clean(
            target="all",
            space="allow",
            special="forbid",
            caps="forbid",
            extra_forbid=["!", "?"],
            extra_allow=["."],
        )

        assert sample_item.cleanparams["space"] == "allow"
        assert sample_item.cleanparams["special"] == "forbid"
        assert sample_item.cleanparams["extra_forbid"] == ["!", "?"]
        assert sample_item.cleanparams["extra_allow"] == ["."]

    def test_tokenize_morphs(self, sample_item):
        """Test morphological tokenization."""
        with patch("fastrs.core.preprocessor.tokenize") as mock_tokenize:
            mock_tokenize.return_value = ["?좏겙1", "?좏겙2"]

            result = sample_item.tokenize(target="all", option="morphs")

            assert isinstance(result, tuple)
            mock_tokenize.assert_called()

    def test_tokenize_nouns(self, sample_item):
        """Test noun extraction tokenization."""
        with patch("fastrs.core.preprocessor.tokenize") as mock_tokenize:
            mock_tokenize.return_value = ["紐낆궗1", "紐낆궗2"]

            result = sample_item.tokenize(target="answer", option="nouns")

            assert isinstance(result, tuple)
            mock_tokenize.assert_called()

    def test_jamoize_all_targets(self, sample_item):
        """Test jamo decomposition for all targets."""
        with patch("fastrs.core.preprocessor.jamoize") as mock_jamoize:
            mock_jamoize.return_value = "?덀뀖?곥뀠"

            result = sample_item.jamoize(target="all")

            assert isinstance(result, tuple)
            assert hasattr(sample_item, "jamodict")
            mock_jamoize.assert_called()

    def test_formatize_with_combine(self, preprocessed_item):
        """Test formatize with combine=True."""
        result = preprocessed_item.formatize(
            iterables=["answer", "response"], anchor="information", combine=True
        )

        assert isinstance(result, list)
        assert hasattr(preprocessed_item, "feed")

    def test_formatize_without_combine(self, preprocessed_item):
        """Test formatize with combine=False."""
        result = preprocessed_item.formatize(
            iterables=["answer", "response"], anchor="information", combine=False
        )

        assert isinstance(result, list)


class TestItemVisualization:
    """Test Item visualization methods."""

    def test_visualize_success(self, item_with_coordinates):
        """Test successful visualization creation."""
        with patch("fastrs.core.object.scatter") as mock_viz:
            mock_plot = Mock()
            mock_viz.return_value = mock_plot

            result = item_with_coordinates.visualize()

            assert result == mock_plot
            assert hasattr(item_with_coordinates, "plot")
            mock_viz.assert_called_once()

    def test_visualize_without_coordinates_raises_error(self, sample_item):
        """Test visualization without coordinates raises ItemError."""
        with pytest.raises(ItemError, match="No coordinates found"):
            sample_item.visualize()


class TestItemPrivateMethods:
    """Test Item private methods."""

    def test_match_target_all(self, sample_item):
        """Test _match_target method with 'all' target."""

        def dummy_func(text):
            return text.upper()

        result = sample_item._match_target("all", dummy_func)

        assert "answer" in result
        assert "response" in result
        assert "information" in result

    def test_match_target_specific(self, sample_item):
        """Test _match_target method with specific targets."""

        def dummy_func(text):
            return text.upper()

        result = sample_item._match_target(["answer"], dummy_func)

        assert isinstance(result["answer"], list)
        assert result["response"] == sample_item.response  # unchanged

    def test_parse_return_all(self, sample_item):
        """Test _parse_return method with 'all' target."""
        result = sample_item._parse_return("all")

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_parse_return_specific(self, sample_item):
        """Test _parse_return method with specific targets."""
        result = sample_item._parse_return(["answer", "response"])

        assert isinstance(result, tuple)
        assert len(result) == 2


# Fixtures specific to this test module


@pytest.fixture
def preprocessed_item(sample_item):
    """Provide an Item instance with preprocessed data."""
    # Mock preprocessed attributes
    sample_item.answer = ["processed", "answer"]
    sample_item.response = ["processed", "student", "answer"]
    sample_item.information = "processed info"

    return sample_item


@pytest.fixture
def item_with_coordinates(sample_item):
    """Provide an Item instance with coordinates for visualization."""
    sample_item.coordinates = pd.DataFrame(
        {
            "response": ["허무", "공허"],
            "token": ["token1", "token2"],
            "x": [0.1, 0.2],
            "y": [0.3, 0.4],
        }
    )

    sample_item.original_answer = ["허무"]

    return sample_item
