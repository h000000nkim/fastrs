"""
Unit tests for utility functions.

Tests the utility functions including model loading, configuration management,
type checking, data validation, and helper functions.
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, mock_open
from gensim.models import FastText

from fastrs.core import util
from fastrs.core.exceptions import UtilError


class TestConfigurationLoading:
    """Test configuration loading functions."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        mock_config = {"test": "value", "number": 42}

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            result = util.load_config("test_config")

        assert result == mock_config

    def test_load_config_file_not_found(self):
        """Test configuration loading with missing file."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(UtilError, match="설정 파일을 찾을 수 없습니다"):
                util.load_config("missing_config")

    def test_load_config_invalid_json(self):
        """Test configuration loading with invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with pytest.raises(UtilError, match="설정 파일을 파싱할 수 없습니다"):
                util.load_config("invalid_config")

    def test_load_color_schemes(self):
        """Test color schemes configuration loading."""
        mock_config = {"scheme1": ["#FF0000", "#00FF00"]}

        with patch("fastrs.core.util.load_config", return_value=mock_config):
            result = util.load_color_schemes()

        assert result == mock_config

    def test_load_plot_config(self):
        """Test plot configuration loading."""
        mock_config = {"width": 800, "height": 600}

        with patch("fastrs.core.util.load_config", return_value=mock_config):
            result = util.load_plot_config()

        assert result == mock_config

    def test_load_reduction_defaults(self):
        """Test reduction defaults configuration loading."""
        mock_config = {"umap": {"n_components": 2}, "pca": {"n_components": 2}}

        with patch("fastrs.core.util.load_config", return_value=mock_config):
            result = util.load_reduction_defaults()

        assert result == mock_config

    def test_load_fasttext_defaults(self):
        """Test FastText defaults configuration loading."""
        mock_config = {"vector_size": 100, "window": 5}

        with patch("fastrs.core.util.load_config", return_value=mock_config):
            result = util.load_fasttext_defaults()

        assert result == mock_config


class TestPretrainedModelLoading:
    """Test pretrained model loading functionality."""

    @patch("fastrs.core.util.load_facebook_model")
    @patch("os.path.exists")
    def test_get_pretrained_model_existing_bin(self, mock_exists, mock_load):
        """Test loading existing .bin model file."""
        mock_exists.side_effect = lambda path: path.endswith(".bin")
        mock_model = Mock(spec=FastText)
        mock_load.return_value = mock_model

        result = util.get_pretrained_model()

        assert result == mock_model
        mock_load.assert_called_once()

    @patch("fastrs.core.util.load_facebook_model")
    @patch("shutil.copyfileobj")
    @patch("gzip.open")
    @patch("urllib.request.urlretrieve")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_get_pretrained_model_download_and_extract(
        self,
        mock_makedirs,
        mock_exists,
        mock_urlretrieve,
        mock_gzip,
        mock_copyfileobj,
        mock_load,
    ):
        """Test downloading and extracting model file."""
        # Simulate: .bin doesn't exist, .gz doesn't exist
        mock_exists.return_value = False
        mock_model = Mock(spec=FastText)
        mock_load.return_value = mock_model

        result = util.get_pretrained_model()

        mock_makedirs.assert_called_once()
        mock_urlretrieve.assert_called_once()
        mock_gzip.assert_called_once()
        mock_copyfileobj.assert_called_once()
        assert result == mock_model

    @patch("fastrs.core.util.load_facebook_model")
    @patch("shutil.copyfileobj")
    @patch("gzip.open")
    @patch("os.path.exists")
    def test_get_pretrained_model_existing_gz(
        self, mock_exists, mock_gzip, mock_copyfileobj, mock_load
    ):
        """Test extracting existing .gz file."""
        # Simulate: .bin doesn't exist, .gz exists
        mock_exists.side_effect = lambda path: path.endswith(".gz")
        mock_model = Mock(spec=FastText)
        mock_load.return_value = mock_model

        result = util.get_pretrained_model()

        mock_gzip.assert_called_once()
        mock_copyfileobj.assert_called_once()
        assert result == mock_model

    def test_get_pretrained_model_custom_url_and_dir(self):
        """Test model loading with custom URL and directory."""
        custom_url = "https://example.com/model.bin.gz"
        custom_dir = "./custom_models"

        with patch("fastrs.core.util.load_facebook_model") as mock_load:
            with patch("os.path.exists", return_value=True):
                mock_model = Mock(spec=FastText)
                mock_load.return_value = mock_model

                result = util.get_pretrained_model(url=custom_url, model_dir=custom_dir)

        assert result == mock_model


class TestTypeChecking:
    """Test type checking utility functions."""

    def test_typecheck_single_valid_type(self):
        """Test type checking with single valid type."""
        # Should not raise any exception
        util.typecheck("hello", str)
        util.typecheck(42, int)
        util.typecheck([1, 2, 3], list)

    def test_typecheck_single_invalid_type(self):
        """Test type checking with single invalid type."""
        with pytest.raises(TypeError, match="Type of input must be"):
            util.typecheck("hello", int)

        with pytest.raises(TypeError, match="Type of input must be"):
            util.typecheck(42, str)

    def test_typecheck_tuple_with_single_expected_type(self):
        """Test type checking with tuple input and single expected type."""
        util.typecheck(("hello", "world"), str)

        with pytest.raises(TypeError):
            util.typecheck(("hello", 42), str)

    def test_typecheck_tuple_with_multiple_expected_types(self):
        """Test type checking with tuple input and multiple expected types."""
        util.typecheck(("hello", 42), [str, int])

        with pytest.raises(TypeError):
            util.typecheck(("hello", 42), [str, str])

    def test_typecheck_empty_tuple(self):
        """Test type checking with empty tuple."""
        util.typecheck((), str)  # Should pass without errors

    def test_typecheck_none_values(self):
        """Test type checking with None values."""
        util.typecheck(None, type(None))

        with pytest.raises(TypeError):
            util.typecheck(None, str)


class TestLiteralChecking:
    """Test literal value checking utility functions."""

    def test_literalcheck_single_valid_value(self):
        """Test literal checking with single valid value."""
        util.literalcheck("option1", ["option1", "option2", "option3"])

    def test_literalcheck_single_invalid_value(self):
        """Test literal checking with single invalid value."""
        with pytest.raises(ValueError, match="Invalid input"):
            util.literalcheck("invalid", ["option1", "option2"])

    def test_literalcheck_list_valid_values(self):
        """Test literal checking with list of valid values."""
        util.literalcheck(["option1", "option2"], ["option1", "option2", "option3"])

    def test_literalcheck_list_with_invalid_value(self):
        """Test literal checking with list containing invalid value."""
        with pytest.raises(ValueError, match="Invalid input"):
            util.literalcheck(["option1", "invalid"], ["option1", "option2"])

    def test_literalcheck_empty_input(self):
        """Test literal checking with empty input."""
        util.literalcheck([], ["option1", "option2"])  # Should pass

    def test_literalcheck_empty_literal_list(self):
        """Test literal checking with empty literal list."""
        with pytest.raises(ValueError):
            util.literalcheck("anything", [])


class TestDataFormatting:
    """Test data formatting utility functions."""

    def test_formatData_valid_arrays(self):
        """Test formatData with valid numpy arrays."""
        answers = np.array([["ans1", "ans2"], ["ans3", "ans4"]])
        responses = np.array([["resp1", "resp2"], ["resp3", "resp4"]])
        informations = np.array(["info1", "info2"])

        result = util.formatData(answers, responses, informations)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "item1" in result
        assert "item2" in result
        np.testing.assert_array_equal(result["item1"]["answer"], ["ans1", "ans2"])
        np.testing.assert_array_equal(result["item1"]["response"], ["resp1", "resp2"])
        assert result["item1"]["information"] == "info1"

    def test_formatData_mismatched_lengths(self):
        """Test formatData with mismatched array lengths."""
        answers = np.array([["ans1"], ["ans2"]])
        responses = np.array([["resp1"]])  # Different length
        informations = np.array(["info1", "info2"])

        with pytest.raises(UtilError, match="Length of answer, response"):
            util.formatData(answers, responses, informations)

    def test_formatData_none_values(self):
        """Test formatData with None values."""
        with pytest.raises(
            UtilError, match="answer, response, and information must be provided"
        ):
            util.formatData(None, None, None)


class TestDataValidation:
    """Test data validation utility functions."""

    def test_validData_valid_structure(self):
        """Test validData with valid data structure."""
        data = {
            "item1": {
                "answer": ["ans1", "ans2"],
                "response": ["resp1", "resp2"],
                "information": "info1",
            }
        }

        # Should not raise any exception
        util.validData(data)

    def test_validData_invalid_answer_type(self):
        """Test validData with invalid answer type."""
        data = {
            "item1": {
                "answer": "not_a_list",  # Should be list/tuple/array
                "response": ["resp1"],
                "information": "info1",
            }
        }

        with pytest.raises(UtilError, match="answer must be list, tuple"):
            util.validData(data)

    def test_validData_invalid_response_type(self):
        """Test validData with invalid response type."""
        data = {
            "item1": {
                "answer": ["ans1"],
                "response": "not_a_list",  # Should be list/tuple/array
                "information": "info1",
            }
        }

        with pytest.raises(UtilError, match="response must be list, tuple"):
            util.validData(data)

    def test_validData_invalid_information_type(self):
        """Test validData with invalid information type."""
        data = {
            "item1": {
                "answer": ["ans1"],
                "response": ["resp1"],
                "information": 123,  # Should be string
            }
        }

        with pytest.raises(UtilError, match="information must be str"):
            util.validData(data)

    def test_validData_mismatched_lengths(self):
        """Test validData with mismatched answer/response lengths - this is allowed."""
        data = {
            "item1": {
                "answer": ["ans1", "ans2"],
                "response": ["resp1"],  # Different length is allowed
                "information": "info1",
            }
        }

        # Should not raise any exception - different lengths are allowed
        util.validData(data)

    def test_validData_empty_answers(self):
        """Test validData with empty answers."""
        data = {
            "item1": {
                "answer": [],  # Empty
                "response": [],
                "information": "info1",
            }
        }

        with pytest.raises(UtilError, match="Length of answer must be greater than 0"):
            util.validData(data)

    def test_validData_non_string_elements(self):
        """Test validData with non-string elements in answer/response."""
        data = {
            "item1": {
                "answer": ["ans1", 42],  # Non-string element
                "response": ["resp1", "resp2"],
                "information": "info1",
            }
        }

        with pytest.raises(UtilError, match="All elements in answer must be type str"):
            util.validData(data)

    def test_validData_with_numpy_arrays(self):
        """Test validData with valid numpy arrays."""
        data = {
            "item1": {
                "answer": np.array(["ans1", "ans2"]),
                "response": np.array(["resp1", "resp2"]),
                "information": "info1",
            }
        }

        # Should not raise any exception
        util.validData(data)

    def test_validData_with_multidimensional_arrays(self):
        """Test validData with invalid multidimensional arrays."""
        data = {
            "item1": {
                "answer": np.array([["ans1"], ["ans2"]]),  # 2D array
                "response": np.array(["resp1", "resp2"]),
                "information": "info1",
            }
        }

        with pytest.raises(UtilError, match="answer must be 1-dimensional"):
            util.validData(data)


class TestUtilityIntegration:
    """Test integration between utility functions."""

    def test_format_then_validate_data(self):
        """Test formatting data then validating it."""
        answers = np.array([["ans1", "ans2"]])
        responses = np.array([["resp1", "resp2"]])
        informations = np.array(["info1"])

        # Format data
        formatted_data = util.formatData(answers, responses, informations)

        # Validate formatted data
        util.validData(formatted_data)  # Should not raise exception

    @patch("fastrs.core.util.load_config")
    def test_config_loading_chain(self, mock_load_config):
        """Test chaining of configuration loading functions."""
        mock_config = {"test": "value"}
        mock_load_config.return_value = mock_config

        # Load different configs
        color_config = util.load_color_schemes()
        plot_config = util.load_plot_config()
        reduction_config = util.load_reduction_defaults()
        fasttext_config = util.load_fasttext_defaults()

        # All should return the same mock config
        assert color_config == mock_config
        assert plot_config == mock_config
        assert reduction_config == mock_config
        assert fasttext_config == mock_config

        # Should have called load_config 4 times
        assert mock_load_config.call_count == 4
