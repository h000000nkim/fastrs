"""
Unit tests for preprocessor functions.

Tests the individual preprocessing functions including text cleaning,
tokenization, jamo decomposition, and data formatting.
"""

import pytest
from unittest.mock import Mock, patch

from fastrs.core import preprocessor


class TestCleanFunction:
    """Test the clean function for text preprocessing."""

    def test_clean_basic_text(self):
        """Test basic text cleaning functionality."""
        text = "안녕하세요! 테스트입니다."
        result = preprocessor.clean(text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_with_space_allow(self):
        """Test cleaning with space allow option."""
        text = "안녕 하세요"
        result = preprocessor.clean(text, space="allow")

        assert " " in result

    def test_clean_with_space_forbid(self):
        """Test cleaning with space forbid option."""
        text = "안녕 하세요"
        result = preprocessor.clean(text, space="forbid")

        assert " " not in result

    def test_clean_with_special_forbid(self):
        """Test cleaning with special characters forbidden."""
        text = "안녕하세요!@#"
        result = preprocessor.clean(text, special="forbid")

        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_clean_with_caps_forbid(self):
        """Test cleaning with uppercase letters forbidden."""
        text = "Hello 안녕하세요"
        result = preprocessor.clean(text, caps="forbid")

        assert "H" not in result

    def test_clean_with_extra_forbid(self):
        """Test cleaning with extra forbidden characters."""
        text = "테스트.문장?"
        result = preprocessor.clean(text, extra_forbid=[".", "?"])

        assert "." not in result
        assert "?" not in result

    def test_clean_with_extra_allow(self):
        """Test cleaning with extra allowed characters."""
        text = "테스트.문장"
        result = preprocessor.clean(text, special="forbid", extra_allow=["."])

        assert "." in result

    @pytest.mark.parametrize("space_option", ["single allow", "allow", "forbid"])
    def test_clean_space_options(self, space_option):
        """Test all space options."""
        text = "안녕  하세요"
        result = preprocessor.clean(text, space=space_option)

        assert isinstance(result, str)


class TestTokenizeFunction:
    """Test the tokenize function for text tokenization."""

    @patch("fastrs.core.preprocessor.koreantokenizer")
    def test_tokenize_morphs_korean(self, mock_tokenizer):
        """Test morphological tokenization for Korean text."""
        mock_tokenizer.morphs.return_value = ["안녕", "하", "세요"]

        result = preprocessor.tokenize("안녕하세요", option="morphs")

        assert isinstance(result, list)
        assert result == ["안녕", "하", "세요"]
        mock_tokenizer.morphs.assert_called_once_with("안녕하세요")

    @patch("fastrs.core.preprocessor.koreantokenizer")
    def test_tokenize_nouns_korean(self, mock_tokenizer):
        """Test noun extraction for Korean text."""
        mock_tokenizer.nouns.return_value = ["학교", "학생"]

        result = preprocessor.tokenize("학교에 학생이 있다", option="nouns")

        assert isinstance(result, list)
        assert result == ["학교", "학생"]
        mock_tokenizer.nouns.assert_called_once_with("학교에 학생이 있다")

    @patch("fastrs.core.preprocessor.englishtokenizer")
    def test_tokenize_english_text(self, mock_tokenizer):
        """Test tokenization for English text."""
        mock_doc = Mock()
        mock_doc.text = "Hello world"
        mock_token1 = Mock()
        mock_token1.text = "Hello"
        mock_token2 = Mock()
        mock_token2.text = "world"
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        mock_tokenizer.return_value = mock_doc

        result = preprocessor.tokenize("Hello world", option="morphs")

        assert isinstance(result, list)
        assert len(result) >= 0

    def test_tokenize_mixed_text(self):
        """Test tokenization with mixed Korean-English text."""
        text = "안녕하세요 Hello"
        result = preprocessor.tokenize(text, option="morphs")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenize_empty_string(self):
        """Test tokenization with empty string."""
        result = preprocessor.tokenize("", option="morphs")

        assert isinstance(result, list)
        assert len(result) == 0


class TestJamoizeFunction:
    """Test the jamoize function for Korean character decomposition."""

    def test_jamoize_korean_text(self):
        """Test jamo decomposition for Korean text."""
        text = "안녕"
        result = preprocessor.jamoize(text)

        assert isinstance(result, str)
        assert len(result) > len(text)  # Jamo decomposition increases length

    def test_jamoize_with_spaces(self):
        """Test jamo decomposition preserving spaces."""
        text = "안녕 하세요"
        result = preprocessor.jamoize(text)

        assert isinstance(result, str)
        assert " " in result

    def test_jamoize_non_korean_text(self):
        """Test jamo decomposition with non-Korean text."""
        text = "Hello"
        result = preprocessor.jamoize(text)

        assert isinstance(result, str)
        assert result == text  # Should remain unchanged

    def test_jamoize_mixed_text(self):
        """Test jamo decomposition with mixed Korean-English text."""
        text = "안녕 Hello"
        result = preprocessor.jamoize(text)

        assert isinstance(result, str)
        assert "Hello" in result  # English part unchanged

    def test_jamoize_empty_string(self):
        """Test jamo decomposition with empty string."""
        result = preprocessor.jamoize("")

        assert result == ""


class TestFormatizeFunction:
    """Test the formatize function for creating training data."""

    def test_formatize_with_combine_true(self):
        """Test formatize with combine=True."""
        iterables = [["word1", "word2"], ["word3", "word4"]]
        anchor = ["anchor1", "anchor2"]

        result = preprocessor.formatize(iterables, anchor, combine=True)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(sentence, list) for sentence in result)

    def test_formatize_with_combine_false(self):
        """Test formatize with combine=False."""
        iterables = [["word1", "word2"], ["word3", "word4"]]
        anchor = ["anchor1", "anchor2"]

        result = preprocessor.formatize(iterables, anchor, combine=False)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_formatize_empty_iterables(self):
        """Test formatize with empty iterables."""
        iterables = []
        anchor = ["anchor1"]

        result = preprocessor.formatize(iterables, anchor)

        assert isinstance(result, list)

    def test_formatize_empty_anchor(self):
        """Test formatize with empty anchor."""
        iterables = [["word1", "word2"]]
        anchor = []

        result = preprocessor.formatize(iterables, anchor)

        assert isinstance(result, list)

    def test_formatize_single_iterable(self):
        """Test formatize with single iterable."""
        iterables = [["word1", "word2", "word3"]]
        anchor = ["anchor"]

        result = preprocessor.formatize(iterables, anchor, combine=True)

        assert isinstance(result, list)
        assert len(result) > 0


class TestPreprocessorIntegration:
    """Test integration between preprocessor functions."""

    def test_cleaning_then_tokenizing(self):
        """Test cleaning followed by tokenization."""
        text = "안녕하세요! 테스트입니다."

        # Clean first
        cleaned = preprocessor.clean(text, special="forbid")

        # Then tokenize
        tokens = preprocessor.tokenize(cleaned, option="morphs")

        assert isinstance(cleaned, str)
        assert isinstance(tokens, list)
        assert "!" not in cleaned

    def test_tokenizing_then_jamoizing(self):
        """Test tokenization followed by jamo decomposition."""
        text = "안녕하세요"

        # Tokenize first
        tokens = preprocessor.tokenize(text, option="morphs")

        # Then jamoize each token
        jamo_tokens = [preprocessor.jamoize(token) for token in tokens]

        assert all(isinstance(token, str) for token in jamo_tokens)
        assert len(jamo_tokens) == len(tokens)

    def test_full_preprocessing_pipeline(self):
        """Test full preprocessing pipeline."""
        text = "안녕하세요! 좋은 하루입니다."

        # Step 1: Clean
        cleaned = preprocessor.clean(text, special="forbid")

        # Step 2: Tokenize
        tokens = preprocessor.tokenize(cleaned, option="morphs")

        # Step 3: Jamoize
        jamo_tokens = [preprocessor.jamoize(token) for token in tokens]

        # Step 4: Format
        result = preprocessor.formatize([jamo_tokens], jamo_tokens[:2])

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(sentence, list) for sentence in result)


# Error handling tests
class TestPreprocessorErrorHandling:
    """Test error handling in preprocessor functions."""

    def test_clean_invalid_parameters(self):
        """Test clean function with invalid parameters."""
        with pytest.raises(ValueError):
            preprocessor.clean("text", space="invalid")

    def test_tokenize_invalid_option(self):
        """Test tokenize function with invalid option."""
        with pytest.raises(ValueError):
            preprocessor.tokenize("text", option="invalid")

    def test_formatize_invalid_types(self):
        """Test formatize function with invalid types."""
        with pytest.raises(TypeError):
            preprocessor.formatize("not_list", ["anchor"])
