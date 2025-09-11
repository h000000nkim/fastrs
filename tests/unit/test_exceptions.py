"""
Unit tests for custom exceptions.

Tests the custom exception classes and their inheritance hierarchy,
ensuring proper error handling throughout the FastRS package.
"""

import pytest

from fastrs.core.exceptions import (
    FastrsError,
    PreprocessingError,
    TrainingError,
    UtilError,
    ReducerError,
    ItemError,
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy and inheritance."""

    def test_fastrs_error_base_class(self):
        """Test that FastrsError is the base exception class."""
        error = FastrsError("Test error")

        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_preprocessing_error_inheritance(self):
        """Test that PreprocessingError inherits from FastrsError."""
        error = PreprocessingError("Preprocessing failed")

        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        assert str(error) == "Preprocessing failed"

    def test_training_error_inheritance(self):
        """Test that TrainingError inherits from FastrsError."""
        error = TrainingError("Training failed")

        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        assert str(error) == "Training failed"

    def test_util_error_inheritance(self):
        """Test that UtilError inherits from FastrsError."""
        error = UtilError("Utility function failed")

        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        assert str(error) == "Utility function failed"

    def test_reducer_error_inheritance(self):
        """Test that ReducerError inherits from FastrsError."""
        error = ReducerError("Dimensionality reduction failed")

        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        assert str(error) == "Dimensionality reduction failed"

    def test_item_error_inheritance(self):
        """Test that ItemError inherits from FastrsError."""
        error = ItemError("Item processing failed")

        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        assert str(error) == "Item processing failed"


class TestExceptionRaising:
    """Test raising and catching custom exceptions."""

    def test_raise_and_catch_fastrs_error(self):
        """Test raising and catching FastrsError."""
        with pytest.raises(FastrsError, match="Base error"):
            raise FastrsError("Base error")

    def test_raise_and_catch_preprocessing_error(self):
        """Test raising and catching PreprocessingError."""
        with pytest.raises(PreprocessingError, match="Preprocessing issue"):
            raise PreprocessingError("Preprocessing issue")

    def test_raise_and_catch_training_error(self):
        """Test raising and catching TrainingError."""
        with pytest.raises(TrainingError, match="Training issue"):
            raise TrainingError("Training issue")

    def test_raise_and_catch_util_error(self):
        """Test raising and catching UtilError."""
        with pytest.raises(UtilError, match="Utility issue"):
            raise UtilError("Utility issue")

    def test_raise_and_catch_reducer_error(self):
        """Test raising and catching ReducerError."""
        with pytest.raises(ReducerError, match="Reduction issue"):
            raise ReducerError("Reduction issue")

    def test_raise_and_catch_item_error(self):
        """Test raising and catching ItemError."""
        with pytest.raises(ItemError, match="Item issue"):
            raise ItemError("Item issue")


class TestExceptionCatchingPolymorphism:
    """Test polymorphic exception catching."""

    def test_catch_specific_error_as_fastrs_error(self):
        """Test that specific errors can be caught as FastrsError."""
        with pytest.raises(FastrsError):
            raise PreprocessingError("Specific error")

        with pytest.raises(FastrsError):
            raise TrainingError("Specific error")

        with pytest.raises(FastrsError):
            raise UtilError("Specific error")

        with pytest.raises(FastrsError):
            raise ReducerError("Specific error")

        with pytest.raises(FastrsError):
            raise ItemError("Specific error")

    def test_catch_specific_error_as_exception(self):
        """Test that all errors can be caught as base Exception."""
        with pytest.raises(Exception):
            raise FastrsError("Any error")

        with pytest.raises(Exception):
            raise PreprocessingError("Any error")

        with pytest.raises(Exception):
            raise TrainingError("Any error")


class TestExceptionWithDifferentArguments:
    """Test exceptions with various argument types."""

    def test_exception_with_no_arguments(self):
        """Test exception with no arguments."""
        error = FastrsError()
        assert str(error) == ""

    def test_exception_with_string_argument(self):
        """Test exception with string argument."""
        message = "Something went wrong"
        error = PreprocessingError(message)
        assert str(error) == message

    def test_exception_with_multiple_arguments(self):
        """Test exception with multiple arguments."""
        error = TrainingError("Error", "Additional info")
        # The exact string representation depends on Python's implementation
        assert isinstance(str(error), str)

    def test_exception_with_formatted_string(self):
        """Test exception with formatted string."""
        item_name = "test_item"
        error_code = 404
        error = UtilError(f"Item '{item_name}' not found (code: {error_code})")

        expected_message = "Item 'test_item' not found (code: 404)"
        assert str(error) == expected_message


class TestExceptionDocstrings:
    """Test that exceptions have proper docstrings."""

    def test_fastrs_error_docstring(self):
        """Test FastrsError has docstring."""
        assert FastrsError.__doc__ is not None
        assert "Base class for exceptions raised by Fastrs" in FastrsError.__doc__

    def test_preprocessing_error_docstring(self):
        """Test PreprocessingError has docstring."""
        assert PreprocessingError.__doc__ is not None
        assert "preprocessing" in PreprocessingError.__doc__.lower()

    def test_training_error_docstring(self):
        """Test TrainingError has docstring."""
        assert TrainingError.__doc__ is not None
        assert "training" in TrainingError.__doc__.lower()

    def test_util_error_docstring(self):
        """Test UtilError has docstring."""
        assert UtilError.__doc__ is not None
        assert "utility" in UtilError.__doc__.lower()

    def test_reducer_error_docstring(self):
        """Test ReducerError has docstring."""
        assert ReducerError.__doc__ is not None
        assert "dimensionality reduction" in ReducerError.__doc__.lower()

    def test_item_error_docstring(self):
        """Test ItemError has docstring."""
        assert ItemError.__doc__ is not None
        assert "Item" in ItemError.__doc__


class TestExceptionInRealUseCases:
    """Test exceptions in realistic usage scenarios."""

    def test_preprocessing_error_scenario(self):
        """Test PreprocessingError in a realistic preprocessing scenario."""

        def mock_preprocess_text(text):
            if not isinstance(text, str):
                raise PreprocessingError(
                    f"Expected string input, got {type(text).__name__}"
                )
            return text.lower()

        # Valid case
        result = mock_preprocess_text("Hello World")
        assert result == "hello world"

        # Error case
        with pytest.raises(PreprocessingError, match="Expected string input, got int"):
            mock_preprocess_text(123)

    def test_training_error_scenario(self):
        """Test TrainingError in a realistic training scenario."""

        def mock_train_model(data):
            if not data:
                raise TrainingError("Cannot train model with empty data")
            if len(data) < 10:
                raise TrainingError(
                    f"Insufficient training data: {len(data)} samples (minimum: 10)"
                )
            return "trained_model"

        # Valid case
        data = list(range(15))
        result = mock_train_model(data)
        assert result == "trained_model"

        # Error cases
        with pytest.raises(TrainingError, match="Cannot train model with empty data"):
            mock_train_model([])

        with pytest.raises(TrainingError, match="Insufficient training data: 5"):
            mock_train_model(list(range(5)))

    def test_util_error_scenario(self):
        """Test UtilError in a realistic utility function scenario."""

        def mock_load_config(config_name):
            valid_configs = ["plot_config", "model_config", "data_config"]
            if config_name not in valid_configs:
                raise UtilError(
                    f"Unknown configuration: {config_name}. "
                    f"Available: {', '.join(valid_configs)}"
                )
            return {"loaded": config_name}

        # Valid case
        result = mock_load_config("plot_config")
        assert result == {"loaded": "plot_config"}

        # Error case
        with pytest.raises(UtilError, match="Unknown configuration: invalid_config"):
            mock_load_config("invalid_config")

    def test_reducer_error_scenario(self):
        """Test ReducerError in a realistic dimensionality reduction scenario."""

        def mock_reduce_dimensions(vectors, method="pca"):
            if len(vectors) == 0:
                raise ReducerError("Cannot reduce dimensions of empty vector set")
            if method not in ["pca", "tsne", "umap"]:
                raise ReducerError(f"Unsupported reduction method: {method}")
            return f"reduced_with_{method}"

        # Valid case
        result = mock_reduce_dimensions([1, 2, 3], "pca")
        assert result == "reduced_with_pca"

        # Error cases
        with pytest.raises(ReducerError, match="Cannot reduce dimensions of empty"):
            mock_reduce_dimensions([])

        with pytest.raises(ReducerError, match="Unsupported reduction method: invalid"):
            mock_reduce_dimensions([1, 2, 3], "invalid")

    def test_item_error_scenario(self):
        """Test ItemError in a realistic item processing scenario."""

        def mock_process_item(item):
            if not hasattr(item, "data"):
                raise ItemError("Item must have 'data' attribute")
            if not item.data:
                raise ItemError("Item data cannot be empty")
            return f"processed_{len(item.data)}_elements"

        # Mock item classes
        class ValidItem:
            def __init__(self, data):
                self.data = data

        class InvalidItem:
            pass

        # Valid case
        valid_item = ValidItem([1, 2, 3])
        result = mock_process_item(valid_item)
        assert result == "processed_3_elements"

        # Error cases
        with pytest.raises(ItemError, match="Item must have 'data' attribute"):
            mock_process_item(InvalidItem())

        with pytest.raises(ItemError, match="Item data cannot be empty"):
            mock_process_item(ValidItem([]))
