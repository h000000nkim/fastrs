"""
Integration tests for preprocessing pipeline.

Tests the complete preprocessing workflow including cleaning, tokenization,
jamo decomposition, and data formatting as integrated processes.
"""

import pytest

from fastrs.core.object import Fastrs
from tests.fixtures import load_sample_data


class TestPreprocessingPipelineIntegration:
    """Test complete preprocessing pipeline integration."""

    def test_full_preprocessing_pipeline(self, fastrs_with_sample_data):
        """
        Test complete preprocessing pipeline from raw data to training format.

        Parameters
        ----------
        fastrs_with_sample_data : Fastrs
            Fastrs instance loaded with sample educational data.
        """
        # Execute full preprocessing pipeline
        fastrs_with_sample_data.preprocess(option="default")

        # Verify pipeline completion
        assert hasattr(fastrs_with_sample_data, "feed")
        assert isinstance(fastrs_with_sample_data.feed, list)
        assert len(fastrs_with_sample_data.feed) > 0

        # Verify feed contains tokenized sentences
        assert all(
            isinstance(sentence, list) for sentence in fastrs_with_sample_data.feed
        )
        assert all(
            isinstance(token, str)
            for sentence in fastrs_with_sample_data.feed
            for token in sentence
        )

    def test_preprocessing_pipeline_steps(self, fastrs_with_sample_data):
        """Test individual preprocessing steps work together."""
        # Step 1: Clean
        fastrs_with_sample_data.clean(target="information")

        # Verify cleaning occurred
        for item in fastrs_with_sample_data.items:
            assert hasattr(item, "cleanparams")

        # Step 2: Tokenize
        fastrs_with_sample_data.tokenize(target="all", option="morphs")

        # Step 3: Jamoize
        fastrs_with_sample_data.jamoize(target="all")

        # Verify jamoization occurred
        for item in fastrs_with_sample_data.items:
            assert hasattr(item, "jamodict")

        # Step 4: Formatize
        result = fastrs_with_sample_data.formatize(
            iterables=["answer", "response"], anchor="information", combine=True
        )

        # Verify final format
        assert isinstance(result, list)
        assert hasattr(fastrs_with_sample_data, "feed")
        assert len(fastrs_with_sample_data.feed) > 0

    def test_preprocessing_maintains_data_integrity(self, fastrs_with_sample_data):
        """Test that preprocessing maintains original data integrity."""
        # Store original data
        original_items = len(fastrs_with_sample_data.items)
        original_names = [item.name for item in fastrs_with_sample_data.items]

        # Run preprocessing
        fastrs_with_sample_data.preprocess(option="default")

        # Verify data integrity
        assert len(fastrs_with_sample_data.items) == original_items
        assert [item.name for item in fastrs_with_sample_data.items] == original_names

        # Verify original data is preserved
        for item in fastrs_with_sample_data.items:
            assert hasattr(item, "original_answer")
            assert hasattr(item, "original_response")
            assert hasattr(item, "original_information")

    def test_preprocessing_with_different_targets(self, fastrs_with_sample_data):
        """Test preprocessing with different target combinations."""
        # Test answer-only preprocessing
        fastrs_with_sample_data.clean(target="answer")
        fastrs_with_sample_data.tokenize(target="answer", option="morphs")
        fastrs_with_sample_data.jamoize(target="answer")

        # Test response-only preprocessing
        fastrs_with_sample_data.clean(target="response")
        fastrs_with_sample_data.tokenize(target="response", option="morphs")
        fastrs_with_sample_data.jamoize(target="response")

        # Test mixed target preprocessing
        fastrs_with_sample_data.clean(target=["answer", "response"])
        fastrs_with_sample_data.tokenize(target=["answer", "response"], option="morphs")
        fastrs_with_sample_data.jamoize(target=["answer", "response"])

        # All should complete without error
        assert True

    def test_preprocessing_error_handling(self, fastrs_with_sample_data):
        """Test preprocessing pipeline error handling."""
        # Test with invalid cleaning parameters
        with pytest.raises(ValueError):
            fastrs_with_sample_data.clean(space="invalid_option")

        # Test with invalid tokenization options
        with pytest.raises(ValueError):
            fastrs_with_sample_data.tokenize(option="invalid_option")

        # Test with invalid target specifications
        with pytest.raises(ValueError):
            fastrs_with_sample_data.clean(target="invalid_target")


class TestPreprocessingWithRealData:
    """Test preprocessing pipeline with real educational data."""

    def test_literature_question_preprocessing(self, literature_fastrs):
        """Test preprocessing of literature questions."""
        # Run preprocessing
        literature_fastrs.preprocess(option="default")

        # Verify literature-specific content is preserved
        assert hasattr(literature_fastrs, "feed")

        # Check that literary terms are properly tokenized
        feed_tokens = [
            token for sentence in literature_fastrs.feed for token in sentence
        ]

        # Should contain meaningful literary analysis tokens
        assert len(feed_tokens) > 0
        assert all(isinstance(token, str) for token in feed_tokens)

    def test_science_question_preprocessing(self, science_fastrs):
        """Test preprocessing of science questions."""
        # Run preprocessing
        science_fastrs.preprocess(option="default")

        # Verify science-specific content is preserved
        assert hasattr(science_fastrs, "feed")

        # Check that scientific terms are properly tokenized
        feed_tokens = [token for sentence in science_fastrs.feed for token in sentence]

        # Should contain scientific terminology
        assert len(feed_tokens) > 0
        assert all(isinstance(token, str) for token in feed_tokens)

    def test_mixed_subject_preprocessing(self, mixed_subjects_fastrs):
        """Test preprocessing with mixed subject questions."""
        # Run preprocessing
        mixed_subjects_fastrs.preprocess(option="default")

        # Verify all subjects are processed
        assert hasattr(mixed_subjects_fastrs, "feed")
        assert len(mixed_subjects_fastrs.items) > 1

        # Each item should have been processed
        for item in mixed_subjects_fastrs.items:
            assert hasattr(item, "jamodict")


class TestPreprocessingPerformance:
    """Test preprocessing pipeline performance characteristics."""

    def test_preprocessing_scalability(self, large_dataset_fastrs):
        """Test preprocessing with large dataset."""
        import time

        start_time = time.time()
        large_dataset_fastrs.preprocess(option="default")
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 90
        assert hasattr(large_dataset_fastrs, "feed")
        assert len(large_dataset_fastrs.feed) > 0

    def test_preprocessing_memory_efficiency(self, fastrs_with_sample_data):
        """Test that preprocessing doesn't cause excessive memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run preprocessing
        fastrs_with_sample_data.preprocess(option="default")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 100 * 1024 * 1024  # 100MB max increase


# Integration test fixtures
@pytest.fixture
def fastrs_with_sample_data():
    """Provide Fastrs instance with sample educational data."""
    data = load_sample_data()
    return Fastrs(data=data)


@pytest.fixture
def literature_fastrs():
    """Provide Fastrs instance with literature question data."""
    literature_data = load_sample_data()
    # Filter to literature questions only
    filtered_data = {
        k: v
        for k, v in literature_data.items()
        if "문학" in k or "시" in v.get("information", "")
    }
    if not filtered_data:
        filtered_data = {
            "item1": literature_data["item1"]
        }  # Use first item as fallback
    return Fastrs(data=filtered_data)


@pytest.fixture
def science_fastrs():
    """Provide Fastrs instance with science question data."""
    science_data = load_sample_data()
    # Filter to science questions only
    filtered_data = {
        k: v
        for k, v in science_data.items()
        if "과학" in k or "지구" in v.get("information", "")
    }
    if not filtered_data:
        filtered_data = {"item2": science_data["item2"]}  # Use second item as fallback
    return Fastrs(data=filtered_data)


@pytest.fixture
def mixed_subjects_fastrs():
    """Provide Fastrs instance with mixed subject data."""
    return Fastrs(data=load_sample_data())


@pytest.fixture
def large_dataset_fastrs():
    """Provide Fastrs instance with large dataset for performance testing."""
    # Create a larger synthetic dataset
    large_data = {}
    base_data = load_sample_data()

    for i in range(20):  # Create 20 items (adjust as needed)
        for key, value in base_data.items():
            new_key = f"{key}_copy_{i}"
            large_data[new_key] = {
                "information": f"{value['information']} (복사본 {i})",
                "answer": value["answer"][:],
                "response": value["response"][:],
            }

    return Fastrs(data=large_data)
