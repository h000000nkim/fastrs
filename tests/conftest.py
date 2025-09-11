"""
Pytest configuration and shared fixtures for FastRS package testing.

This module provides shared fixtures, configuration, and utilities used
across all test modules in the FastRS package test suite.
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
from unittest.mock import Mock
from gensim.models import FastText

from fastrs.core.object import Fastrs, Item
from tests.fixtures import load_sample_data, create_minimal_data


# ================================
# Pytest Configuration
# ================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers automatically."""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to potentially slow tests
        if any(
            keyword in item.name.lower()
            for keyword in ["large", "performance", "scalability"]
        ):
            item.add_marker(pytest.mark.slow)


# ================================
# Core Data Fixtures
# ================================


@pytest.fixture
def sample_data():
    """
    Provide sample educational assessment data.

    Returns
    -------
    dict
        Dictionary containing Korean educational assessment questions
        with answers and student responses.
    """
    return load_sample_data()


@pytest.fixture
def minimal_data():
    """
    Provide minimal test data for basic functionality tests.

    Returns
    -------
    dict
        Minimal data dictionary with one test item.
    """
    return create_minimal_data()


@pytest.fixture
def korean_text_data():
    """
    Provide Korean text data for language-specific testing.

    Returns
    -------
    dict
        Korean text data for preprocessing and tokenization tests.
    """
    return {
        "korean_item1": {
            "information": "한국어 자연어 처리를 위한 테스트 데이터입니다. 형태소 분석과 자모 분해를 테스트합니다.",
            "answer": ["정답", "올바른답안", "맞는답"],
            "response": [
                "정답",
                "올바른답안",
                "맞는답",
                "틀린답",
                "부분정답",
                "애매한답안",
            ],
        },
        "korean_item2": {
            "information": "또 다른 한국어 텍스트 처리 예제입니다. 다양한 어휘와 표현을 포함합니다.",
            "answer": ["해답", "정확한답", "표준답안"],
            "response": [
                "해답",
                "정확한답",
                "표준답안",
                "오답",
                "근접답안",
                "다른해석",
            ],
        },
    }


@pytest.fixture
def literature_data():
    """
    Provide literature-specific test data.

    Returns
    -------
    dict
        Literature analysis data with poetic and literary terms.
    """
    return {
        "literature_item": {
            "information": "박목월의 시 「나그네」 분석 문제입니다. 화자의 정서와 시상 전개를 파악하는 문제입니다.",
            "answer": ["허무", "허무감"],
            "response": [
                "허무",
                "허무감",
                "공허",
                "무상",
                "초월",
                "달관",
                "체념",
                "탈속",
                "현실도피",
                "허탈",
            ],
        }
    }


@pytest.fixture
def science_data():
    """
    Provide science-specific test data.

    Returns
    -------
    dict
        Science reading comprehension data with technical terms.
    """
    return {
        "science_item": {
            "information": "지구 온난화와 북극해 해빙 감소에 대한 과학적 현상을 설명하는 지문입니다.",
            "answer": ["흡수율", "태양에너지흡수율"],
            "response": [
                "흡수율",
                "태양에너지흡수율",
                "흡수",
                "반사율",
                "알베도",
                "온도상승",
                "열흡수",
            ],
        }
    }


# ================================
# FastRS Instance Fixtures
# ================================


@pytest.fixture
def fastrs_instance(sample_data):
    """
    Provide basic Fastrs instance.

    Parameters
    ----------
    sample_data : dict
        Sample educational assessment data.

    Returns
    -------
    Fastrs
        Basic Fastrs instance with sample data loaded.
    """
    return Fastrs(data=sample_data)


@pytest.fixture
def fastrs_minimal(minimal_data):
    """
    Provide minimal Fastrs instance for quick tests.

    Parameters
    ----------
    minimal_data : dict
        Minimal test data.

    Returns
    -------
    Fastrs
        Fastrs instance with minimal data.
    """
    return Fastrs(data=minimal_data)


@pytest.fixture
def fastrs_korean(korean_text_data):
    """
    Provide Fastrs instance with Korean text data.

    Parameters
    ----------
    korean_text_data : dict
        Korean text data for language processing tests.

    Returns
    -------
    Fastrs
        Fastrs instance loaded with Korean text data.
    """
    return Fastrs(data=korean_text_data)


@pytest.fixture
def fastrs_literature(literature_data):
    """
    Provide Fastrs instance with literature data.

    Parameters
    ----------
    literature_data : dict
        Literature analysis data.

    Returns
    -------
    Fastrs
        Fastrs instance for literature analysis testing.
    """
    return Fastrs(data=literature_data)


@pytest.fixture
def fastrs_science(science_data):
    """
    Provide Fastrs instance with science data.

    Parameters
    ----------
    science_data : dict
        Science reading comprehension data.

    Returns
    -------
    Fastrs
        Fastrs instance for science text analysis testing.
    """
    return Fastrs(data=science_data)


# ================================
# Preprocessed FastRS Fixtures
# ================================


@pytest.fixture
def preprocessed_fastrs(fastrs_instance):
    """
    Provide Fastrs instance with completed preprocessing.

    Parameters
    ----------
    fastrs_instance : Fastrs
        Basic Fastrs instance.

    Returns
    -------
    Fastrs
        Fastrs instance with preprocessed data ready for training.
    """
    fastrs_instance.preprocess(option="default")
    return fastrs_instance


@pytest.fixture
def preprocessed_korean_fastrs(fastrs_korean):
    """
    Provide preprocessed Fastrs instance with Korean data.

    Parameters
    ----------
    fastrs_korean : Fastrs
        Fastrs instance with Korean text data.

    Returns
    -------
    Fastrs
        Preprocessed Fastrs instance with Korean data.
    """
    fastrs_korean.preprocess(option="default")
    return fastrs_korean


# ================================
# Mock Model Fixtures
# ================================


@pytest.fixture
def mock_fasttext_model():
    """
    Provide mock FastText model for testing.

    Returns
    -------
    Mock
        Mock FastText model with required attributes and methods.
    """
    mock_model = Mock(spec=FastText)
    mock_model.wv = Mock()
    mock_model.wv.vectors = np.random.rand(100, 50)  # 100 words, 50 dimensions
    mock_model.wv.key_to_index = {f"word_{i}": i for i in range(100)}

    # Mock training methods
    mock_model.build_vocab = Mock()
    mock_model.train = Mock()

    return mock_model


@pytest.fixture
def trained_fastrs(preprocessed_fastrs, mock_fasttext_model):
    """
    Provide Fastrs instance with trained model.

    Parameters
    ----------
    preprocessed_fastrs : Fastrs
        Preprocessed Fastrs instance.
    mock_fasttext_model : Mock
        Mock FastText model.

    Returns
    -------
    Fastrs
        Fastrs instance with mock trained model.
    """
    preprocessed_fastrs.model = mock_fasttext_model

    # Create mock jamodict for visualization
    all_responses = []
    for item in preprocessed_fastrs.items:
        all_responses.extend(item.original_response)

    preprocessed_fastrs.jamodict = {
        f"token_{i}": resp for i, resp in enumerate(all_responses)
    }

    return preprocessed_fastrs


# ================================
# Visualization Fixtures
# ================================


@pytest.fixture
def sample_coordinates():
    """
    Provide sample coordinate data for visualization testing.

    Returns
    -------
    pd.DataFrame
        Sample coordinate data with x, y, response, and token columns.
    """
    return pd.DataFrame(
        {
            "x": np.random.randn(50),
            "y": np.random.randn(50),
            "response": [f"response_{i % 10}" for i in range(50)],
            "token": [f"token_{i}" for i in range(50)],
        }
    )


@pytest.fixture
def reduced_fastrs(trained_fastrs, sample_coordinates):
    """
    Provide Fastrs instance with reduced coordinates.

    Parameters
    ----------
    trained_fastrs : Fastrs
        Fastrs instance with trained model.
    sample_coordinates : pd.DataFrame
        Sample coordinate data.

    Returns
    -------
    Fastrs
        Fastrs instance with reduced coordinates for visualization.
    """
    trained_fastrs.coordinates = sample_coordinates

    # Assign coordinates to each item
    coords_per_item = len(sample_coordinates) // len(trained_fastrs.items)
    for i, item in enumerate(trained_fastrs.items):
        start_idx = i * coords_per_item
        end_idx = min((i + 1) * coords_per_item, len(sample_coordinates))
        item.coordinates = (
            sample_coordinates.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        )

    return trained_fastrs


# ================================
# Item-Level Fixtures
# ================================


@pytest.fixture
def sample_item():
    """
    Provide sample Item instance.

    Returns
    -------
    Item
        Sample Item instance with Korean literature data.
    """
    return Item(
        name="sample_literature_item",
        answer=["허무", "허무감"],
        response=["허무", "허무감", "공허", "무상", "초월"],
        information="박목월의 시 「나그네」 분석 문제입니다.",
    )


@pytest.fixture
def preprocessed_item(sample_item):
    """
    Provide Item instance with preprocessed data.

    Parameters
    ----------
    sample_item : Item
        Basic Item instance.

    Returns
    -------
    Item
        Item instance with mock preprocessed attributes.
    """
    # Mock preprocessed data
    sample_item.answer = ["처리된답", "전처리답"]
    sample_item.response = ["처리된답", "전처리답", "처리된오답"]
    sample_item.information = "전처리된 정보입니다."

    # Mock preprocessing attributes
    sample_item.cleanparams = {
        "space": "forbid",
        "special": "forbid",
        "extra_forbid": [],
        "extra_allow": [],
    }
    sample_item.jamodict = {
        "처리된답": "ㅊㅓㄹㅣㄷㅚㄴㄷㅏㅂ",
        "전처리답": "ㅈㅓㄴㅊㅓㄹㅣㄷㅏㅂ",
        "처리된오답": "ㅊㅓㄹㅣㄷㅚㄴㅇㅗㄷㅏㅂ",
    }

    return sample_item


# ================================
# Array Data Fixtures
# ================================


@pytest.fixture
def sample_arrays():
    """
    Provide sample numpy arrays for Fastrs initialization.

    Returns
    -------
    tuple
        Tuple of (answers, responses, informations) numpy arrays.
    """
    answers = np.array([["답안1", "답안2"], ["답안3", "답안4"], ["답안5", "답안6"]])

    responses = np.array(
        [
            ["학생응답1", "학생응답2"],
            ["학생응답3", "학생응답4"],
            ["학생응답5", "학생응답6"],
        ]
    )

    informations = np.array(
        ["문제 정보 1입니다.", "문제 정보 2입니다.", "문제 정보 3입니다."]
    )

    return answers, responses, informations


# ================================
# Performance Testing Fixtures
# ================================


@pytest.fixture
def large_dataset():
    """
    Provide large dataset for performance testing.

    Returns
    -------
    dict
        Large dataset with many items for performance testing.
    """
    base_data = load_sample_data()
    large_data = {}

    # Create 50 items for performance testing
    for i in range(50):
        for j, (key, value) in enumerate(base_data.items()):
            new_key = f"{key}_perf_{i}_{j}"
            large_data[new_key] = {
                "information": f"{value['information']} (성능 테스트 {i}-{j})",
                "answer": value["answer"][:],
                "response": value["response"][:],
            }

    return large_data


@pytest.fixture
def large_fastrs(large_dataset):
    """
    Provide Fastrs instance with large dataset.

    Parameters
    ----------
    large_dataset : dict
        Large dataset for performance testing.

    Returns
    -------
    Fastrs
        Fastrs instance with large dataset.
    """
    return Fastrs(data=large_dataset)


# ================================
# Utility Fixtures
# ================================


@pytest.fixture
def temp_config_file(tmp_path):
    """
    Provide temporary configuration file for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture.

    Returns
    -------
    Path
        Path to temporary configuration file.
    """
    config_data = {
        "test_setting": "test_value",
        "number_setting": 42,
        "list_setting": ["item1", "item2", "item3"],
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    return config_file


@pytest.fixture
def mock_tokenizer():
    """
    Provide mock tokenizer for preprocessing tests.

    Returns
    -------
    Mock
        Mock tokenizer with morphs and nouns methods.
    """
    mock_tokenizer = Mock()
    mock_tokenizer.morphs.return_value = ["토큰1", "토큰2", "토큰3"]
    mock_tokenizer.nouns.return_value = ["명사1", "명사2"]
    return mock_tokenizer


# ================================
# Error Testing Fixtures
# ================================


@pytest.fixture
def invalid_data():
    """
    Provide invalid data for error testing.

    Returns
    -------
    dict
        Invalid data structures for testing error handling.
    """
    return {
        "invalid_answer_type": {
            "information": "Valid information",
            "answer": "not_a_list",  # Should be list
            "response": ["valid", "response"],
        },
        "mismatched_lengths": {
            "information": "Valid information",
            "answer": ["ans1", "ans2"],
            "response": ["resp1"],  # Different length
        },
        "non_string_elements": {
            "information": "Valid information",
            "answer": ["ans1", 42],  # Non-string element
            "response": ["resp1", "resp2"],
        },
    }


# ================================
# Cleanup Fixtures
# ================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatically clean up after each test.

    This fixture runs after each test to ensure clean state.
    """
    yield  # Test runs here

    # Cleanup code runs after test
    # Add any necessary cleanup here
    pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment at session start.

    This fixture runs once per test session to set up the testing environment.
    """
    # Setup code runs before any tests
    os.environ["FASTRS_TEST_MODE"] = "1"

    yield  # All tests run here

    # Teardown code runs after all tests
    if "FASTRS_TEST_MODE" in os.environ:
        del os.environ["FASTRS_TEST_MODE"]
