"""
pytest configuration and fixtures
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
    return {
        "item1": {
            "answer": ["좋은", "나쁜", "보통"],
            "response": ["매우 좋음", "좋음", "나쁨"],
            "information": "감정 분석 테스트"
        },
        "item2": {
            "answer": ["빠른", "느린"],
            "response": ["매우 빠름", "빠름", "느림", "매우 느림"],
            "information": "속도 평가 테스트"
        }
    }


@pytest.fixture
def sample_arrays():
    """테스트용 배열 데이터"""
    answers = np.array([
        ["좋은", "나쁜", "보통"],
        ["빠른", "느린"]
    ], dtype=object)
    
    responses = np.array([
        ["매우 좋음", "좋음", "나쁨"],
        ["매우 빠름", "빠름", "느림", "매우 느림"]
    ], dtype=object)
    
    informations = np.array([
        "감정 분석 테스트",
        "속도 평가 테스트"
    ])
    
    return answers, responses, informations


@pytest.fixture
def sample_preprocessor_data():
    """전처리기 테스트용 데이터"""
    return {
        "answers": ["좋은", "나쁜"],
        "responses": ["매우 좋음!", "정말 나쁘다..."],
        "information": "테스트 데이터"
    }


@pytest.fixture
def sample_embedding_dataframe():
    """시각화 테스트용 임베딩 데이터프레임"""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.5, 2.5, 3.5, 4.5, 5.5],
        "token": ["좋은", "나쁜", "빠른", "느린", "보통"],
        "label": ["1", "0", "1", "0", "2"],
        "answer_vis": ['"좋은"', '""', '"빠른"', '""', '""'],
        "token_vis": ['"좋은"', '"나쁜"', '"빠른"', '"느린"', '"보통"'],
        "answer": ["좋은", None, "빠른", None, None]
    })
