"""
Tests for util module
"""
import pytest
import numpy as np
import json
import os
from fastrs.core import util
from fastrs.core.exceptions import UtilError


class TestConfigLoaders:
    """설정 로더 함수들 테스트"""
    
    def test_load_color_schemes(self):
        """색상 스키마 로딩 테스트"""
        color_schemes = util.load_color_schemes()
        
        assert isinstance(color_schemes, dict)
        assert "default" in color_schemes
        assert "colorblind_friendly" in color_schemes
        
        # default 스키마 구조 확인
        default = color_schemes["default"]
        assert "label_colors" in default
        assert "answer_marker" in default
        assert "annotation" in default
        
    def test_load_plot_config(self):
        """플롯 설정 로딩 테스트"""
        plot_config = util.load_plot_config()
        
        assert isinstance(plot_config, dict)
        assert "template" in plot_config
        assert "width" in plot_config
        assert "height" in plot_config
        assert "marker_size" in plot_config
        
    def test_load_reduction_defaults(self):
        """차원축소 기본값 로딩 테스트"""
        reduction_defaults = util.load_reduction_defaults()
        
        assert isinstance(reduction_defaults, dict)
        assert "umap" in reduction_defaults
        assert "pca" in reduction_defaults
        assert "tsne" in reduction_defaults
        
    def test_load_fasttext_defaults(self):
        """FastText 기본값 로딩 테스트"""
        fasttext_defaults = util.load_fasttext_defaults()
        
        assert isinstance(fasttext_defaults, dict)
        assert "default" in fasttext_defaults
        assert "fast_training" in fasttext_defaults
        assert "high_quality" in fasttext_defaults
        
        # default 설정 구조 확인
        default = fasttext_defaults["default"]
        assert "vector_size" in default
        assert "window" in default
        assert "epochs" in default
        
    def test_load_config_invalid_file(self):
        """존재하지 않는 설정 파일 로딩 테스트"""
        with pytest.raises(UtilError):
            util.load_config("nonexistent_config")


class TestDataFormatting:
    """데이터 포맷팅 함수들 테스트"""
    
    def test_format_data_valid(self, sample_arrays):
        """유효한 데이터 포맷팅 테스트"""
        answers, responses, informations = sample_arrays
        
        result = util.formatData(answers, responses, informations)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "item1" in result
        assert "item2" in result
        
        # 첫 번째 아이템 구조 확인
        item1 = result["item1"]
        assert "answer" in item1
        assert "response" in item1
        assert "information" in item1
        
    def test_format_data_length_mismatch(self):
        """길이가 맞지 않는 데이터 포맷팅 테스트"""
        answers = np.array(["test1"])
        responses = np.array(["resp1", "resp2"])  # 길이 다름
        informations = np.array(["info1"])
        
        with pytest.raises(UtilError):
            util.formatData(answers, responses, informations)
            
    def test_format_data_none_input(self):
        """None 입력 데이터 포맷팅 테스트"""
        with pytest.raises(UtilError):
            util.formatData(None, None, None)


class TestDataValidation:
    """데이터 검증 함수들 테스트"""
    
    def test_valid_data_success(self, sample_data):
        """유효한 데이터 검증 테스트"""
        # 예외가 발생하지 않아야 함
        util.validData(sample_data)
        
    def test_valid_data_invalid_answer_type(self):
        """잘못된 answer 타입 검증 테스트"""
        invalid_data = {
            "item1": {
                "answer": "not_a_list",  # 리스트가 아님
                "response": ["resp1"],
                "information": "test"
            }
        }
        
        with pytest.raises(UtilError):
            util.validData(invalid_data)
            
    def test_valid_data_length_mismatch(self):
        """answer와 response 길이 불일치 검증 테스트"""
        invalid_data = {
            "item1": {
                "answer": ["ans1", "ans2"],
                "response": ["resp1"],  # 길이 다름
                "information": "test"
            }
        }
        
        with pytest.raises(UtilError):
            util.validData(invalid_data)
            
    def test_valid_data_empty_answer(self):
        """빈 answer 검증 테스트"""
        invalid_data = {
            "item1": {
                "answer": [],  # 빈 리스트
                "response": [],
                "information": "test"
            }
        }
        
        with pytest.raises(UtilError):
            util.validData(invalid_data)
            
    def test_valid_data_invalid_information_type(self):
        """잘못된 information 타입 검증 테스트"""
        invalid_data = {
            "item1": {
                "answer": ["ans1"],
                "response": ["resp1"],
                "information": 123  # 문자열이 아님
            }
        }
        
        with pytest.raises(UtilError):
            util.validData(invalid_data)


class TestUtilityFunctions:
    """기타 유틸리티 함수들 테스트"""
    
    def test_copy_signature_decorator(self):
        """함수 시그니처 복사 데코레이터 테스트"""
        def original_func(a: int, b: str = "default") -> str:
            """Original function docstring"""
            return f"{a}_{b}"
            
        @util.copysignature(original_func)
        def decorated_func(*args, **kwargs):
            return "decorated"
            
        # 시그니처가 복사되었는지 확인
        assert decorated_func.__signature__ == original_func.__signature__
        assert decorated_func.__doc__ == original_func.__doc__
