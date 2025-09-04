"""
Tests for preprocessor module
"""
import pytest
from fastrs.core.preprocessor import Preprocessor


class TestPreprocessor:
    """Preprocessor 클래스 테스트"""
    
    def test_init_automatic(self, sample_preprocessor_data):
        """자동 모드 초기화 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="automatic"
        )
        
        assert preprocessor.answers == sample_preprocessor_data["answers"]
        assert preprocessor.responses == sample_preprocessor_data["responses"]
        assert preprocessor.information == sample_preprocessor_data["information"]
        assert preprocessor.option == "automatic"
        
    def test_init_manual(self, sample_preprocessor_data):
        """수동 모드 초기화 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        assert preprocessor.option == "manual"
        
    def test_clean_text_basic(self, sample_preprocessor_data):
        """기본 텍스트 클리닝 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        # 특수문자가 포함된 텍스트
        dirty_text = "안녕하세요!!! 좋은 하루입니다... @#$"
        cleaned = preprocessor.clean(dirty_text)
        
        # 특수문자가 제거되었는지 확인
        assert "!" not in cleaned
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned
        
    def test_clean_text_with_options(self, sample_preprocessor_data):
        """옵션별 텍스트 클리닝 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        test_text = "  안녕하세요!!!  좋은\t하루입니다  "
        
        # 공백 정리 옵션
        cleaned_space = preprocessor.clean(test_text, space=True)
        assert "  " not in cleaned_space  # 연속 공백 제거
        
        # 탭 정리 옵션
        cleaned_tabs = preprocessor.clean(test_text, tabs=True)
        assert "\t" not in cleaned_tabs
        
    def test_tokenize_text(self, sample_preprocessor_data):
        """텍스트 토큰화 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        text = "안녕하세요 좋은 하루입니다"
        tokens = preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # 각 토큰이 문자열인지 확인
        assert all(isinstance(token, str) for token in tokens)
        
    def test_jamoize_text(self, sample_preprocessor_data):
        """자모 분리 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        text = "안녕"
        jamoized = preprocessor.jamoize(text)
        
        assert isinstance(jamoized, str)
        assert len(jamoized) > len(text)  # 자모 분리로 길이가 늘어남
        
    def test_format_processing(self, sample_preprocessor_data):
        """포맷 처리 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="manual"
        )
        
        text = "안녕하세요"
        formatted = preprocessor.format(text)
        
        assert isinstance(formatted, list)
        assert len(formatted) > 0
        
    def test_iteration_protocol(self, sample_preprocessor_data):
        """반복자 프로토콜 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="automatic"
        )
        
        # 반복 가능한지 확인
        sentences = list(preprocessor)
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        
        # 각 문장이 토큰 리스트인지 확인
        for sentence in sentences:
            assert isinstance(sentence, list)
            assert all(isinstance(token, str) for token in sentence)
            
    def test_string_input_conversion(self):
        """문자열 입력 자동 변환 테스트"""
        # answers가 문자열로 입력된 경우
        preprocessor = Preprocessor(
            answers="좋은",  # 문자열
            responses=["매우 좋음"],
            information="테스트",
            option="manual"
        )
        
        assert isinstance(preprocessor.answers, list)
        assert preprocessor.answers == ["좋은"]
        
    def test_none_information_handling(self):
        """information이 None인 경우 처리 테스트"""
        preprocessor = Preprocessor(
            answers=["좋은"],
            responses=["매우 좋음"],
            information=None,  # None
            option="manual"
        )
        
        assert preprocessor.information is None
