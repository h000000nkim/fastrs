"""
Integration tests for FastRS package
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock
from fastrs import Fastrs, Preprocessor, util


class TestIntegration:
    """통합 테스트"""
    
    def test_package_imports(self):
        """패키지 import 테스트"""
        # 모든 주요 컴포넌트가 import되는지 확인
        assert Fastrs is not None
        assert Preprocessor is not None
        assert util is not None
        
    def test_config_loading_integration(self):
        """설정 로딩 통합 테스트"""
        # 모든 설정 파일이 올바르게 로드되는지 확인
        color_schemes = util.load_color_schemes()
        plot_config = util.load_plot_config()
        reduction_defaults = util.load_reduction_defaults()
        fasttext_defaults = util.load_fasttext_defaults()
        
        assert isinstance(color_schemes, dict)
        assert isinstance(plot_config, dict)
        assert isinstance(reduction_defaults, dict)
        assert isinstance(fasttext_defaults, dict)
        
    def test_preprocessor_integration(self, sample_preprocessor_data):
        """전처리기 통합 테스트"""
        preprocessor = Preprocessor(
            answers=sample_preprocessor_data["answers"],
            responses=sample_preprocessor_data["responses"],
            information=sample_preprocessor_data["information"],
            option="automatic"
        )
        
        # 전체 파이프라인이 동작하는지 확인
        sentences = list(preprocessor)
        assert len(sentences) > 0
        assert all(isinstance(sentence, list) for sentence in sentences)
        
    @patch('fastrs.core.object.util.get_pretrained_model')
    @patch('fastrs.core.object.Preprocessor')
    def test_fastrs_full_pipeline_mock(self, mock_preprocessor, mock_get_model, sample_data):
        """Fastrs 전체 파이프라인 테스트 (Mock 사용)"""
        # Mock 설정
        mock_model = Mock()
        mock_model.wv.vectors = np.random.rand(10, 300)  # 임의의 벡터
        mock_get_model.return_value = mock_model
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.__iter__ = Mock(return_value=iter([
            ["좋은"], ["나쁜"], ["보통"]
        ]))
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Fastrs 생성 및 자동 처리
        fastrs = Fastrs(data=sample_data, option="automatic")
        
        # 속성들이 올바르게 설정되었는지 확인
        assert fastrs.data == sample_data
        assert hasattr(fastrs, 'traindata')
        assert hasattr(fastrs, 'sentences')
        assert hasattr(fastrs, 'fasttext_params')
        
    def test_data_format_and_validation_flow(self, sample_arrays):
        """데이터 포맷팅과 검증 흐름 테스트"""
        answers, responses, informations = sample_arrays
        
        # 데이터 포맷팅
        formatted_data = util.formatData(answers, responses, informations)
        
        # 데이터 검증
        util.validData(formatted_data)  # 예외가 발생하지 않아야 함
        
        # Fastrs로 사용
        fastrs = Fastrs(data=formatted_data, option="manual")
        assert fastrs.data == formatted_data
        
    def test_config_and_fastrs_integration(self, sample_data):
        """설정과 Fastrs 통합 테스트"""
        # 다양한 프리셋으로 Fastrs 생성
        fastrs_default = Fastrs(
            data=sample_data, 
            option="manual",
            fasttext_preset="default"
        )
        
        fastrs_fast = Fastrs(
            data=sample_data,
            option="manual", 
            fasttext_preset="fast_training"
        )
        
        # 프리셋별로 다른 설정이 적용되었는지 확인
        assert fastrs_default.fasttext_params["vector_size"] == 300
        assert fastrs_fast.fasttext_params["vector_size"] == 100
        
        assert fastrs_default.epochs == 5
        assert fastrs_fast.epochs == 3
        
    @patch('fastrs.core.visualizer.util.load_color_schemes')
    @patch('fastrs.core.visualizer.util.load_plot_config')
    def test_visualizer_config_integration(self, mock_plot_config, mock_color_schemes, sample_embedding_dataframe):
        """시각화와 설정 통합 테스트"""
        # Mock 설정
        mock_color_schemes.return_value = {
            "default": {
                "label_colors": {"0": "#000", "1": "#111", "2": "#222"},
                "answer_marker": {"color": "#FFF", "size": 10, "symbol": "circle"},
                "annotation": {"bgcolor": "white", "bordercolor": "black", "font_color": "black", "font_size": 10}
            }
        }
        mock_plot_config.return_value = {
            "template": "plotly_white",
            "width": 800,
            "height": 600,
            "marker_size": 8,
            "marker_opacity": {"normal": 0.8, "faded": 0.3},
            "grid_lines": {"color": "#CCC", "dash": "dot", "opacity": 0.5}
        }
        
        from fastrs.core.visualizer import visualize_embeddings
        
        fig = visualize_embeddings(
            df=sample_embedding_dataframe,
            answers=["좋은"],
            show=False
        )
        
        # 설정 로더가 호출되었는지 확인
        mock_color_schemes.assert_called()
        mock_plot_config.assert_called()
        
        assert fig is not None
        
    def test_error_handling_integration(self):
        """에러 처리 통합 테스트"""
        from fastrs.core.exceptions import UtilError
        
        # 잘못된 데이터로 Fastrs 생성 시 에러 발생
        invalid_data = {
            "item1": {
                "answer": 123,  # 잘못된 타입
                "response": ["resp"],
                "information": "test"
            }
        }
        
        with pytest.raises(UtilError):
            Fastrs(data=invalid_data, option="manual")
            
    def test_end_to_end_workflow_simulation(self, sample_data):
        """종단간 워크플로우 시뮬레이션"""
        with patch('fastrs.core.object.util.get_pretrained_model') as mock_get_model, \
             patch('fastrs.core.object.Preprocessor') as mock_preprocessor, \
             patch('fastrs.core.object.visualize_embeddings') as mock_visualize:
            
            # Mock 설정
            mock_model = Mock()
            mock_model.wv.vectors = np.random.rand(10, 300)
            mock_get_model.return_value = mock_model
            
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.__iter__ = Mock(return_value=iter([["token1"], ["token2"]]))
            mock_preprocessor.return_value = mock_preprocessor_instance
            
            mock_visualize.return_value = Mock()
            
            # 1. Fastrs 생성 (자동 모드)
            fastrs = Fastrs(data=sample_data, option="automatic")
            
            # 2. 차원 축소
            with patch.object(fastrs, 'reduce') as mock_reduce:
                import pandas as pd
                mock_reduce.return_value = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
                
                # 3. 시각화
                result = fastrs.visualize(answers=["좋은"], show=False)
                
                # 모든 단계가 실행되었는지 확인
                assert mock_preprocessor.called
                assert mock_reduce.called
                assert mock_visualize.called
                assert result is not None
