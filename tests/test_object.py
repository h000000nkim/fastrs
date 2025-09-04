"""
Tests for Fastrs main class
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from fastrs.core.object import Fastrs


class TestFastrsInitialization:
    """Fastrs 초기화 테스트"""
    
    def test_init_with_data_dict(self, sample_data):
        """딕셔너리 데이터로 초기화 테스트"""
        fastrs = Fastrs(data=sample_data, option="manual")
        
        assert fastrs.data == sample_data
        assert fastrs.option == "manual"
        assert hasattr(fastrs, 'fasttext_params')
        assert hasattr(fastrs, 'epochs')
        
    def test_init_with_arrays(self, sample_arrays):
        """배열 데이터로 초기화 테스트"""
        answers, responses, informations = sample_arrays
        
        fastrs = Fastrs(
            answers=answers,
            responses=responses,
            informations=informations,
            option="manual"
        )
        
        assert fastrs.data is not None
        assert isinstance(fastrs.data, dict)
        assert len(fastrs.data) == len(answers)
        
    def test_init_with_fasttext_preset(self, sample_data):
        """FastText 프리셋으로 초기화 테스트"""
        fastrs = Fastrs(
            data=sample_data,
            option="manual",
            fasttext_preset="fast_training"
        )
        
        # fast_training 프리셋이 적용되었는지 확인
        assert fastrs.fasttext_params["vector_size"] == 100
        assert fastrs.epochs == 3
        
    def test_init_with_custom_params(self, sample_data):
        """커스텀 파라미터로 초기화 테스트"""
        custom_vector_size = 200
        custom_epochs = 7
        
        fastrs = Fastrs(
            data=sample_data,
            option="manual",
            vector_size=custom_vector_size,
            epochs=custom_epochs
        )
        
        # 커스텀 파라미터가 적용되었는지 확인
        assert fastrs.fasttext_params["vector_size"] == custom_vector_size
        assert fastrs.epochs == custom_epochs
        
    @patch('fastrs.core.object.util.get_pretrained_model')
    @patch('fastrs.core.object.Preprocessor')
    def test_init_automatic_mode(self, mock_preprocessor, mock_get_model, sample_data):
        """자동 모드 초기화 테스트"""
        # Mock 설정
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.__iter__ = Mock(return_value=iter([["token1"], ["token2"]]))
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        fastrs = Fastrs(data=sample_data, option="automatic")
        
        # Preprocessor가 각 아이템에 대해 호출되었는지 확인
        assert mock_preprocessor.call_count == len(sample_data)
        assert hasattr(fastrs, 'traindata')
        assert hasattr(fastrs, 'sentences')


class TestFastrsDataValidation:
    """Fastrs 데이터 검증 테스트"""
    
    def test_invalid_data_raises_error(self):
        """잘못된 데이터로 초기화 시 에러 발생 테스트"""
        invalid_data = {
            "item1": {
                "answer": "not_a_list",  # 리스트가 아님
                "response": ["resp1"],
                "information": "test"
            }
        }
        
        with pytest.raises(Exception):  # UtilError 또는 다른 검증 에러
            Fastrs(data=invalid_data, option="manual")


class TestFastrsMethods:
    """Fastrs 메서드들 테스트"""
    
    @patch('fastrs.core.object.FastText')
    def test_train_method(self, mock_fasttext, sample_data):
        """train 메서드 테스트"""
        # Mock 설정
        mock_model = Mock()
        mock_fasttext.return_value = mock_model
        
        fastrs = Fastrs(data=sample_data, option="manual")
        fastrs.sentences = [["token1"], ["token2"]]  # 테스트용 문장
        
        result = fastrs.train()
        
        # FastText가 올바른 파라미터로 호출되었는지 확인
        mock_fasttext.assert_called_once_with(
            sentences=fastrs.sentences,
            **fastrs.fasttext_params
        )
        assert fastrs.model == mock_model
        assert result == mock_model
        
    @patch('fastrs.core.object.util.get_pretrained_model')
    def test_finetune_method(self, mock_get_model, sample_data):
        """finetune 메서드 테스트"""
        # Mock 설정
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        fastrs = Fastrs(data=sample_data, option="manual")
        fastrs.sentences = [["token1"], ["token2"]]
        
        result = fastrs.finetune()
        
        # 모델이 올바르게 파인튜닝되었는지 확인
        mock_model.build_vocab.assert_called_once()
        mock_model.train.assert_called_once()
        assert fastrs.model == mock_model
        assert result == mock_model
        
    @patch('fastrs.core.object.UMAP')
    def test_reduce_method_umap(self, mock_umap, sample_data):
        """reduce 메서드 UMAP 테스트"""
        # Mock 설정
        mock_reducer = Mock()
        mock_reducer.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_umap.return_value = mock_reducer
        
        fastrs = Fastrs(data=sample_data, option="manual")
        mock_model = Mock()
        mock_model.wv.vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
        fastrs.model = mock_model
        
        result = fastrs.reduce(method="umap")
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 2
        
    @patch('fastrs.core.object.PCA')
    def test_reduce_method_pca(self, mock_pca, sample_data):
        """reduce 메서드 PCA 테스트"""
        # Mock 설정
        mock_reducer = Mock()
        mock_reducer.fit_transform.return_value = np.array([[1, 2], [3, 4]])
        mock_pca.return_value = mock_reducer
        
        fastrs = Fastrs(data=sample_data, option="manual")
        mock_model = Mock()
        mock_model.wv.vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
        fastrs.model = mock_model
        
        result = fastrs.reduce(method="pca")
        
        assert isinstance(result, pd.DataFrame)
        
    def test_reduce_method_invalid(self, sample_data):
        """reduce 메서드 잘못된 방법 테스트"""
        fastrs = Fastrs(data=sample_data, option="manual")
        mock_model = Mock()
        fastrs.model = mock_model
        
        with pytest.raises(ValueError, match="Unknown method"):
            fastrs.reduce(method="invalid_method")
            
    def test_visualize_no_model_error(self, sample_data):
        """모델 없이 시각화 시 에러 테스트"""
        fastrs = Fastrs(data=sample_data, option="manual")
        fastrs.model = None
        
        with pytest.raises(ValueError, match="모델이 없습니다"):
            fastrs.visualize(answers=["test"])
            
    @patch('fastrs.core.object.visualize_embeddings')
    def test_visualize_method(self, mock_visualize, sample_data):
        """visualize 메서드 테스트"""
        # Mock 설정
        mock_figure = Mock()
        mock_visualize.return_value = mock_figure
        
        fastrs = Fastrs(data=sample_data, option="manual")
        mock_model = Mock()
        mock_model.wv.vectors = np.array([[0.1, 0.2], [0.3, 0.4]])
        fastrs.model = mock_model
        
        # reduce 메서드 mock
        with patch.object(fastrs, 'reduce') as mock_reduce:
            mock_reduce.return_value = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            
            result = fastrs.visualize(answers=["좋은"], show=False)
            
            # reduce가 호출되었는지 확인
            mock_reduce.assert_called_once()
            # visualize_embeddings가 호출되었는지 확인
            mock_visualize.assert_called_once()
            assert result == mock_figure
            
    def test_visualize_auto_extract_answers(self, sample_data):
        """답변 자동 추출 시각화 테스트"""
        fastrs = Fastrs(data=sample_data, option="manual")
        mock_model = Mock()
        fastrs.model = mock_model
        
        with patch.object(fastrs, 'reduce') as mock_reduce, \
             patch('fastrs.core.object.visualize_embeddings') as mock_visualize:
            
            mock_reduce.return_value = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            
            fastrs.visualize(answers=None, show=False)  # answers=None
            
            # visualize_embeddings가 호출되었고, answers가 자동 추출되었는지 확인
            mock_visualize.assert_called_once()
            call_args = mock_visualize.call_args
            extracted_answers = call_args[1]['answers']  # keyword arguments
            assert isinstance(extracted_answers, list)
            assert len(extracted_answers) <= 5  # 최대 5개까지
