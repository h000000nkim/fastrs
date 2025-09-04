"""
Tests for exceptions module
"""
import pytest
from fastrs.core.exceptions import FastrsError, PreprocessingError, TrainingError, UtilError


class TestExceptions:
    """예외 클래스들 테스트"""
    
    def test_fastrs_error_base(self):
        """기본 FastrsError 테스트"""
        error = FastrsError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)
        
    def test_preprocessing_error(self):
        """PreprocessingError 테스트"""
        error = PreprocessingError("Preprocessing failed")
        assert str(error) == "Preprocessing failed"
        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        
    def test_training_error(self):
        """TrainingError 테스트"""
        error = TrainingError("Training failed")
        assert str(error) == "Training failed"
        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        
    def test_util_error(self):
        """UtilError 테스트"""
        error = UtilError("Utility function failed")
        assert str(error) == "Utility function failed"
        assert isinstance(error, FastrsError)
        assert isinstance(error, Exception)
        
    def test_exception_hierarchy(self):
        """예외 계층구조 테스트"""
        # 모든 커스텀 예외가 FastrsError를 상속하는지 확인
        assert issubclass(PreprocessingError, FastrsError)
        assert issubclass(TrainingError, FastrsError)
        assert issubclass(UtilError, FastrsError)
        
        # FastrsError가 Exception을 상속하는지 확인
        assert issubclass(FastrsError, Exception)
        
    def test_raising_exceptions(self):
        """예외 발생 테스트"""
        with pytest.raises(PreprocessingError):
            raise PreprocessingError("Test preprocessing error")
            
        with pytest.raises(TrainingError):
            raise TrainingError("Test training error")
            
        with pytest.raises(UtilError):
            raise UtilError("Test util error")
            
        with pytest.raises(FastrsError):
            raise FastrsError("Test base error")
            
    def test_catching_base_exception(self):
        """기본 예외로 하위 예외 잡기 테스트"""
        try:
            raise PreprocessingError("Specific error")
        except FastrsError as e:
            assert str(e) == "Specific error"
        else:
            pytest.fail("FastrsError should catch PreprocessingError")
            
        try:
            raise TrainingError("Training specific error")
        except FastrsError as e:
            assert str(e) == "Training specific error"
        else:
            pytest.fail("FastrsError should catch TrainingError")
            
        try:
            raise UtilError("Util specific error")
        except FastrsError as e:
            assert str(e) == "Util specific error"
        else:
            pytest.fail("FastrsError should catch UtilError")
