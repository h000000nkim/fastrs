"""
Tests for visualizer module
"""
import pytest
import pandas as pd
import plotly.graph_objects as go
from fastrs.core.visualizer import visualize_embeddings, create_scatter_plot


class TestVisualizerFunctions:
    """시각화 함수들 테스트"""
    
    def test_visualize_embeddings_basic(self, sample_embedding_dataframe):
        """기본 임베딩 시각화 테스트"""
        fig = visualize_embeddings(
            df=sample_embedding_dataframe,
            answers=["좋은", "빠른"],
            show=False  # 실제로 표시하지 않음
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.data is not None
        assert len(fig.data) > 0
        
    def test_visualize_embeddings_string_answer(self, sample_embedding_dataframe):
        """문자열 답변으로 시각화 테스트"""
        fig = visualize_embeddings(
            df=sample_embedding_dataframe,
            answers="좋은",  # 문자열
            show=False
        )
        
        assert isinstance(fig, go.Figure)
        
    def test_visualize_embeddings_colorblind_friendly(self, sample_embedding_dataframe):
        """색맹 친화 색상으로 시각화 테스트"""
        fig = visualize_embeddings(
            df=sample_embedding_dataframe,
            answers=["좋은"],
            color_scheme="colorblind_friendly",
            show=False
        )
        
        assert isinstance(fig, go.Figure)
        
    def test_visualize_embeddings_custom_title(self, sample_embedding_dataframe):
        """커스텀 제목으로 시각화 테스트"""
        custom_title = "테스트 임베딩 시각화"
        fig = visualize_embeddings(
            df=sample_embedding_dataframe,
            answers=["좋은"],
            title=custom_title,
            show=False
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == custom_title
        
    def test_visualize_embeddings_missing_columns(self):
        """필수 컬럼이 없는 DataFrame 테스트"""
        incomplete_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [1, 2, 3]
            # 필수 컬럼들 누락
        })
        
        with pytest.raises(ValueError, match="DataFrame에 필수 컬럼이 없습니다"):
            visualize_embeddings(
                df=incomplete_df,
                answers=["test"],
                show=False
            )
            
    def test_create_scatter_plot_basic(self, sample_embedding_dataframe):
        """기본 산점도 생성 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=["좋은", "빠른"]
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # 레이아웃 설정 확인
        assert fig.layout.legend.title.text == "레이블"
        assert fig.layout.yaxis.scaleanchor == "x"
        
    def test_create_scatter_plot_string_answer(self, sample_embedding_dataframe):
        """문자열 답변으로 산점도 생성 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers="좋은"  # 문자열
        )
        
        assert isinstance(fig, go.Figure)
        
    def test_create_scatter_plot_default_title(self, sample_embedding_dataframe):
        """기본 제목으로 산점도 생성 테스트"""
        answers = ["좋은", "빠른"]
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=answers
        )
        
        expected_title = f"임베딩 시각화 - {', '.join(answers)}"
        assert fig.layout.title.text == expected_title
        
    def test_create_scatter_plot_with_colorblind_scheme(self, sample_embedding_dataframe):
        """색맹 친화 색상으로 산점도 생성 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=["좋은"],
            color_scheme="colorblind_friendly"
        )
        
        assert isinstance(fig, go.Figure)
        
    def test_grid_lines_added(self, sample_embedding_dataframe):
        """격자선이 추가되는지 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=["좋은"]
        )
        
        # 격자선 확인 (hline, vline)
        shapes = fig.layout.shapes
        assert len(shapes) >= 2  # 최소 수평선, 수직선
        
    def test_answer_markers_and_annotations(self, sample_embedding_dataframe):
        """정답 마커와 주석이 추가되는지 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=["좋은", "빠른"]
        )
        
        # 정답 마커가 추가되었는지 확인
        trace_names = [trace.name for trace in fig.data]
        assert any("정답: 좋은" in name for name in trace_names)
        assert any("정답: 빠른" in name for name in trace_names)
        
        # 주석이 추가되었는지 확인
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        
    def test_label_based_grouping(self, sample_embedding_dataframe):
        """레이블별 그룹화가 올바른지 테스트"""
        fig = create_scatter_plot(
            df=sample_embedding_dataframe,
            answers=["좋은"]
        )
        
        # 레이블별 trace가 생성되었는지 확인
        trace_names = [trace.name for trace in fig.data if trace.name.startswith("label=")]
        expected_labels = ["label=0", "label=1", "label=2"]
        
        for label in expected_labels:
            # 해당 레이블의 데이터가 있는 경우에만 trace가 생성됨
            if len(sample_embedding_dataframe[sample_embedding_dataframe["label"] == label.split("=")[1]]) > 0:
                assert label in trace_names
