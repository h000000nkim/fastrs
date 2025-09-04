import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union, List, Dict, Optional, Literal
from . import util


def create_scatter_plot(
    df: pd.DataFrame,
    answers: Union[str, List[str]],
    color_scheme: Literal["default", "colorblind_friendly"] = "default",
    title: Optional[str] = None
) -> go.Figure:
    """
    DataFrame으로부터 산점도 생성
    
    Args:
        df: 시각화할 데이터프레임 (x, y, token, label, answer_vis, token_vis 컬럼 필요)
        answers: 시각화할 정답들
        color_scheme: 색상 스키마 ("default", "colorblind_friendly")
        title: 그래프 제목
    
    Returns:
        plotly Figure 객체
    """
    if isinstance(answers, str): answers = [answers]
    
    color_schemes = util.load_color_schemes()
    plot_config = util.load_plot_config()
    
    colors = color_schemes[color_scheme]
    
    fig = go.Figure()
    
    # 레이블별로 점들 그리기
    draw_order = ["0", "1", "2"]
    for lab in draw_order:
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
            
        opacity = plot_config["marker_opacity"]["faded"] if lab == "0" else plot_config["marker_opacity"]["normal"]
        
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"],
            mode="markers",
            name=f"label={lab}",
            marker=dict(
                size=plot_config["marker_size"],
                color=colors["label_colors"][lab],
                opacity=opacity
            ),
            customdata=np.stack([
                sub["answer_vis"].values, 
                sub["token_vis"].values, 
                sub["label"].values
            ], axis=-1),
            hovertemplate="<b>answer: %{customdata[0]}</b><br>token: %{customdata[1]}<br>label: %{customdata[2]}<extra></extra>"
        ))
    
    # 격자선 추가
    grid_config = plot_config["grid_lines"]
    fig.add_hline(y=0, line_dash=grid_config["dash"], line_color=grid_config["color"], opacity=grid_config["opacity"])
    fig.add_vline(x=0, line_dash=grid_config["dash"], line_color=grid_config["color"], opacity=grid_config["opacity"])
    
    # 정답 마커와 주석 추가
    marker_config = colors["answer_marker"]
    annotation_config = colors["annotation"]
    
    for original_ans in answers:
        answer_rows = df[df["answer"] == original_ans]
        if not answer_rows.empty:
            row = answer_rows.iloc[0]
            
            # 마커 추가
            fig.add_trace(go.Scatter(
                x=[row["x"]], y=[row["y"]],
                mode="markers",
                marker=dict(
                    size=marker_config["size"],
                    symbol=marker_config["symbol"],
                    color=marker_config["color"],
                    line=dict(width=1)
                ),
                name=f"정답: {original_ans}",
                hovertemplate=f"<b>정답</b>: {original_ans}<extra></extra>"
            ))
            
            # 주석 추가
            fig.add_annotation(
                x=row["x"], y=row["y"],
                text=original_ans,
                showarrow=True, 
                arrowhead=2, 
                ax=28, ay=-28,
                bgcolor=annotation_config["bgcolor"],
                bordercolor=annotation_config["bordercolor"],
                font=dict(
                    color=annotation_config["font_color"],
                    size=annotation_config["font_size"]
                )
            )
    
    # 레이아웃 설정
    if title is None:
        title = f"임베딩 시각화 - {', '.join(answers)}"
    
    fig.update_layout(
        title=title,
        legend_title_text="레이블",
        xaxis_title=None,
        yaxis_title=None,
        template=plot_config["template"],
        width=plot_config["width"],
        height=plot_config["height"],
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend_traceorder="normal"
    )
    
    return fig


def visualize_embeddings(
    df: pd.DataFrame,
    answers: Union[str, List[str]],
    color_scheme: Literal["default", "colorblind_friendly"] = "default",
    title: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    임베딩 시각화 메인 함수
    
    Args:
        df: 시각화할 데이터프레임 (x, y, token, label, answer_vis, token_vis, answer 컬럼 필요)
        answers: 시각화할 정답들 
        color_scheme: 색상 스키마 ("default", "colorblind_friendly")
        title: 그래프 제목
        show: 그래프 표시 여부
    
    Returns:
        plotly Figure 객체
    """
    # 필수 컬럼 확인
    required_cols = ["x", "y", "token", "label", "answer_vis", "token_vis", "answer"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame에 필수 컬럼이 없습니다: {missing_cols}")
    
    # 시각화
    fig = create_scatter_plot(df, answers, color_scheme, title)
    
    if show:
        fig.show()
    
    return fig
