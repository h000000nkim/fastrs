import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union, List, Literal
from . import util


def scatter(
    df: pd.DataFrame,
    answers: Union[str, List[str]],
    title: Union[str, None] = None,
    theme: Literal["default", "colorblind_friendly"] = "default",
    scatter_type: str = "simple",
    show: bool = False,
) -> go.Figure:
    """
    Create interactive scatter plot visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'x', 'y', 'response', 'token' columns.
    answers : str or list of str
        Answer strings to highlight in the visualization.
    title : str, optional
        Plot title.
    theme : {"default", "colorblind_friendly"}, default="default"
        Color theme for the plot.
    scatter_type : str, default="simple"
        Type of scatter plot: "simple", "valuecount", "labeled", "combined".
    show : bool, default=False
        Whether to display the plot immediately.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter plot figure.
    """
    if answers is None:
        raise ValueError("Answers cannot be None")

    if isinstance(answers, str):
        answers = [answers]

    if len(answers) == 0:
        raise ValueError("Answers list cannot be empty")

    if scatter_type == "simple":
        fig = _simple_scatter(df, answers, title, theme)
    elif scatter_type == "valuecount":
        fig = _valuecount_scatter(df, answers, title, theme)
    elif scatter_type == "labeled":
        fig = _labeled_scatter(df, answers, title, theme)
    elif scatter_type == "combined":
        fig = _combined_scatter(df, answers, title, theme)
    else:
        raise ValueError(f"Unknown scatter_type: {scatter_type}")

    if show:
        fig.show()

    return fig


def _simple_scatter(
    df: pd.DataFrame,
    answers: List[str],
    title: Union[str, None] = None,
    theme: str = "default",
) -> go.Figure:
    """
    Basic 2D scatterplot
    Required columns: ["response", "token", "x", "y"]
    """
    color_schemes = util.load_color_schemes()
    colors = color_schemes[theme]
    plot_config = util.load_plot_config()

    df_viz = df.copy()

    first_answer = answers[0]
    origin_point = (
        df_viz[df_viz["response"] == first_answer].iloc[0]
        if not df_viz[df_viz["response"] == first_answer].empty
        else None
    )

    if origin_point is not None:
        df_viz["x"] = df_viz["x"] - origin_point["x"]
        df_viz["y"] = df_viz["y"] - origin_point["y"]

    answer_mask = df_viz["response"].isin(answers)
    answers_df = df_viz[answer_mask]
    non_answers_df = df_viz[~answer_mask]

    fig = go.Figure()

    if not non_answers_df.empty:
        fig.add_trace(
            go.Scatter(
                x=non_answers_df["x"],
                y=non_answers_df["y"],
                mode="markers",
                name="responses",
                marker=dict(
                    size=plot_config["marker_size"],
                    color="lightgray",
                    opacity=plot_config["marker_opacity"]["faded"],
                ),
                customdata=np.stack(
                    [
                        non_answers_df["response"].values,
                        non_answers_df["token"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate="<b>response: %{customdata[0]}</b><br>token: %{customdata[1]}<extra></extra>",
            )
        )

    if not answers_df.empty:
        fig.add_trace(
            go.Scatter(
                x=answers_df["x"],
                y=answers_df["y"],
                mode="markers+text",
                name="answers",
                text=answers_df["response"],
                textposition="middle center",
                marker=dict(
                    size=plot_config["marker_size"] * 1.5,
                    color=colors["label_colors"]["1"],
                    opacity=plot_config["marker_opacity"]["normal"],
                ),
                textfont=dict(size=12, color="white"),
                hovertemplate="<b>Answer: %{text}</b><extra></extra>",
            )
        )

    if title is None:
        title = f"Embedding Visualization - {', '.join(answers)}"

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        template=plot_config["template"],
        width=plot_config["width"],
        height=plot_config["height"],
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True,
    )

    return fig


def _valuecount_scatter(
    df: pd.DataFrame,
    answers: List[str],
    title: Union[str, None] = None,
    theme: str = "default",
) -> go.Figure:
    """
    3D scatter plot with count information
    Required columns: ["response", "token", "x", "y", "count"]
    """
    color_schemes = util.load_color_schemes()
    colors = color_schemes[theme]
    plot_config = util.load_plot_config()

    df_viz = df.copy()

    first_answer = answers[0]
    origin_point = (
        df_viz[df_viz["response"] == first_answer].iloc[0]
        if not df_viz[df_viz["response"] == first_answer].empty
        else None
    )

    if origin_point is not None:
        df_viz["x"] = df_viz["x"] - origin_point["x"]
        df_viz["y"] = df_viz["y"] - origin_point["y"]

    answer_mask = df_viz["response"].isin(answers)
    answers_df = df_viz[answer_mask]
    non_answers_df = df_viz[~answer_mask]

    fig = go.Figure()

    if not non_answers_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=non_answers_df["x"],
                y=non_answers_df["y"],
                z=non_answers_df["count"],
                mode="markers",
                name="responses",
                marker=dict(
                    size=plot_config["marker_size"],
                    color="lightgray",
                    opacity=plot_config["marker_opacity"]["faded"],
                ),
                customdata=np.stack(
                    [
                        non_answers_df["response"].values,
                        non_answers_df["token"].values,
                        non_answers_df["count"].values,
                    ],
                    axis=-1,
                ),
                hovertemplate="<b>response: %{customdata[0]}</b><br>token: %{customdata[1]}<br>count: %{customdata[2]}<extra></extra>",
            )
        )

    if not answers_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=answers_df["x"],
                y=answers_df["y"],
                z=answers_df["count"],
                mode="markers+text",
                name="answers",
                text=answers_df["response"],
                textposition="middle center",
                marker=dict(
                    size=plot_config["marker_size"] * 1.5,
                    color=colors["label_colors"]["1"],
                    opacity=plot_config["marker_opacity"]["normal"],
                ),
                textfont=dict(size=12, color="white"),
                hovertemplate="<b>Answer: %{text}</b><br>count: %{z}<extra></extra>",
            )
        )

    if title is None:
        title = f"Embedding Visualization (Count) - {', '.join(answers)}"

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Count"),
        width=plot_config["width"],
        height=plot_config["height"],
        showlegend=True,
    )

    return fig


def _labeled_scatter(
    df: pd.DataFrame,
    answers: List[str],
    title: Union[str, None] = None,
    theme: str = "default",
) -> go.Figure:
    """
    Scatterplot with different colors by label
    Required columns: ["response", "token", "x", "y", "label"]
    """
    color_schemes = util.load_color_schemes()
    colors = color_schemes[theme]
    plot_config = util.load_plot_config()

    df_viz = df.copy()

    first_answer = answers[0]
    origin_point = (
        df_viz[df_viz["response"] == first_answer].iloc[0]
        if not df_viz[df_viz["response"] == first_answer].empty
        else None
    )

    if origin_point is not None:
        df_viz["x"] = df_viz["x"] - origin_point["x"]
        df_viz["y"] = df_viz["y"] - origin_point["y"]

    fig = go.Figure()

    unique_labels = df_viz["label"].unique()
    df_viz["response"].isin(answers)

    available_colors = list(colors["label_colors"].keys())

    for i, label in enumerate(unique_labels):
        label_data = df_viz[df_viz["label"] == label]

        label_answers = label_data[label_data["response"].isin(answers)]
        label_non_answers = label_data[~label_data["response"].isin(answers)]

        color_key = available_colors[i % len(available_colors)]
        color = colors["label_colors"][color_key]

        if not label_non_answers.empty:
            fig.add_trace(
                go.Scatter(
                    x=label_non_answers["x"],
                    y=label_non_answers["y"],
                    mode="markers",
                    name=f"label {label}",
                    marker=dict(
                        size=plot_config["marker_size"],
                        color=color,
                        opacity=plot_config["marker_opacity"]["faded"],
                    ),
                    customdata=np.stack(
                        [
                            label_non_answers["response"].values,
                            label_non_answers["token"].values,
                            label_non_answers["label"].values,
                        ],
                        axis=-1,
                    ),
                    hovertemplate="<b>response: %{customdata[0]}</b><br>token: %{customdata[1]}<br>label: %{customdata[2]}<extra></extra>",
                )
            )

        if not label_answers.empty:
            fig.add_trace(
                go.Scatter(
                    x=label_answers["x"],
                    y=label_answers["y"],
                    mode="markers+text",
                    name=f"Answer label {label}",
                    text=label_answers["response"],
                    textposition="middle center",
                    marker=dict(
                        size=plot_config["marker_size"] * 1.5,
                        color=color,
                        opacity=plot_config["marker_opacity"]["normal"],
                    ),
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>Answer: %{text}</b><br>label: %{customdata[2]}<extra></extra>",
                    customdata=np.stack(
                        [
                            label_answers["response"].values,
                            label_answers["token"].values,
                            label_answers["label"].values,
                        ],
                        axis=-1,
                    ),
                )
            )

    if title is None:
        title = f"Embedding Visualization (Labeled) - {', '.join(answers)}"

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        template=plot_config["template"],
        width=plot_config["width"],
        height=plot_config["height"],
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True,
    )

    return fig


def _combined_scatter(
    df: pd.DataFrame,
    answers: List[str],
    title: Union[str, None] = None,
    theme: str = "default",
) -> go.Figure:
    """
    Combined visualization of valuecount_scatter and labeled_scatter
    Required columns: ["response", "token", "x", "y", "count", "label"]
    """
    color_schemes = util.load_color_schemes()
    colors = color_schemes[theme]
    plot_config = util.load_plot_config()

    df_viz = df.copy()

    first_answer = answers[0]
    origin_point = (
        df_viz[df_viz["response"] == first_answer].iloc[0]
        if not df_viz[df_viz["response"] == first_answer].empty
        else None
    )

    if origin_point is not None:
        df_viz["x"] = df_viz["x"] - origin_point["x"]
        df_viz["y"] = df_viz["y"] - origin_point["y"]

    fig = go.Figure()

    unique_labels = df_viz["label"].unique()
    available_colors = list(colors["label_colors"].keys())

    for i, label in enumerate(unique_labels):
        label_data = df_viz[df_viz["label"] == label]

        label_answers = label_data[label_data["response"].isin(answers)]
        label_non_answers = label_data[~label_data["response"].isin(answers)]

        color_key = available_colors[i % len(available_colors)]
        color = colors["label_colors"][color_key]

        if not label_non_answers.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=label_non_answers["x"],
                    y=label_non_answers["y"],
                    z=label_non_answers["count"],
                    mode="markers",
                    name=f"label {label}",
                    marker=dict(
                        size=plot_config["marker_size"],
                        color=color,
                        opacity=plot_config["marker_opacity"]["faded"],
                    ),
                    customdata=np.stack(
                        [
                            label_non_answers["response"].values,
                            label_non_answers["token"].values,
                            label_non_answers["label"].values,
                            label_non_answers["count"].values,
                        ],
                        axis=-1,
                    ),
                    hovertemplate="<b>response: %{customdata[0]}</b><br>token: %{customdata[1]}<br>label: %{customdata[2]}<br>count: %{customdata[3]}<extra></extra>",
                )
            )

        if not label_answers.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=label_answers["x"],
                    y=label_answers["y"],
                    z=label_answers["count"],
                    mode="markers+text",
                    name=f"Answer label {label}",
                    text=label_answers["response"],
                    textposition="middle center",
                    marker=dict(
                        size=plot_config["marker_size"] * 1.5,
                        color=color,
                        opacity=plot_config["marker_opacity"]["normal"],
                    ),
                    textfont=dict(size=12, color="white"),
                    hovertemplate="<b>Answer: %{text}</b><br>label: %{customdata[2]}<br>count: %{customdata[3]}<extra></extra>",
                    customdata=np.stack(
                        [
                            label_answers["response"].values,
                            label_answers["token"].values,
                            label_answers["label"].values,
                            label_answers["count"].values,
                        ],
                        axis=-1,
                    ),
                )
            )

    if title is None:
        title = f"Embedding Visualization (Combined) - {', '.join(answers)}"

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Count"),
        width=plot_config["width"],
        height=plot_config["height"],
        showlegend=True,
    )

    return fig
