"""
This file is for creating a graph layout
"""

from dash.dependencies import Input, Output
from dash import dcc, callback
import plotly.graph_objects as go
from pyquac.settings import settings

AXIS_SIZE = 13

GRAPH_STYLE = {
    "position": "fixed",
    "top": "110px",
    "left": "16rem",
    "bottom": 0,
    "width": "45rem",
    "height": "37rem",
    # "z-index": 1,
    "overflow-x": "hidden",
    "transition": settings.transition_time,
    # "transition-delay": "width 500ms",
    # "transition-property": "margin-right",
    # "padding": "0.5rem 1rem",
}

GRAPH_HIDEN = {
    "position": "fixed",
    "top": "110px",
    "left": 0,
    "bottom": 0,
    "right": 0,
    "width": "45rem",
    # "width": "70rem",
    "height": "37rem",
    # "z-index": 1,
    # "overflow-x": "hidden",
    "transition": settings.transition_time,
    # "transition-delay": "500ms",
    # "transition-property": "margin-right",
    # "padding": "0.5rem 1rem",
}


def define_figure(
    z,
    x,
    y,
    x_axis_title: str,
    y_axis_title: str,
    cmap: str,
):
    """sets figure layout

    Args:
        data (Spectroscopy): spectroscopy-like object

    Returns:
        go.gigure: plotly figure
    """
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=cmap))

    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        autosize=False,
        separators=".",
    )

    fig.update_yaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_xaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_layout(yaxis=dict(showexponent="none", exponentformat="e"))
    fig.update_traces(zhoverformat=".2f")
    fig.update_layout(width=650, height=550)
    return fig


def define_figure_extend(
    z,
    x,
    y,
    x_axis_title: str,
    y_axis_title: str,
    cmap: str,
):
    """sets figure layout

    Args:
        data (Spectroscopy): spectroscopy-like object

    Returns:
        go.gigure: plotly figure
    """
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=cmap))

    fig.add_trace(go.Scatter(x=None, y=None, mode="lines", xaxis="x2"))

    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        autosize=False,
        separators=".",
    )

    fig.update_layout(
        autosize=False,
        xaxis=dict(zeroline=False, domain=[0, 0.60], showgrid=False),
        # yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
        xaxis2=dict(zeroline=False, domain=[0.60, 1], showgrid=False),
        # yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
        bargap=0,
        hovermode="closest",
    )

    fig.update_yaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_xaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_layout(yaxis=dict(showexponent="none", exponentformat="e"))
    # fig.update_traces(zhoverformat=".2f")
    fig.update_layout(width=850, height=550)
    return fig


def figure_layout(
    data,
    x_axis_title: str,
    y_axis_title: str,
    cmap: str,
):
    """constructor for heatmap layout

    Args:
        data (pandas DataFrame): DataFrame with the columns in the following order: [x, y, z]
    """
    z = data.njit_result
    x = data.x_1d
    y = data.y_1d
    figure = dcc.Graph(
        id="heatmap",
        figure=define_figure(
            z=z,
            x=x,
            y=y,
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
            cmap=cmap,
        ),
        style=GRAPH_STYLE,
    )
    return figure


@callback(
    Output("heatmap", "style"),
    Output("heatmap", "figure"),
    Input("btn_sidebar", "n_clicks"),
    Input("side_click", "data"),
    Input("z_store", "data"),
    Input("x_store", "data"),
    Input("y_store", "data"),
    Input("x_label", "data"),
    Input("y_label", "data"),
    Input("cmap", "data"),
)
def toggle_graph(n, nclick, z, x, y, x_label, y_label, cmap):
    """function to hide and reveal sidebar

    Args:
        n (int): _description_
        nclick (int): _description_

    Returns:
        dict: style objects
    """
    fig = define_figure(
        z,
        x,
        y,
        x_label,
        y_label,
        cmap,
    )
    if n:
        if nclick == "SHOW":
            graph_style = GRAPH_STYLE
            fig = define_figure(
                z,
                x,
                y,
                x_label,
                y_label,
                cmap,
            )
        else:
            graph_style = GRAPH_HIDEN
            fig = define_figure_extend(
                z,
                x,
                y,
                x_label,
                y_label,
                cmap,
            )
    else:
        graph_style = GRAPH_STYLE

    return graph_style, fig


# @callback(
#     Output("heatmap", "figure"),
#     Input("interval-graph-update", "n_intervals"),
#     Input("z_store", "data"),
#     State("heatmap", "figure"),
# )
# def update_graph(i, z, fig):

#     if i == 0:
#         raise PreventUpdate
#     return go.Figure(fig).update_traces(z=z)
