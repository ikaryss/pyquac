"""
This file is for creating a graph layout
"""

from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dash import dcc, callback


# def figure_layout(data, cmap):
#     def define_figure(data, cmap):
#         fig = go.Figure(
#             data=go.Heatmap(
#                 z=data.iloc[:, 2], x=data.iloc[:, 0], y=data.iloc[:, 1], colorscale=cmap
#             )
#         )
#         return fig

#     figure = dcc.Graph(id="heatmap", figure=define_figure(data, cmap))
#     return figure

GRAPH_STYLE = {
    "position": "fixed",
    "top": "100px",
    "left": "16rem",
    "bottom": 0,
    "width": "45rem",
    "height": "37rem",
    # "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    # "transition-delay": "width 500ms",
    # "transition-property": "margin-right",
    # "padding": "0.5rem 1rem",
}

GRAPH_HIDEN = {
    "position": "fixed",
    "top": "100px",
    "left": 0,
    "bottom": 0,
    "width": "70rem",
    "height": "37rem",
    # "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "transition-delay": "500ms",
    # "transition-property": "margin-right",
    # "padding": "0.5rem 1rem",
}


def figure_layout(data):
    """constructor for heatmap layout

    Args:
        data (pandas DataFrame): DataFrame with the columns in the following order: [x, y, z]
    """

    def define_figure(data):
        fig = go.Figure(
            data=go.Heatmap(z=data.iloc[:, 2], x=data.iloc[:, 0], y=data.iloc[:, 1])
        )
        fig.update_layout()
        return fig

    figure = dcc.Graph(id="heatmap", figure=define_figure(data), style=GRAPH_STYLE)
    return figure


@callback(
    Output("heatmap", "style"),
    Input("btn_sidebar", "n_clicks"),
    Input("side_click", "data"),
)
def toggle_graph(n, nclick):
    """function to hide and reveal sidebar

    Args:
        n (int): _description_
        nclick (int): _description_

    Returns:
        dict: style objects
    """
    if n:
        if nclick == "SHOW":
            graph_style = GRAPH_STYLE
        else:
            graph_style = GRAPH_HIDEN
    else:
        graph_style = GRAPH_STYLE

    return graph_style
