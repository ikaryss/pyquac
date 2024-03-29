"""
This file is for creating a graph layout
"""

from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from dash import dcc
import plotly.graph_objects as go
import numpy as np
from pyquac.settings import settings

AXIS_SIZE = 13

GRAPH_STYLE = {
    "position": "fixed",
    # "top": "90px",
    "left": "16rem",
    # "bottom": 0,
    # "width": "45rem",
    # "height": 500,
    # "z-index": 1,
    # "overflow-x": "hidden",
    "transition": settings.transition_time,
    # "transition-delay": "width 500ms",
    # "transition-property": "margin-right",
    # "padding": "0.5rem 1rem",
}

GRAPH_HIDEN = {
    "position": "fixed",
    # "top": "90px",
    "left": 0,
    # "bottom": 0,
    # "right": 0,
    "width": "45rem",
    # "width": "70rem",
    "height": "50rem",
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

    fig.add_trace(go.Scatter(x=None, y=None, mode="lines", xaxis="x2", name="vertical"))

    fig.add_trace(
        go.Scatter(x=None, y=None, mode="lines", yaxis="y2", name="horizontal")
    )

    fig.add_trace(
        go.Scatter(
            x=None,
            y=None,
            mode="lines",
            # yaxis="y2",
            name="fit",
            line_color="#000000",
            line_width=3,
            visible=False,
        )
    )

    # fig.update_layout(
    #     yaxis={"range": [min(y), max(y)], "autorange"}, xaxis={"range": [min(x), max(x)]}
    # )
    # fig.update_layout(
    #     yaxis={"range": [min(y), max(y)], "autorange": False}, xaxis={"range": [min(x), max(x)], "autorange": False}
    # )
    # fig.update_yaxes(range=[min(y), max(y)])
    # print(fig["data"])

    fig.add_vline(
        x=None,
        visible=False,
        line=dict(
            color="Black",
            width=2,
            dash="dash",
        ),
        opacity=1.0,
        name="vline",
    )

    fig.add_hline(
        y=None,
        visible=False,
        line=dict(
            color="Black",
            width=2,
            dash="dash",
        ),
        opacity=1.0,
        name="hline",
    )

    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        autosize=False,
        separators=".",
    )

    fig.update_layout(
        yaxis2=dict(title="Mag, dB", side="left"),
        xaxis2=dict(title="Mag, dB", side="right"),
    )

    fig.update_layout(
        autosize=False,
        xaxis=dict(zeroline=False, domain=[0, 0.72], showgrid=True, tickangle=0),
        yaxis=dict(zeroline=False, domain=[0, 0.72], showgrid=True, tickangle=0),
        xaxis2=dict(
            zeroline=True,
            domain=[0.76, 1],
            showgrid=True,
            visible=True,
            tickangle=0,
        ),
        yaxis2=dict(zeroline=False, domain=[0.76, 1], showgrid=True, tickangle=0),
        bargap=1,
        hovermode="closest",
    )

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor="black", mirror=False
    )  ## Mirror True for box
    # fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(showlegend=False)

    fig.update_yaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_xaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_layout(yaxis=dict(showexponent="none", exponentformat="e"))
    # fig.update_traces(zhoverformat=".2f")
    fig.update_layout(height=600)
    # https://plotly.com/python/hover-text-and-formatting/
    fig.update_xaxes(
        showspikes=True,
        spikecolor="black",
        spikethickness=2,
        spikesnap="cursor",
        spikemode="across",
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="black",
        spikethickness=2,
        spikesnap="cursor",
        spikemode="across",
    )
    fig.update_traces(hoverinfo="x+y+z", selector=dict(type="heatmap"))
    return fig


def define_figure_simple(
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

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    fig.update_yaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_xaxes(title_font={"size": AXIS_SIZE}, tickfont_size=AXIS_SIZE)
    fig.update_layout(yaxis=dict(showexponent="none", exponentformat="e"))
    fig.update_traces(zhoverformat=".2f")
    fig.update_layout(height=700, width=700)
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
        animate=False,
        config={
            "scrollZoom": False,
        },
        style=GRAPH_STYLE,
    )
    return figure


# @app.callback(
#     Output("heatmap", "style"),
#     Input("btn_sidebar", "n_clicks"),
#     Input("side_click", "data"),
# )
# def toggle_graph(n, nclick):
#     """function to hide and reveal sidebar

#     Args:
#         n (int): _description_
#         nclick (int): _description_

#     Returns:
#         dict: style objects
#     """
#     if n:
#         if nclick == "SHOW":
#             graph_style = GRAPH_STYLE
#         else:
#             graph_style = GRAPH_HIDEN
#     else:
#         graph_style = GRAPH_STYLE

#     return graph_style


# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="refresh_graph"),
#     Output("heatmap", "figure"),
#     Input("interval-graph-update", "n_intervals"),
#     Input("x_click", "data"),
#     Input("y_click", "data"),
#     Input("heatmap", "clickData"),
#     Input("modal_close", "n_clicks"),
#     Input("modal_db_close", "n_clicks"),
#     State("x_store", "data"),
#     State("y_store", "data"),
#     State("z_store", "data"),
#     State("heatmap", "figure"),
#     State("y_scatter", "data"),
#     State("yz_scatter", "data"),
#     State("x_scatter", "data"),
#     State("xz_scatter", "data"),
#     State("line-switches", "on"),
#     State("x-title", "value"),
#     State("y-title", "value"),
#     prevent_initial_call=True,
# )


# @app.callback(
#     Output("x_scatter", "data"),
#     Output("y_scatter", "data"),
#     Output("xz_scatter", "data"),
#     Output("yz_scatter", "data"),
#     Output("x_click", "data"),
#     Output("y_click", "data"),
#     Input("heatmap", "clickData"),
#     State("x_raw", "data"),
#     State("y_raw", "data"),
#     State("z_raw", "data"),
# )
# def update_click_data(click, x_raw, y_raw, z_raw):
#     if (click is None) or (click["points"][0]["curveNumber"] != 0):
#         raise PreventUpdate

#     data_click = click["points"][0]
#     x_click = data_click["x"]
#     y_click = data_click["y"]

#     x_mask = np.equal(np.array(x_raw), np.array(x_click))
#     y_mask = np.equal(np.array(y_raw), np.array(y_click))
#     x_scatter = np.array(x_raw)[y_mask]
#     y_scatter = np.array(y_raw)[x_mask]

#     xz_scatter = np.array(z_raw)[y_mask]
#     yz_scatter = np.array(z_raw)[x_mask]

#     return (
#         x_scatter,
#         y_scatter,
#         xz_scatter,
#         yz_scatter,
#         x_click,
#         y_click,
#     )


# app.clientside_callback(
#     """
#     function(i) {
#         const triggered_id = dash_clientside.callback_context.triggered.map(t => t.prop_id)[0];
#         return triggered_id;
#     }
#     """,
#     Output("temp", "data"),
#     Input("modal_close", "n_clicks"),
#     prevent_initial_call=True,
# )
