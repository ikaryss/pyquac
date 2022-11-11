# notes
"""
This file is for creating a side bar that will sit on the left of your application.
Dash Bootstrap Components documentation on creating simple side bar linked below:
https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
"""
from dash.dependencies import Input, Output, State
from dash import html
from dash import callback
import dash_bootstrap_components as dbc

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "70px",
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    # "transition-delay": "500ms",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": "70px",
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

########################## Sidebar ##########################
sidebar = html.Div(
    [
        html.H5("Spectroscopy"),
        html.Hr(),
        html.P("A simple sidebar for data settings"),
        dbc.Nav(
            [
                dbc.NavLink("btn 1", href="/", active="exact"),
                dbc.NavLink("btn 2", href="/page-1", active="exact"),
                dbc.NavLink("btn 3", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)


@callback(
    Output("sidebar", "style"),
    Output("page-content", "style"),
    Output("side_click", "data"),
    Input("btn_sidebar", "n_clicks"),
    State("side_click", "data"),
)
def toggle_sidebar(n, nclick):
    """function to hide and reveal sidebar

    Args:
        n (int): _description_
        nclick (int): _description_

    Returns:
        dict: style objects
    """
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = "SHOW"

    return sidebar_style, content_style, cur_nclick
