# notes
"""
This file is for creating a property nav that will sit under main navigation bar.
Dash Bootstrap Components documentation linked below:
https://dash-bootstrap-components.opensource.faculty.ai/docs/components/nav/
"""

from dash.dependencies import Input, Output
from dash import callback
import dash_bootstrap_components as dbc

PROPERTY_NAV_STYLE = {
    "position": "fixed",
    # "top": "70px",
    "left": "16rem",
    # "bottom": 0,
    # "width": "16rem",
    # "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
}

PROPERTY_NAV_HIDEN = {
    "position": "fixed",
    # "top": "70px",
    "left": 0,
    # "bottom": 0,
    # "width": "16rem",
    # "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
}
property_nav = dbc.Nav(
    [
        dbc.NavItem(dbc.Label("Status")),
        dbc.NavItem(
            dbc.Spinner(color="danger", type="grow", size="sm"),
            style={"margin-left": "0.5rem"},
        ),
        dbc.NavItem(
            dbc.Label("Heatmap updating"),
            style={"margin-left": "2rem"},
        ),
        dbc.NavItem(
            dbc.Checklist(
                options=[
                    {"label": "Pause", "value": 1},
                ],
                value=[],
                id="switches-inline-input",
                inline=True,
                switch=True,
            ),
            style={"margin-left": "0.5rem"},
        ),
    ],
    id="property-nav",
    style=PROPERTY_NAV_STYLE,
)


@callback(
    Output("property-nav", "style"),
    Input("btn_sidebar", "n_clicks"),
    Input("side_click", "data"),
)
def toggle_nav(n, nclick):
    """function to hide and reveal sidebar

    Args:
        n (int): _description_
        nclick (int): _description_

    Returns:
        dict: style objects
    """
    if n:
        if nclick == "SHOW":
            nav_style = PROPERTY_NAV_STYLE
        else:
            nav_style = PROPERTY_NAV_HIDEN
    else:
        nav_style = PROPERTY_NAV_STYLE

    return nav_style
