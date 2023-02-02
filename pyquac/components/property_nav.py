# notes
"""
This file is for creating a property nav that will sit under main navigation bar.
Dash Bootstrap Components documentation linked below:
https://dash-bootstrap-components.opensource.faculty.ai/docs/components/nav/
"""

from dash.dependencies import Input, Output
from dash import callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
from pyquac.settings import settings

PROPERTY_NAV_STYLE = {
    "position": "fixed",
    # "top": "70px",
    "left": "16rem",
    # "bottom": 0,
    # "width": "16rem",
    # "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": settings.transition_time,
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
    "transition": settings.transition_time,
    "padding": "0.5rem 1rem",
}
property_nav = dbc.Nav(
    [
        dbc.NavItem(dbc.Label("Status")),
        # dbc.NavItem(
        #     dbc.Spinner(color="danger", type="grow", size="sm"),
        #     style={"margin-left": "0.5rem"},
        # ),
        dbc.NavItem(
            dbc.Label("Heatmap updating"),
            style={"margin-left": "2rem"},
        ),
        dbc.NavItem(
            [
                daq.BooleanSwitch(id="interval-switches", on=True, color="#3459e6"),
            ],
            style={"margin-left": "0.5rem"},
        ),
        dbc.NavItem(
            dbc.Input(
                id="update-interval-value",
                type="number",
                min=800,
                step=1,
                placeholder="Update graph in... ms",
                disabled=False,
                html_size=16,
                size="sm",
            ),
            style={"margin-left": "1rem"},
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


@callback(
    Output("interval-graph-update", "max_intervals"), Input("interval-switches", "on")
)
def toggle_checklist(switch_state):
    """function to change max interval property

    Args:
        n (_type_): _description_
        max_interval (_type_): _description_

    Returns:
        _type_: _description_
    """

    if switch_state is True:
        new_max_interval = -1
        print("I will update")
    else:
        new_max_interval = 0
        print("I stop updating")
    return new_max_interval


@callback(
    Output("interval-graph-update", "interval"), Input("update-interval-value", "value")
)
def change_interval_update(new_interval):

    if new_interval is not None:
        print(f"HEY! now I update in {new_interval} ms")
        return new_interval
    else:
        raise PreventUpdate
