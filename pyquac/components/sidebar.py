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
from pyquac.settings import settings

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
    "transition": settings.transition_time,
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
    "transition": settings.transition_time,
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

# items
items = [
    dbc.DropdownMenuItem("First"),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Second"),
]

########################## Sidebar ##########################
sidebar = html.Div(
    [
        html.H5("Spectroscopy"),
        html.P("control panel", className="summary"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.DropdownMenu(
                    label="Select data",
                    children=items,
                    direction="end",
                    className="mt-2",
                ),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavItem(html.P("Data saving"), class_name="mt-3"),
                dbc.NavItem(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("CSV file", title="save current"),
                            dbc.Button(class_name="fa fa-server", title="save all"),
                        ],
                        size="md",
                        # className="my-3",
                    )
                ),
                dbc.NavItem(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("raw CSV file", title="save current"),
                            dbc.Button(class_name="fa fa-server", title="save all"),
                        ],
                        size="md",
                        className="my-3",
                    )
                ),
            ],
            vertical=True,
            pills=True,
        ),
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("PDF file", title="save current"),
                            dbc.Button(class_name="fa fa-server", title="save all"),
                        ],
                        size="md",
                        # className="my-3",
                    )
                ),
                dbc.NavItem(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("SVG file", title="save current"),
                            dbc.Button(class_name="fa fa-server", title="save all"),
                        ],
                        size="md",
                        className="my-3",
                    )
                ),
                dbc.NavItem(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("HTML file", title="save current"),
                            dbc.Button(class_name="fa fa-server", title="save all"),
                        ],
                        size="md",
                        # className="my-3",
                    )
                ),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Button(
            class_name="fa fa-gear",
            size="lg",
            outline=True,
            color="#f8f9fa",
            title="project settings",
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
