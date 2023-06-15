# notes
"""
This file is for creating a navigation bar that will sit at the top of your application.
Dash Bootstrap Components documentation linked below:
https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/
"""

from dash.dependencies import Input, Output
from dash import html
import dash_bootstrap_components as dbc

from ..settings import settings

APP_LOGO = settings.app_logo
LOGO_HEIGHT = "42px" if APP_LOGO == settings.app_logo_pyquac else "35px"

########################## Navbar ##########################
# Input
COLLAPSE = html.Div(
    [
        dbc.Button(
            [html.I(className="fa fa-align-justify")],
            id="btn_sidebar",
            className="offset-2",
            n_clicks=0,
        ),
    ]
)

LOGO = html.A(
    dbc.Row(
        [
            dbc.Col(html.Img(src=APP_LOGO, height=LOGO_HEIGHT)),
            dbc.Col(dbc.NavbarBrand(settings.app_name, className="ms-2")),
        ],
        align="start",
        className="g-0",
    ),
    href=settings.app_link,
    target="_blank",
    style={"textDecoration": "none", "margin-left": "1rem"},
)

ABOUT = dbc.NavItem(
    html.Div(
        [
            dbc.NavLink("About", href="/", id="about-popover", active=True),
            dbc.Popover(
                id="about",
                is_open=False,
                target="about-popover",
                children=[
                    dbc.PopoverHeader("How it Works"),
                    dbc.PopoverBody("Some text"),
                ],
            ),
        ]
    )
)

# git_link = dcc.Clipboard(title="copy link", content="https://github.com/ikaryss/pyquac")
DROPDOWN = dbc.DropdownMenu(
    label="Links",
    in_navbar=True,
    children=[
        dbc.DropdownMenuItem(
            [html.I(className="fa fa-github"), " Code"],
            href=settings.app_github_url,
            target="_blank",
        ),
    ],
    color="primary",
)

navbar = dbc.Navbar(
    class_name="nav nav-pills",
    children=[COLLAPSE, LOGO, ABOUT, DROPDOWN],
    color="primary",
    dark=True,
    expand="sm",
)

# Callback
# @app.callback(
#     Output(component_id="about", component_property="is_open"),
#     Output(component_id="about-popover", component_property="active"),
#     Input(component_id="about-popover", component_property="n_clicks"),
#     Input("about", "is_open"),
#     Input("about-popover", "active"),
# )
# def about_popover(n, is_open, active):
#     """toggle about popover

#     Args:
#         n (int): number of clicks
#         is_open (bool): current state
#         active (bool): _description_

#     Returns:
#         tuple(bool): status flip
#     """
#     if n:
#         return not is_open, active
#     return is_open, active
