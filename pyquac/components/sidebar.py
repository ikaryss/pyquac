# notes
"""
This file is for creating a side bar that will sit on the left of your application.
Dash Bootstrap Components documentation on creating simple side bar linked below:
https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
"""
from dash.dependencies import Input, Output, State
from dash import html
from dash.exceptions import PreventUpdate
from dash import callback
import dash_bootstrap_components as dbc
import dash_daq as daq
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
    "transition": settings.margin_transition_time,
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE1 = {
    "transition": settings.margin_transition_time,
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
def sidebar(data):
    return html.Div(
        [
            html.H5("Spectroscopy"),
            html.P("control panel", className="summary"),
            # html.Hr(),
            # dbc.Nav(
            #     [
            #         dbc.DropdownMenu(
            #             label="Select data",
            #             children=items,
            #             direction="end",
            #             className="mt-2",
            #         ),
            #     ],
            #     vertical=True,
            #     pills=True,
            # ),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.Div("heatmap updating"), width=8),
                                    dbc.Col(
                                        daq.BooleanSwitch(
                                            id="interval-switches",
                                            on=True,
                                            color="#3459e6",
                                        ),
                                        width=4,
                                    ),
                                ]
                            ),
                        ],
                        # style={"margin-left": "2rem"},
                    ),
                    dbc.NavItem(
                        dbc.Input(
                            id="update-interval-value",
                            type="number",
                            min=500,
                            step=1,
                            placeholder="Update graph in... ms",
                            disabled=False,
                            html_size=16,
                            size="sm",
                        ),
                        # style={"margin-left": "1rem"},
                    ),
                    html.Hr(),
                    dbc.NavItem(
                        dbc.Row(
                            [
                                dbc.Col(html.Div("data saving", id="log")),
                            ]
                        )
                    ),
                    dbc.NavItem(
                        dbc.ButtonGroup(
                            [
                                dbc.Button("csv", id="csv", title="save current CSV"),
                                dbc.Button(
                                    "stack csv",
                                    id="raw csv",
                                    title="save current stacked CSV",
                                ),
                                dbc.Button("pdf", id="pdf", title="save current PDF"),
                                dbc.Button("svg", id="svg", title="save current SVG"),
                                dbc.Button(
                                    "html", id="html", title="save current HTML"
                                ),
                                # dbc.Button(class_name="fa fa-server", title="save all"),
                            ],
                            size="sm",
                            # className="my-3",
                            className="mt-2",
                        )
                    ),
                    dbc.NavItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.Div("show XY lines"), width=8),
                                    dbc.Col(
                                        daq.BooleanSwitch(
                                            id="line-switches",
                                            on=settings.init_xy_lines_state,
                                            color="#3459e6",
                                        ),
                                        width=4,
                                    ),
                                ]
                            ),
                        ],
                        class_name="mt-3",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            dbc.Button(
                id="open_modal",
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


@callback(
    Output("interval-graph-update", "max_intervals"),
    Output("line-switches", "disabled"),
    Input("interval-switches", "on"),
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
        disabled = False
    else:
        new_max_interval = 0
        disabled = True
    return new_max_interval, disabled


@callback(
    Output("interval-graph-update", "interval"), Input("update-interval-value", "value")
)
def change_interval_update(new_interval):

    if new_interval is not None:
        return new_interval
    else:
        raise PreventUpdate


@callback(
    Output("xy_lines_state", "data"),
    Input("line-switches", "on"),
)
def toggle_xy_lines(switch_state):

    if switch_state is True:
        new_xy_state = True
    else:
        new_xy_state = False
    return new_xy_state
