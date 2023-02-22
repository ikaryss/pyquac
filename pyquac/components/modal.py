from dash.dependencies import Input, Output, State
from dash import html, dcc
from dash.exceptions import PreventUpdate
from dash import callback
import dash_bootstrap_components as dbc
import dash_daq as daq
from pyquac.settings import settings

tooltip = html.Div(
    [
        html.P(
            [
                "for colormap you can use ",
                html.Span(
                    "pre-defined or specified",
                    id="tooltip-target",
                    style={"textDecoration": "underline", "cursor": "pointer"},
                ),
                " values",
            ]
        ),
        dbc.Tooltip(
            """
            - A list of 2-element lists where the first element is the normalized color level value (starting at 0 and ending at 1), and the second item is a valid color string. (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
            - One of the following named colorscales: 'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance' ...
            """,
            target="tooltip-target",
            autohide=False,
        ),
    ]
)
path_input = html.Div(
    [
        dbc.Label("Save path"),
        dbc.Input(
            id="default-path",
            placeholder="Enter root saving path",
            value=settings.default_root_path,
            type="text",
        ),
        dbc.FormText(
            "Enter new root path to save data. Example: D:/Scripts/qubit",
        ),
    ],
    className="mb-3",
)

xaxis_input = html.Div(
    [
        dbc.Label("X-axis title"),
        dbc.Input(
            id="x-title",
            placeholder="Enter new x-axis title",
            value=settings.init_x_label,
            type="text",
        ),
        dbc.Label("Y-axis title", class_name="mt-2"),
        dbc.Input(
            id="y-title",
            placeholder="Enter new y-axis title",
            value=settings.init_y_label,
            type="text",
        ),
        dbc.Label("heatmap color map", class_name="mt-2"),
        dbc.Input(
            id="cmap-title",
            placeholder="Enter new cmap",
            value=settings.init_cmap,
            type="text",
            disabled=True,
        ),
        tooltip,
    ],
    className="mb-3",
)

modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Settings")),
        dbc.ModalBody([path_input, xaxis_input]),
        dbc.ModalFooter(
            dbc.Button(
                "Save and close", id="modal_close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="modal",
    is_open=False,
    keyboard=False,
    backdrop="static",
)


@callback(
    Output("modal", "is_open"),
    [Input("open_modal", "n_clicks"), Input("modal_close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
