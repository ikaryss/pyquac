# notes
"""
This file is for housing the main dash application.
This is where we define the various css items to fetch as well as the layout of our application.
"""

from dash import dcc, html, callback
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from jupyter_dash import JupyterDash

import numpy as np

from pyquac.settings import settings
from pyquac.utils import initial_spectroscopy_in_app
from pyquac.components.navbar import navbar
from pyquac.components.sidebar import sidebar, content
from pyquac.components.modal import modal
from pyquac.components.modal_db import modal_db

from pyquac.components.property_nav import property_nav
from pyquac.components.heatmap import figure_layout

# App Instance
THEME = dbc.themes.ZEPHYR
CSS = dbc.icons.FONT_AWESOME
ICONS = dbc.icons.BOOTSTRAP
load_figure_template("ZEPHYR")

STORE = {
    "temp": None,
    "side_click": "SHOW",
    "xy_lines_state": settings.init_xy_lines_state,
    "x_raw": None,
    "y_raw": None,
    "z_raw": None,
    "x_scatter": None,
    "y_scatter": None,
    "xz_scatter": None,
    "yz_scatter": None,
    "x_click": None,
    "y_click": None,
    "x_label": settings.init_x_label,
    "y_label": settings.init_y_label,
}

app = JupyterDash(__name__, external_stylesheets=[THEME, CSS, ICONS])
app.title = settings.app_name

########################## App Layout ##########################


def conf_app(spectroscopy, cmap: str = settings.init_cmap):

    # Data configuration block
    data_zoo, chip_zoo, qubit_zoo, type_zoo = initial_spectroscopy_in_app(spectroscopy)

    if not isinstance(data_zoo, list):
        data = data_zoo
        chip = chip_zoo
        qubit_toggle = qubit_zoo
        spectroscopy_type = type_zoo
        db_disabled = True
    else:
        data = data_zoo[0]
        chip = chip_zoo[0]
        qubit_toggle = qubit_zoo[0]
        spectroscopy_type = type_zoo[0]
        db_disabled = False

    # Layout block
    def serve_layout():
        """Define the layout of the application

        Returns:
            DIV component: app layout
        """
        return html.Div(
            children=[
                *[
                    dcc.Store(id=store[0], data=store[1], storage_type="session")
                    for store in STORE.items()
                ],
                dcc.Store(id="z_store", data=data.njit_result, storage_type="session"),
                dcc.Store(id="x_store", data=data.x_1d, storage_type="session"),
                dcc.Store(id="y_store", data=data.y_1d, storage_type="session"),
                dcc.Store(
                    id="save_attributes",
                    data=dict(
                        qubit_toggle=qubit_toggle,
                        chip=chip,
                        spectroscopy_type=spectroscopy_type,
                    ),
                    storage_type="session",
                ),
                dcc.Store(id="cmap", data=cmap, storage_type="session"),
                dcc.Store(
                    id="database",
                    data=dict(
                        db_disabled=db_disabled,
                        chip=chip_zoo,
                        qubit_toggle=qubit_zoo,
                        type=type_zoo,
                    ),
                    # storage_type="session",
                ),
                dcc.Store(id="current_data", data=None),
                dcc.Interval(
                    id="interval-graph-update",
                    interval=settings.init_interval,
                    n_intervals=0,
                    max_intervals=settings.init_max_interval,
                ),
                # dcc.Interval(id="clientside-interval", n_intervals=0, interval=250),
                dbc.Row(
                    [
                        dbc.Col(navbar),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            children=[sidebar, content, modal, modal_db],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    children=[
                                        property_nav,
                                    ]
                                ),
                                dbc.Row(
                                    children=[
                                        figure_layout(
                                            data,
                                            settings.init_x_label,
                                            settings.init_y_label,
                                            settings.init_cmap,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    @callback(
        Output("x_store", "data"),
        Output("y_store", "data"),
        Output("z_store", "data"),
        Output("x_raw", "data"),
        Output("y_raw", "data"),
        Output("z_raw", "data"),
        Input("interval-graph-update", "n_intervals"),
        # Input("tbl", "active_cell")
        Input("current_data", "data"),
    )
    def update_fig_data(i, active_cell):
        if active_cell is None:
            return (
                data.x_1d,
                data.y_1d,
                data.njit_result,
                data.x_raw,
                data.y_raw,
                data.z_raw,
            )
        current_data = data_zoo[active_cell]
        return (
            current_data.x_1d,
            current_data.y_1d,
            current_data.njit_result,
            current_data.x_raw,
            current_data.y_raw,
            current_data.z_raw,
        )

    @callback(
        Output("x_scatter", "data"),
        Output("y_scatter", "data"),
        Output("xz_scatter", "data"),
        Output("yz_scatter", "data"),
        Output("x_click", "data"),
        Output("y_click", "data"),
        Input("heatmap", "clickData"),
        State("x_raw", "data"),
        State("y_raw", "data"),
        State("z_raw", "data"),
    )
    def update_click_data(click, x_raw, y_raw, z_raw):
        if (click is None) or (click["points"][0]["curveNumber"] != 0):
            raise PreventUpdate

        data_click = click["points"][0]
        x_click = data_click["x"]
        y_click = data_click["y"]

        x_mask = np.equal(np.array(x_raw), np.array(x_click))
        y_mask = np.equal(np.array(y_raw), np.array(y_click))
        x_scatter = np.array(x_raw)[y_mask]
        y_scatter = np.array(y_raw)[x_mask]

        xz_scatter = np.array(z_raw)[y_mask]
        yz_scatter = np.array(z_raw)[x_mask]

        return (
            x_scatter,
            y_scatter,
            xz_scatter,
            yz_scatter,
            x_click,
            y_click,
        )

    app.layout = serve_layout
    return app


########################## Run ##########################
# if __name__ == "__main__":
#     app.run_server(debug=True, port=6050, mode="inline")
