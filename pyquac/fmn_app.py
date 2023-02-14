# notes
"""
This file is for housing the main dash application.
This is where we define the various css items to fetch as well as the layout of our application.
"""

from dash import dcc, callback, ctx
from dash.dependencies import Input, Output
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from jupyter_dash import JupyterDash

from pyquac.settings import settings
from pyquac.components.navbar import navbar
from pyquac.components.sidebar import sidebar, content
from pyquac.components.property_nav import property_nav
from pyquac.components.heatmap import figure_layout

import numpy as np

# App Instance
THEME = dbc.themes.ZEPHYR
CSS = settings.css_url
ICONS = dbc.icons.BOOTSTRAP
load_figure_template("ZEPHYR")


app = JupyterDash(__name__, external_stylesheets=[THEME, CSS, ICONS])
app.title = settings.app_name


########################## App Layout ##########################


def conf_app(data):
    def serve_layout():
        """Define the layout of the application

        Returns:
            DIV component: app layout
        """
        return html.Div(
            children=[
                dcc.Store(id="side_click", data="SHOW"),
                dcc.Store(id="z_store", data=data.njit_result),
                dcc.Store(id="x_store", data=data.x_1d),
                dcc.Store(id="y_store", data=data.y_1d),
                dcc.Store(id="y_scatter", data=None),
                dcc.Store(id="z_scatter", data=None),
                # dcc.Store(id="update_interval", data=settings.init_interval),
                # dcc.Store(id="max_interval_value", data=settings.init_max_interval),
                dcc.Store(id="x_label", data=settings.init_x_label),
                dcc.Store(id="y_label", data=settings.init_y_label),
                dcc.Store(id="cmap", data=settings.init_cmap),
                dcc.Interval(
                    id="interval-graph-update",
                    interval=settings.init_interval,
                    n_intervals=0,
                    max_intervals=settings.init_max_interval,
                ),
                dcc.Interval(id="clientside-interval", n_intervals=0, interval=250),
                dbc.Row(
                    [
                        dbc.Col(navbar),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(children=[sidebar(data), content], width=3),
                        dbc.Col(
                            [
                                # dbc.Row(
                                #     children=[
                                #         property_nav,
                                #     ]
                                # ),
                                dbc.Row(
                                    children=[
                                        figure_layout(
                                            data,
                                            settings.init_x_label,
                                            settings.init_y_label,
                                            settings.init_cmap,
                                        ),
                                        # dcc.Graph(
                                        #     id="empty",
                                        #     figure=go.Scatter(
                                        #         x=[1, 2, 3], y=[1, 2, 3], mode="lines"
                                        #     ),
                                        #     style={"display": "none"},
                                        # ),
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
        Input("interval-graph-update", "n_intervals"),
    )
    def update_fig_data(i):
        return (
            data.x_1d,
            data.y_1d,
            data.njit_result,
        )

    @callback(
        Output("y_scatter", "data"),
        Output("z_scatter", "data"),
        Input("heatmap", "clickData"),
        Input("x-slider", "value"),
    )
    def update_click_data(click, x_slider):
        triggered_id = ctx.triggered_id
        if (click == None) and (x_slider == None):
            y_scatter, z_scatter = None, None

        if triggered_id == "heatmap":
            if click is None:
                y_scatter, z_scatter = None, None
            else:
                x_click = click["points"][0]["x"]
                print(x_click)
                mask = np.equal(data.x_raw, np.array(x_click))
                y_scatter = np.array(data.y_raw)[mask]
                z_scatter = np.array(data.z_raw)[mask]

        if triggered_id == "x-slider":
            print("check")
            mask = np.equal(data.x_raw, np.array(x_slider))
            y_scatter = np.array(data.y_raw)[mask]
            z_scatter = np.array(data.z_raw)[mask]
        return y_scatter, z_scatter

    # @callback(
    #     Output("y_scatter", "data"),
    #     Output("z_scatter", "data"),
    #     Input("x-slider", "value"),
    # )
    # def update_slide_data_x(x_val):

    #     mask = np.equal(data.x_raw, np.array(x_val))
    #     y_scatter = np.array(data.y_raw)[mask]
    #     z_scatter = np.array(data.z_raw)[mask]
    #     return y_scatter, z_scatter

    app.layout = serve_layout
    return app


########################## Run ##########################
# if __name__ == "__main__":
#     app.run_server(debug=True, port=6050, mode="inline")
