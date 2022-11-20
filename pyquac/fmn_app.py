# notes
"""
This file is for housing the main dash application.
This is where we define the various css items to fetch as well as the layout of our application.
"""

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from jupyter_dash import JupyterDash

from pyquac.settings import settings
from pyquac.components.navbar import navbar
from pyquac.components.sidebar import sidebar, content
from pyquac.components.property_nav import property_nav
from pyquac.components.heatmap import figure_layout

# App Instance
THEME = dbc.themes.ZEPHYR
CSS = settings.css_url
ICONS = dbc.icons.BOOTSTRAP
load_figure_template("ZEPHYR")


app = JupyterDash(__name__, external_stylesheets=[THEME, CSS, ICONS])
app.title = settings.app_name


########################## App Layout ##########################


def conf_app(
    data,
    x_axis_title: str = "Voltages, V",
    y_axis_title: str = "Frequencies, GHz",
    cmap: str = "rdylbu",
):
    def serve_layout():
        """Define the layout of the application

        Returns:
            DIV component: app layout
        """
        return html.Div(
            children=[
                dcc.Store(id="side_click", data="SHOW"),
                dbc.Row(
                    [
                        dbc.Col(navbar),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(children=[sidebar, content], width=3),
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
                                            data, x_axis_title, y_axis_title, cmap
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    app.layout = serve_layout
    return app


########################## Run ##########################
# if __name__ == "__main__":
#     app.run_server(debug=True, port=6050, mode="inline")
