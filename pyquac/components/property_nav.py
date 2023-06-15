# # # notes
# # """
# # This file is for creating a property nav that will sit under main navigation bar.
# # Dash Bootstrap Components documentation linked below:
# # https://dash-bootstrap-components.opensource.faculty.ai/docs/components/nav/
# # """

import os
from datetime import date, datetime
import pandas as pd
import numpy as np

from dash.dependencies import Input, Output, State
from dash import ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from ..settings import settings
from ..components.heatmap import define_figure_simple


def _get_result_(x, y, z):
    return pd.DataFrame({"x_value": x, "y_value": y, "z_value": z})


def _get_raw_result_(get_res_df):
    y_l = (get_res_df.groupby("x_value")["y_value"].apply(list)).reset_index()
    z_l = get_res_df.groupby("x_value")["z_value"].apply(list)
    x_l = z_l.index.to_series(index=np.arange(len(z_l))).to_frame()
    z_l = z_l.reset_index()

    return x_l.merge(y_l, on="x_value", how="left").merge(z_l, on="x_value", how="left")


def _file_name_(qubit_id: str, time):
    return f"q{qubit_id}{time}"


def _save_path_(filename: str, chip_id: str, default_path: str, spectroscopy_type: str):
    spectroscopy = (
        "two_tone_spectroscopy"
        if spectroscopy_type == "TTS"
        else "single_tone_spectroscopy"
    )

    current_date = str(date.today())

    parent_dir = default_path
    dir_qubit = os.path.join(default_path, str(chip_id))
    dir_date = os.path.join(dir_qubit, current_date)
    dir_tts = os.path.join(dir_date, spectroscopy)

    if os.path.exists(parent_dir):

        # Checking qubit dir existance
        if not os.path.exists(dir_qubit):
            os.mkdir(dir_qubit)
        else:
            pass

        # Checking date dir existance
        if not os.path.exists(dir_date):
            os.mkdir(dir_date)
        else:
            pass

        # Checking tts dir existance
        if not os.path.exists(dir_tts):
            os.mkdir(dir_tts)
        else:
            pass

        dir_final = os.path.join(dir_tts, filename)
        return dir_final
    else:
        return filename


PROPERTY_NAV_STYLE = {
    "position": "fixed",
    "top": "60px",
    "left": "16rem",
    "overflow-x": "hidden",
    "transition": settings.transition_time,
    "padding": "0.5rem 1rem",
}

PROPERTY_NAV_HIDEN = {
    "position": "fixed",
    "top": "60px",
    "left": 0,
    "overflow-x": "hidden",
    "transition": settings.transition_time,
    "padding": "0.5rem 1rem",
}
property_nav = dbc.Nav(
    [
        dbc.Alert(
            "App is now running",
            color="light",
            id="status-alert",
            is_open=False,
            duration=10000,
        ),
    ],
    id="property-nav",
    style=PROPERTY_NAV_STYLE,
)


# @app.callback(
#     Output("status-alert", "children"),
#     Output("status-alert", "is_open"),
#     Input("csv", "n_clicks"),
#     Input("raw csv", "n_clicks"),
#     Input("pdf", "n_clicks"),
#     Input("svg", "n_clicks"),
#     Input("html", "n_clicks"),
#     Input("heatmap", "clickData"),
#     Input("x_click", "data"),
#     Input("y_click", "data"),
#     Input("interval-switches", "on"),
#     Input("update-interval-value", "value"),
#     Input("default-path", "value"),
#     State("save_attributes", "data"),
#     State("status-alert", "is_open"),
#     State("x_store", "data"),
#     State("y_store", "data"),
#     State("z_store", "data"),
#     State("heatmap", "figure"),
#     State("cmap", "data"),
#     State("x-title", "value"),
#     State("y-title", "value"),
#     State("fig-switches", "on"),
# )
# def save_func(
#     _,
#     __,
#     ___,
#     ____,
#     _____,
#     click,
#     x_click,
#     y_click,
#     on,
#     new_interval,
#     default_path,
#     save_attributes,
#     is_open,
#     x_result,
#     y_result,
#     z_result,
#     fig,
#     cmap,
#     x_title,
#     y_title,
#     fig_switch,
# ):
#     button_clicked = ctx.triggered_id
#     qubit_id, chip_id, spectroscopy_type = (
#         save_attributes["qubit_toggle"],
#         save_attributes["chip"],
#         save_attributes["spectroscopy_type"],
#     )
#     filename = _file_name_(qubit_id, datetime.now().strftime("_%H-%M-%S"))
#     path = _save_path_(filename, chip_id, default_path, spectroscopy_type)

#     if button_clicked == "csv":
#         _get_result_(x=x_result, y=y_result, z=z_result).to_csv(f"{path}.csv")
#         return f"current data saved to {path}.csv", True

#     elif button_clicked == "raw csv":
#         _get_raw_result_(_get_result_(x=x_result, y=y_result, z=z_result)).to_csv(
#             f"{path}_stacked.csv"
#         )
#         return f"current data saved to {path}_stacked.csv", True

#     elif button_clicked == "pdf":
#         if fig_switch is True:
#             define_figure_simple(
#                 x=x_result,
#                 y=y_result,
#                 z=z_result,
#                 x_axis_title=x_title,
#                 y_axis_title=y_title,
#                 cmap=cmap,
#             ).write_image(f"{path}.pdf")
#         else:
#             go.Figure(fig).write_image(f"{path}.pdf")
#         return f"current data saved to {path}.pdf", True

#     elif button_clicked == "svg":
#         if fig_switch is True:
#             define_figure_simple(
#                 x=x_result,
#                 y=y_result,
#                 z=z_result,
#                 x_axis_title=x_title,
#                 y_axis_title=y_title,
#                 cmap=cmap,
#             ).write_image(f"{path}.svg")
#         else:
#             go.Figure(fig).write_image(f"{path}.svg")
#         return f"current data saved to {path}.svg", True

#     elif button_clicked == "html":
#         if fig_switch is True:
#             define_figure_simple(
#                 x=x_result,
#                 y=y_result,
#                 z=z_result,
#                 x_axis_title=x_title,
#                 y_axis_title=y_title,
#                 cmap=cmap,
#             ).write_html(f"{path}.html")
#         else:
#             go.Figure(fig).write_html(f"{path}.html")
#         return f"current data saved to {path}.html", True

#     elif button_clicked == "heatmap":
#         return f"clicked on x: {x_click}\ty: {y_click}", True

#     elif button_clicked == "interval-switches":
#         return f"Graph update is {on}", True

#     elif button_clicked == "update-interval-value":
#         return f"New update interval is {new_interval} ms.", True

#     else:
#         raise PreventUpdate


# @app.callback(
#     Output("property-nav", "style"),
#     Input("btn_sidebar", "n_clicks"),
#     Input("side_click", "data"),
# )
# def toggle_nav(n, nclick):
#     """function to hide and reveal sidebar

#     Args:
#         n (int): _description_
#         nclick (int): _description_

#     Returns:
#         dict: style objects
#     """
#     if n:
#         if nclick == "SHOW":
#             nav_style = PROPERTY_NAV_STYLE
#         else:
#             nav_style = PROPERTY_NAV_HIDEN
#     else:
#         nav_style = PROPERTY_NAV_STYLE

#     return nav_style
