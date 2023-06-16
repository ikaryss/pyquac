# notes
"""
This file is for housing the main dash application.
This is where we define the various css items to fetch as well as the layout of our application.  
some tips https://budavariam.github.io/posts/2021/04/05/plotly-dash-development-tips/
"""

from dash import dcc, html, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash_bootstrap_templates import load_figure_template

import numpy as np
import pandas as pd
from datetime import datetime

# from pyquac.maindash import app
from pyquac.settings import settings
from pyquac.utils import initial_spectroscopy_in_app
from pyquac.components.navbar import navbar
from pyquac.components.sidebar import (
    sidebar,
    content,
    CONTENT_STYLE,
    SIDEBAR_STYLE,
    CONTENT_STYLE1,
    SIDEBAR_HIDEN,
)
from pyquac.components.modal import modal
from pyquac.components.fit_block import fit_block, FIT_STYLE, FIT_HIDEN
from pyquac.components.modal_db import modal_db

from pyquac.components.property_nav import (
    property_nav,
    _get_result_,
    _file_name_,
    _save_path_,
    PROPERTY_NAV_STYLE,
    PROPERTY_NAV_HIDEN,
)
from pyquac.components.heatmap import (
    figure_layout,
    GRAPH_HIDEN,
    GRAPH_STYLE,
    define_figure_simple,
)

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
    "z_click": None,
    "x_label": settings.init_x_label,
    "y_label": settings.init_y_label,
}

########################## App Layout ##########################


def conf_app(spectroscopy, cmap: str = settings.init_cmap):
    """configurate layout and dependencies for plotly dash app

    Args:
        spectroscopy (dict | list of dicts): each dict contains following keys:\n
        data - contains Spectroscopy instance\n
        chip - contains chip id or name\n
        qubit_toggle - contains name of qubit interaction (for instance: "q1" or "q1_q2_sweep")\n
        type - Spectroscopy type "TTS" or "STS"\n
        cmap (str, optional): color map for heatmap graph. Defaults to {settings.init_cmap}.

    Raises:
        PreventUpdate: if dict instance doesn't match the rules

    Returns:
        Dash app: ready to go plotly Jupyter dash application
    """

    # App Instance
    load_figure_template("ZEPHYR")

    app = JupyterDash(__name__)
    app.title = settings.app_name

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
                    dcc.Store(id=store[0], data=store[1], storage_type="memory")
                    for store in STORE.items()
                ],
                dcc.Store(id="z_store", data=data.njit_result, storage_type="memory"),
                dcc.Store(id="x_store", data=data.x_1d, storage_type="memory"),
                dcc.Store(id="y_store", data=data.y_1d, storage_type="memory"),
                dcc.Store(id="x_click_list", data=[], storage_type="memory"),
                dcc.Store(id="y_click_list", data=[], storage_type="memory"),
                dcc.Store(id="z_click_list", data=[], storage_type="memory"),
                dcc.Store(id="x_fit", data=[], storage_type="memory"),
                dcc.Store(id="y_fit", data=[], storage_type="memory"),
                dcc.Store(id="z_fit", data=[], storage_type="memory"),
                dcc.Store(
                    id="save_attributes",
                    data=dict(
                        qubit_toggle=qubit_toggle,
                        chip=chip,
                        spectroscopy_type=spectroscopy_type,
                    ),
                    storage_type="memory",
                ),
                dcc.Store(id="cmap", data=cmap, storage_type="memory"),
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
                            ],
                            # width=6,
                        ),
                        dbc.Col(
                            children=[
                                fit_block,
                            ],
                            # width=3,
                        ),
                    ]
                ),
            ]
        )

    app.layout = serve_layout()

    @app.callback(
        Output("x_store", "data"),
        Output("y_store", "data"),
        Output("z_store", "data"),
        Output("x_raw", "data"),
        Output("y_raw", "data"),
        Output("z_raw", "data"),
        Input("interval-graph-update", "n_intervals"),
        # Input("tbl", "active_cell")
        Input("current_data", "data"),
        Input("tbl", "selected_rows"),
        # Input("temp", "data"),
    )
    def update_fig_data(i, active_cell, row):
        button_clicked = ctx.triggered_id
        # print(temp)
        if button_clicked == "tbl":
            current_data = data_zoo[active_cell]
            return (
                current_data.x_1d,
                current_data.y_1d,
                current_data.njit_result,
                current_data.x_raw,
                current_data.y_raw,
                current_data.z_raw,
            )
        else:
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

    # heatmap
    @app.callback(
        Output("heatmap", "style"),
        Input("btn_sidebar", "n_clicks"),
        Input("side_click", "data"),
    )
    def toggle_graph(n, nclick):
        """function to hide and reveal sidebar

        Args:
            n (int): _description_
            nclick (int): _description_

        Returns:
            dict: style objects
        """
        if n:
            if nclick == "SHOW":
                graph_style = GRAPH_STYLE
            else:
                graph_style = GRAPH_HIDEN
        else:
            graph_style = GRAPH_STYLE

        return graph_style

    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="refresh_graph"),
        Output("heatmap", "figure"),
        Input("interval-graph-update", "n_intervals"),
        Input("x_click", "data"),
        Input("y_click", "data"),
        Input("heatmap", "clickData"),
        Input("modal_close", "n_clicks"),
        Input("modal_db_close", "n_clicks"),
        Input("fit", "n_clicks"),
        Input("x_fit", "data"),
        Input("y_fit", "data"),
        Input("fit_curve_show", "on"),
        Input("line-switches", "on"),
        State("x_store", "data"),
        State("y_store", "data"),
        State("z_store", "data"),
        State("heatmap", "figure"),
        State("y_scatter", "data"),
        State("yz_scatter", "data"),
        State("x_scatter", "data"),
        State("xz_scatter", "data"),
        State("x-title", "value"),
        State("y-title", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("x_scatter", "data"),
        Output("y_scatter", "data"),
        Output("xz_scatter", "data"),
        Output("yz_scatter", "data"),
        Output("x_click", "data"),
        Output("y_click", "data"),
        Output("z_click", "data"),
        Input("heatmap", "clickData"),
        State("x_raw", "data"),
        State("y_raw", "data"),
        State("z_raw", "data"),
    )
    def update_click_data(click, x_raw, y_raw, z_raw):
        if (click is None) or (
            click["points"][0]["curveNumber"]
            not in [
                0,
            ]
        ):
            raise PreventUpdate

        data_click = click["points"][0]
        x_click = data_click["x"]
        y_click = data_click["y"]
        z_click = data_click["z"]

        x_mask = np.equal(np.array(x_raw), np.array(x_click))
        y_mask = np.equal(np.array(y_raw), np.array(y_click))
        x_scatter = np.array(x_raw)[y_mask]

        # sort x and z array to prevent drop plot collapse
        x_sort_mask = np.argsort(x_scatter)
        x_scatter = x_scatter[x_sort_mask]

        y_scatter = np.array(y_raw)[x_mask]

        xz_scatter = np.array(z_raw)[y_mask][x_sort_mask]
        yz_scatter = np.array(z_raw)[x_mask]

        return (x_scatter, y_scatter, xz_scatter, yz_scatter, x_click, y_click, z_click)

    # modal DB
    @app.callback(Output("tbl_out", "children"), Input("tbl", "active_cell"))
    def update_tbl(active_cell):
        return str(active_cell) if active_cell else "Click the table"

    @app.callback(
        Output("tbl", "data"),
        Output("modal_db_open", "disabled"),
        Input("tbl", "data"),
        Input("database", "data"),
    )
    def control_tbl(tbl_data, database):
        if tbl_data is None:
            if database["db_disabled"] is True:
                return 1, True
            else:
                df = database.copy()
                try:
                    del df["db_disabled"]
                except KeyError:
                    pass
                return pd.DataFrame(df).to_dict("records"), False
        else:
            raise PreventUpdate

    @app.callback(
        Output("modal_db", "is_open"),
        [Input("modal_db_open", "n_clicks"), Input("modal_db_close", "n_clicks")],
        [State("modal_db", "is_open")],
    )
    def toggle_db_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    @app.callback(
        Output("current_data", "data"),
        Output("line-switches", "on"),
        Output("save_attributes", "data"),
        Output("fit_curve_show", "on"),
        Input("modal_db_close", "n_clicks"),
        Input("database", "data"),
        Input("tbl", "selected_rows"),
    )
    def change_fig(n, database, row):
        button_clicked = ctx.triggered_id
        if button_clicked == "tbl":
            new_idx = row[0]
            new_save_attributes = dict(
                qubit_toggle=database["qubit_toggle"][new_idx],
                chip=database["chip"][new_idx],
                spectroscopy_type=database["type"][new_idx],
            )
            return new_idx, False, new_save_attributes, False
        else:
            raise PreventUpdate

    # Modal
    @app.callback(
        Output("modal", "is_open"),
        Input("open_modal", "n_clicks"),
        Input("modal_close", "n_clicks"),
        State("modal", "is_open"),
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    # Navbar
    @app.callback(
        Output(component_id="about", component_property="is_open"),
        Output(component_id="about-popover", component_property="active"),
        Input(component_id="about-popover", component_property="n_clicks"),
        Input("about", "is_open"),
        Input("about-popover", "active"),
    )
    def about_popover(n, is_open, active):
        """toggle about popover

        Args:
            n (int): number of clicks
            is_open (bool): current state
            active (bool): _description_

        Returns:
            tuple(bool): status flip
        """
        if n:
            return not is_open, active
        return is_open, active

    # property nav
    @app.callback(
        Output("status-alert", "children"),
        Output("status-alert", "is_open"),
        Input("csv", "n_clicks"),
        Input("raw csv", "n_clicks"),
        Input("pdf", "n_clicks"),
        Input("svg", "n_clicks"),
        Input("html", "n_clicks"),
        Input("heatmap", "clickData"),
        Input("x_click", "data"),
        Input("y_click", "data"),
        Input("current_data", "data"),
        Input("interval-switches", "on"),
        Input("update-interval-value", "value"),
        Input("default-path", "value"),
        Input("additional-info", "value"),
        State("save_attributes", "data"),
        State("status-alert", "is_open"),
        State("x_store", "data"),
        State("y_store", "data"),
        State("z_store", "data"),
        State("heatmap", "figure"),
        State("cmap", "data"),
        State("x-title", "value"),
        State("y-title", "value"),
        State("fig-switches", "on"),
    )
    def save_func(
        _,
        __,
        ___,
        ____,
        _____,
        click,
        x_click,
        y_click,
        active_cell,
        on,
        new_interval,
        default_path,
        additional_info,
        save_attributes,
        is_open,
        x_result,
        y_result,
        z_result,
        fig,
        cmap,
        x_title,
        y_title,
        fig_switch,
    ):
        additional_info = "" if additional_info is None else additional_info
        button_clicked = ctx.triggered_id
        qubit_id, chip_id, spectroscopy_type = (
            save_attributes["qubit_toggle"],
            save_attributes["chip"],
            save_attributes["spectroscopy_type"].upper(),
        )
        filename = _file_name_(qubit_id, datetime.now().strftime("_%H-%M-%S"))
        path = _save_path_(filename, chip_id, default_path, spectroscopy_type)

        if button_clicked == "csv":
            _get_result_(x=x_result, y=y_result, z=z_result).to_csv(
                f"{path}_{additional_info}.csv"
            )
            return f"current data saved to {path}_{additional_info}.csv", True

        elif button_clicked == "raw csv":
            # _get_raw_result_(_get_result_(x=x_result, y=y_result, z=z_result)).to_csv(
            #     f"{path}_stacked.csv"
            # )
            if hasattr(data, "dump_data") and callable(getattr(data, "dump_data")):
                if active_cell is None:
                    data.dump_data(f"{path}_{additional_info}.pickle")
                else:
                    current_data = data_zoo[active_cell]
                    current_data.dump_data(f"{path}_{additional_info}.pickle")

                return f"current data saved to {path}_{additional_info}.pickle", True
            print(
                f"there is no __class__.dump_data() method in {data.__class__.__name__}"
            )

        elif button_clicked == "pdf":
            if fig_switch is True:
                define_figure_simple(
                    x=x_result,
                    y=y_result,
                    z=z_result,
                    x_axis_title=x_title,
                    y_axis_title=y_title,
                    cmap=cmap,
                ).write_image(f"{path}_{additional_info}.pdf")
            else:
                go.Figure(fig).write_image(f"{path}_{additional_info}.pdf")
            return f"current data saved to {path}_{additional_info}.pdf", True

        elif button_clicked == "svg":
            if fig_switch is True:
                define_figure_simple(
                    x=x_result,
                    y=y_result,
                    z=z_result,
                    x_axis_title=x_title,
                    y_axis_title=y_title,
                    cmap=cmap,
                ).write_image(f"{path}_{additional_info}.svg")
            else:
                go.Figure(fig).write_image(f"{path}_{additional_info}.svg")
            return f"current data saved to {path}_{additional_info}.svg", True

        elif button_clicked == "html":
            if fig_switch is True:
                define_figure_simple(
                    x=x_result,
                    y=y_result,
                    z=z_result,
                    x_axis_title=x_title,
                    y_axis_title=y_title,
                    cmap=cmap,
                ).write_html(f"{path}_{additional_info}.html")
            else:
                go.Figure(fig).write_html(f"{path}_{additional_info}.html")
            return f"current data saved to {path}_{additional_info}.html", True

        elif button_clicked == "heatmap":
            return f"clicked on x: {x_click}\ty: {y_click}", True

        elif button_clicked == "interval-switches":
            return f"Graph update is {on}", True

        elif button_clicked == "update-interval-value":
            return f"New update interval is {new_interval} ms.", True

        else:
            raise PreventUpdate

    @app.callback(
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

    # sidebar
    @app.callback(
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
        Output("x_selection", "on"),
        Input("btn_sidebar", "n_clicks"),
        State("x_selection", "on"),
        State("side_click", "data"),
    )
    def toggle_sidebar(n, x_selection, nclick):
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
                x_selection = False
            else:
                sidebar_style = SIDEBAR_STYLE
                content_style = CONTENT_STYLE
                cur_nclick = "SHOW"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"

        return sidebar_style, content_style, cur_nclick, x_selection

    @app.callback(
        Output("interval-graph-update", "max_intervals"),
        Input("interval-switches", "on"),
        Input("modal_db", "is_open"),
    )
    def toggle_checklist(switch_state, db_open):
        """function to change max interval property

        Args:
            n (_type_): _description_
            max_interval (_type_): _description_

        Returns:
            _type_: _description_
        """
        if db_open is False:
            if switch_state is True:
                new_max_interval = -1
            else:
                new_max_interval = 0
            return new_max_interval
        else:
            return 0

    @app.callback(
        Output("interval-graph-update", "interval"),
        Input("update-interval-value", "value"),
    )
    def change_interval_update(new_interval):
        if new_interval is not None:
            return new_interval
        else:
            raise PreventUpdate

    @app.callback(
        Output("xy_lines_state", "data"),
        Input("line-switches", "on"),
    )
    def toggle_xy_lines(switch_state):
        if switch_state is True:
            new_xy_state = True
        else:
            new_xy_state = False
        return new_xy_state

    # Fit block
    @app.callback(
        Output("fit_block", "style"),
        Input("btn_sidebar", "n_clicks"),
        Input("side_click", "data"),
    )
    def toggle_fit_block(n, nclick):
        """function to hide and reveal sidebar

        Args:
            n (int): _description_
            nclick (int): _description_

        Returns:
            dict: style objects
        """
        if n:
            if nclick == "SHOW":
                block_style = FIT_STYLE
            else:
                block_style = FIT_HIDEN
        else:
            block_style = FIT_STYLE

        return block_style

    @app.callback(
        Output("points_selected_output", "children"),
        Output("x_click_list", "data"),
        Output("y_click_list", "data"),
        Input("x_selection", "on"),
        Input("heatmap", "clickData"),
        Input("x_click", "data"),
        Input("y_click", "data"),
        Input("z_click", "data"),
        Input("x_click_list", "data"),
        Input("y_click_list", "data"),
        Input("z_click_list", "data"),
        Input("clear_selection", "n_clicks"),
    )
    def memorize_selection(
        x_selection,
        click,
        x_click,
        y_click,
        z_click,
        x_click_list,
        y_click_list,
        z_click_list,
        clear_sel_btn,
    ):
        element_clicked = ctx.triggered_id
        if x_selection & (element_clicked == "heatmap"):
            x_click_list.append(x_click)
            y_click_list.append(y_click)
            z_click_list.append(z_click)
            # print(x_click_list)
            return (
                f"Points selected: x: {x_click_list}",
                x_click_list,
                y_click_list,
            )
        else:
            if clear_sel_btn:
                return "Points selected: ", [], []
            else:
                raise PreventUpdate

    @app.callback(
        Output("x_fit", "data"),
        Output("y_fit", "data"),
        Input("x_click_list", "data"),
        Input("y_click_list", "data"),
        # Input("thres", "value"),
        # Input("min_dist", "value"),
        # Input("n_last", "value"),
        Input("deg", "value"),
        Input("res_bnd", "value"),
        Input("manual_fit", "on"),
        Input("current_data", "data"),
        Input("fit", "n_clicks"),
    )
    def fit_curve(x_list, y_list, deg, res_bnd, manual_fit, active_cell, fit):
        y_key = y_list if manual_fit is True else None
        element_clicked = ctx.triggered_id

        if element_clicked == "fit":
            if active_cell is None:
                approx = data.approximate(
                    poly_nop=len(data.x_list),
                    x_key=x_list,
                    y_key=y_key,
                    deg=deg,
                    resolving_zone=res_bnd,
                    # thres=thres,
                    # min_dist=min_dist,
                    # n_last=n_last,
                    print_info=False,
                )
                x_fit = data.x_list
                y_fit = approx["poly_line"]["y"]
                return x_fit, y_fit
            else:
                current_data = data_zoo[active_cell]
                approx = current_data.approximate(
                    poly_nop=len(data.x_list),
                    x_key=x_list,
                    y_key=y_key,
                    deg=deg,
                    # thres=thres,
                    # min_dist=min_dist,
                    # n_last=n_last,
                    print_info=False,
                )
                x_fit = approx["poly_line"]["x"]
                y_fit = approx["poly_line"]["y"]
                return x_fit, y_fit
        else:
            raise PreventUpdate

    # @app.callback(
    #     Output("collapse", "is_open"),
    #     [Input("collapse-button", "n_clicks")],
    #     [State("collapse", "is_open")],
    # )
    # def toggle_collapse(n, is_open):
    #     if n:
    #         return not is_open
    #     return is_open

    # app.clientside_callback(
    #     """
    # function(i) {
    #     triggered_id = i["layout"]["yaxis"]["range"];
    #     return triggered_id;
    # }
    # """,
    #     Output("temp", "data"),
    #     Input("heatmap", "figure"),
    #     prevent_initial_call=True,
    # )

    return app


########################## Run ##########################
# if __name__ == "__main__":
#     app.run_server(debug=True, port=6050, mode="inline")
