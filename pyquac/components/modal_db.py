# notes
"""
This file is for creating a modal pop-up object that contains 
database of heatmap instances 
"""

from dash.dependencies import Input, Output, State
from dash import callback, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from pyquac.settings import settings

style_data_conditional = [
    {
        "if": {"state": "active"},
        "backgroundColor": "rgba(150, 180, 225, 0.2)",
        "border": "1px solid blue",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0, 116, 217, .03)",
        "border": "1px solid blue",
    },
]

bd_columns = [
    settings.chip_column,
    settings.qubit_column,
    settings.spectroscopy_type_column,
]

table = dash_table.DataTable(
    data=None,
    columns=[{"name": i, "id": i} for i in bd_columns],
    id="tbl",
    style_data_conditional=style_data_conditional,
)

body = dbc.Container(
    [
        dbc.Label("Click a cell in the table:"),
        table,
        dbc.Alert(id="tbl_out"),
    ]
)

modal_db = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Spectroscopy Database")),
        dbc.ModalBody(body),
        dbc.ModalFooter(
            dbc.Button(
                "Pick and close", id="modal_db_close", className="ms-auto", n_clicks=0
            )
        ),
    ],
    id="modal_db",
    is_open=False,
    keyboard=False,
    backdrop="static",
)


@callback(Output("tbl_out", "children"), Input("tbl", "active_cell"))
def update_tbl(active_cell):
    return str(active_cell) if active_cell else "Click the table"


@callback(
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


@callback(
    Output("modal_db", "is_open"),
    [Input("modal_db_open", "n_clicks"), Input("modal_db_close", "n_clicks")],
    [State("modal_db", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output("current_data", "data"),
    Output("line-switches", "on"),
    Output("save_attributes", "data"),
    Input("tbl", "active_cell"),
    Input("modal_db_close", "n_clicks"),
    Input("database", "data"),
)
def change_fig(active_cell, n, database):
    if n:
        new_idx = active_cell["row"]
        new_save_attributes = dict(
            qubit_toggle=database["qubit_toggle"][new_idx],
            chip=database["chip"][new_idx],
            spectroscopy_type=database["type"][new_idx],
        )
        return new_idx, False, new_save_attributes
    else:
        raise PreventUpdate


@callback(Output("tbl", "style_data_conditional"), [Input("tbl", "active_cell")])
def update_selected_row_color(active):
    style = style_data_conditional.copy()
    if active:
        style.append(
            {
                "if": {"row_index": active["row"]},
                "backgroundColor": "rgba(150, 180, 225, 0.2)",
                "border": "1px solid blue",
            },
        )
    return style
