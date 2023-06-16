from dash.dependencies import Input, Output, State
from dash import html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc
from pyquac.settings import settings

FIT_STYLE = {
    "position": "fixed",
    "top": "70px",
    "left": "64rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": settings.transition_time,
    "padding": "0.5rem 1rem",
}

FIT_HIDEN = {
    "position": "fixed",
    "top": "70px",
    "left": "45rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": settings.transition_time,
    "padding": "0.5rem 1rem",
}


def tooltip_settings(setting_type: str, setting_descr: str):
    return html.Div(
        [
            html.P(
                [
                    html.Span(
                        setting_type,
                        id=f"tooltip-target_{setting_type}",
                        style={"textDecoration": "underline", "cursor": "pointer"},
                    ),
                ],
                className="small",
            ),
            dbc.Tooltip(
                setting_descr,
                target=f"tooltip-target_{setting_type}",
                autohide=False,
            ),
        ]
    )


def input_settings(
    setting_type: str,
    setting_descr: str,
    setting_id: str,
    min_value,
    max_value,
    default_value,
    disabled=True,
):
    return (
        dbc.NavItem(
            [
                dbc.Input(
                    id=setting_id,
                    type="number",
                    min=min_value,
                    max=max_value,
                    placeholder=setting_type,
                    disabled=disabled,
                    html_size=16,
                    size="sm",
                    value=default_value,
                ),
                tooltip_settings(setting_type, setting_descr),
            ]
        ),
    )


thres = input_settings(
    "Threshold",
    "thres : float between [0., 1.]\nNormalized threshold. Only the peaks with amplitude higher than the threshold will be detected",
    "thres",
    min_value=0.0,
    max_value=1.0,
    default_value=0.7,
)

min_dist = input_settings(
    "Minimum distance",
    "min_dist : int\nMinimum distance between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint.",
    "min_dist",
    min_value=1,
    max_value=100000,
    default_value=75,
)

n_last = input_settings(
    "N last",
    "n_last : int\nThe number of the highest values (relative to the mean), which are taken into account when searching for peak values",
    "n_last",
    min_value=1,
    max_value=100000,
    default_value=20,
)

deg = input_settings(
    "Degree value",
    "deg : int\nDegree of the fitting polynomial",
    "deg",
    min_value=1,
    max_value=100,
    default_value=2,
    disabled=False,
)

resolving_zone = input_settings(
    "Resolving band value",
    "deg : float (0:1)\nRelative part of the captured scan when applying the approximation.",
    "res_bnd",
    min_value=0.001,
    max_value=0.99,
    default_value=0.05,
    disabled=False,
)

# info_eq = (
#     "Polynomial: "
#     + r"$$p(x)=c_0+c_1x+...+c_nx^n$$"
#     + "\nCosine: "
#     + r"$$p(x)=A_1cos(2*\pi\omega x+\theta)$$"
# )
info_eq = r"$$p(x)=c_0+c_1x+...+c_nx^n$$"

# collapse = html.Div(
#     [
#         dbc.Button(
#             "Open fit settings",
#             id="collapse-button",
#             className="mb-3",
#             color="primary",
#             n_clicks=0,
#         ),
#         dbc.Collapse(
#             [
#                 # dbc.Card(dbc.CardBody("Settings for curve fitting")),
#                 dcc.Markdown(info_eq, className="small", mathjax=True),
#                 *deg,
#                 *resolving_zone,
#             ],
#             id="collapse",
#             is_open=False,
#         ),
#     ]
# )

fit_block = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavItem(html.H5("Fit block")),
                html.Hr(),
                dbc.NavItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Row(
                                        daq.BooleanSwitch(
                                            id="x_selection",
                                            on=False,
                                            color="#3459e6",
                                            label="Select x points",
                                        ),
                                    )
                                    # dbc.Button(
                                    #     "Select x points",
                                    #     id="x_selection",
                                    #     title="If no points are selected, then fits regarding all existing points",
                                    #     size="sm",
                                    # ),
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Clear selection",
                                        id="clear_selection",
                                        title="clearing cache",
                                        size="sm",
                                    ),
                                ),
                            ]
                        ),
                        # html.Div(
                        #     "If no points are selected, then fits regarding all existing points",
                        #     className="small",
                        # ),
                    ],
                    class_name="mt-3",
                ),
                dbc.NavItem(
                    [
                        html.Div(
                            children="Points selected: ", id="points_selected_output"
                        )
                    ],
                    class_name=["mt-2", "small", "mb-3"],
                ),
                # dbc.NavItem(
                #     dbc.RadioItems(
                #         options=[
                #             {"label": "Polynomial", "value": "poly"},
                #             {"label": "Cosine", "value": "cos"},
                #         ],
                #         value=1,
                #         id="radioitems-inline-input",
                #         inline=True,
                #     )
                # ),
                html.Hr(),
                dcc.Markdown(info_eq, className="small", mathjax=True),
                *deg,
                *resolving_zone,
                # *thres,
                # *min_dist,
                # *n_last,
                dbc.NavItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        "Fit based on manual selection",
                                        className="small",
                                    ),
                                    width=8,
                                ),
                                dbc.Col(
                                    daq.BooleanSwitch(
                                        id="manual_fit",
                                        on=True,
                                        color="#3459e6",
                                        disabled=True,
                                    ),
                                    width=4,
                                ),
                            ]
                        ),
                    ],
                    style={"margin-bottom": "0.5rem"},
                ),
                dbc.NavItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        "Show fitted curve on heatmap",
                                        className="small",
                                    ),
                                    width=8,
                                ),
                                dbc.Col(
                                    daq.BooleanSwitch(
                                        id="fit_curve_show",
                                        on=False,
                                        color="#3459e6",
                                    ),
                                    width=4,
                                ),
                            ]
                        ),
                    ],
                    style={"margin-bottom": "0.5rem"},
                ),
                dbc.NavItem(
                    dbc.Button(
                        "Fit Curve",
                        id="fit",
                        title="Fit Curve",
                    ),
                ),
            ]
        ),
    ],
    style=FIT_STYLE,
    id="fit_block",
)
