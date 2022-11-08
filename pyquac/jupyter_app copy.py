# Setup
from turtle import color
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from matplotlib.pyplot import cla
from jupyter_dash import JupyterDash

from settings import settings

# App Instance
theme = dbc.themes.ZEPHYR
css = settings.css_url
icons = dbc.icons.BOOTSTRAP
load_figure_template('ZEPHYR')

# the style arguments for the sidebar. We use position:fixed and a fixed width
# SIDEBAR_STYLE = {
#     "position": "fixed",
#     "top": "70px",
#     "left": 0,
#     "bottom": 0,
#     "width": "19rem",
#     "padding": "2rem 1rem",
#     "background-color": "#f8f9fa",
# }
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "70px",
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
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
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem"
}

APP_LOGO = settings.app_logo

app = JupyterDash(__name__, external_stylesheets=[theme, css, icons])
app.title = settings.app_name

########################## Sidebar ##########################
sidebar = html.Div(
    [
        html.H5("Spectroscopy"),
        html.Hr(),
        html.P(
            "A simple sidebar for data settings"
        ),
        dbc.Nav(
            [
                dbc.NavLink("btn 1", href="/", active="exact"),
                dbc.NavLink("btn 2", href="/page-1", active="exact"),
                dbc.NavLink("btn 3", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

########################## Navbar ##########################
# Input
COLLAPSE = html.Div(
    [
        dbc.Button(
            [
                html.I(className='fa fa-align-justify')
            ],
            id="btn_sidebar",
            className="offset-2",
            n_clicks=0,
        ),
    ]
)

LOGO = html.A(
        dbc.Row(
            [
                dbc.Col(html.Img(src=APP_LOGO, height='35px')),
                dbc.Col(dbc.NavbarBrand(settings.app_name, className="ms-2"))
            ],
            align='start',
            className="g-0"
        ),
        href=settings.app_link,
        style={"textDecoration": "none", "margin-left": "1rem"}
    )

ABOUT = dbc.NavItem(html.Div([
        dbc.NavLink('About', href='/', id='about-popover', active=True),
        dbc.Popover(id='about', is_open=False, target='about-popover', children=[
            dbc.PopoverHeader('How it Works'),
            dbc.PopoverBody('Some text')
        ])
    ]))

DROPDOWN = dbc.DropdownMenu(label='Links', in_navbar=True, children=[

                    dbc.DropdownMenuItem(
                        [
                        html.I(className='fa fa-linkedin'), ' Contacts'
                        ], 
                        href=settings.app_linkedin_url, target='_blank'),
                    
                    dbc.DropdownMenuItem(
                        [
                        html.I(className='fa fa-github'), ' Code'
                        ], 
                        href=settings.app_github_url, target='_blank')

                ],
                color='primary')

navbar = dbc.Navbar(class_name='nav nav-pills', children=[
    COLLAPSE,
    LOGO,
    ABOUT,
    DROPDOWN
    
                ], 
                color='primary', dark=True, expand='sm',
                )

# Callback
@app.callback(
    Output(component_id='about', component_property='is_open'),
    Output(component_id='about-popover', component_property='active'),
    Input(component_id='about-popover', component_property='n_clicks'),
    Input('about', 'is_open'),
    Input('about-popover', 'active')
)
def about_popover(n, is_open, active):
    if n:
        return not is_open, active
    return is_open, active


########################## App Layout ##########################

app.layout = html.Div(children = [
                dcc.Store(id='side_click'),
                # dbc.Row([
                #     dbc.Col(html.Br()),
                #     ]),
                dbc.Row(
                    [
                        dbc.Col(navbar),
                    ]),
                dbc.Row(
                    [dbc.Col(children=[
                        sidebar,
                        content
                        ],
                         width = 2),
                    ])
    ]
)

@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
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
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

########################## Run ##########################
if __name__ == "__main__":
    app.run_server(debug=True, port=6050, mode='inline')
