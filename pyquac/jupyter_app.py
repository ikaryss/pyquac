# Setup
from turtle import color
import dash
from dash.dependencies import Input, Output
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
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "70px",
    "left": 0,
    "bottom": 0,
    "width": "19rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

APP_LOGO = settings.app_logo_url

app = JupyterDash(name=settings.app_name, external_stylesheets=[theme, css, icons])
app.title = settings.app_name

########################## Navbar ##########################
# Input
navbar = dbc.Navbar(class_name='nav nav-pills', children=[
    # LOGO
    html.A(
        dbc.Row(
            [
                dbc.Col(html.Img(src=APP_LOGO, height='40px')),
                dbc.Col(dbc.NavbarBrand(settings.app_name, className="ms-2"))
            ],
            align='center',
            className="g-0"
        ),
        href=settings.app_link,
        style={"textDecoration": "none", "margin-left": "3rem"}
    ),

    # ABOUT
    dbc.NavItem(html.Div([
        dbc.NavLink('About', href='/', id='about-popover', active=True),
        dbc.Popover(id='about', is_open=False, target='about-popover', children=[
            dbc.PopoverHeader('How it Works'),
            dbc.PopoverBody('Some text')
        ])
    ])),

    # DROPDOWN
    dbc.DropdownMenu(label='Links', in_navbar=True, children=[

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
                ], 
                color='primary', dark=True
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


########################## Sidebar ##########################
sidebar = html.Div(
    [
        html.H5("Spectroscopy", className="display-6"),
        html.Hr(),
        html.P(
            "A simple sidebar for data settings", className="lead"
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
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

########################## App Layout ##########################

app.layout = html.Div(children = [
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

# app.layout = dbc.Container(fluid=True, children=[
#     html.Br(),
#     navbar, 
#     sidebar, content
# ])

########################## Run ##########################
if __name__ == "__main__":
    app.run_server(debug=True)
