"""
plot tools provides fast and elegant solution for plotting quantum experiment data
"""
from typing import Union, Iterable
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from PIL import Image
from datetime import date, datetime

from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import callback_context


def _save_path(data: str, qubit_name: str = 'qubit_default_name',
               default_path: str = 'D:/Scripts/Measurement_automation/data/qubits/',
               spectroscopy_type: str = 'tts'):
    if spectroscopy_type == 'tts':
        spectroscopy = 'two_tone_spectroscopy'
    else:
        spectroscopy = 'single_tone_spectroscopy'

    current_date = str(date.today())

    parent_dir = default_path
    dir_qubit = os.path.join(default_path, str(qubit_name))
    dir_date = os.path.join(dir_qubit, current_date)
    dir_tts = os.path.join(dir_date, spectroscopy)

    if os.path.exists(parent_dir):

        "checking qubit dir existance"
        if not os.path.exists(dir_qubit):
            os.mkdir(dir_qubit)
        else:
            pass

        "checking date dir existance"
        if not os.path.exists(dir_date):
            os.mkdir(dir_date)
        else:
            pass

        "checking tts dir existance"
        if not os.path.exists(dir_tts):
            os.mkdir(dir_tts)
        else:
            pass

        dir_final = os.path.join(dir_tts, data)
        return dir_final
    else:
        return data


class Heatmap:
    """
    Heatmap class provides easy-to-use workspace for building interactive two tone spectroscopy plots
    """

    @staticmethod
    def plot_figure(data: pd.DataFrame = None, *, x: Union[str, np.ndarray, pd.Series, Iterable] = None,
                    y: Union[str, np.ndarray, pd.Series, Iterable] = None,
                    z: Union[str, np.ndarray, pd.Series, Iterable] = None,
                    theme: str = 'dark',
                    cmap: Union[str, Iterable] = ['#ffd200', '#cb2d3e'],
                    title: str = 'Two Tone Spectroscopy',
                    sub_title: str = None,
                    x_axis_title: str = 'Voltages, V',
                    x_axis_size: int = 13,
                    y_axis_title: str = 'Frequencies, GHz',
                    y_axis_size: int = 13,
                    colorbar_text: str = None,
                    logo: bool = False,
                    logo_local=None,
                    y_logo_pos: float = 1.09,
                    x_logo_pos: float = 1,
                    logo_size: float = 0.09,
                    qubit_number: Union[str, float] = None,
                    exponent_mode=False,
                    width=700,
                    height=600
                    ):

        """
        Built a complete plot of the Two Tone Spectroscopy
        :param data: (optional) Pandas data frame with 3 columns (x, y, z)
        :param x: (optional) Pandas series obj | array-like obj. Currents of Two Tone Spectroscopy
        :param y: (optional) Pandas series obj | array-like obj. Frequencies of Two Tone Spectroscopy
        :param z: (optional) Pandas series obj | array-like obj. Response of Two Tone Spectroscopy
        :param theme: ['white', 'dark'] - plot style. default - 'dark'
        :param cmap: The 'colorscale' property is a colorscale and may be specified as:
                      - A list of colors that will be spaced evenly to create the colorscale.
                        Many predefined colorscale lists are included in the sequential, diverging,
                        and cyclical modules in the plotly.colors package.
                      - A list of 2-element lists where the first element is the
                        normalized color level value (starting at 0 and ending at 1),
                        and the second item is a valid color string.
                        (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
                      - One of the following named colorscales:
                            ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
                             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
                             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
                             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
                             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
                             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
                             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
                             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
                             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
                             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
                             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
                             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
                             'ylorrd'].
        :param title: (Optional) Sets the plot's title. Default - 'Two Tone Spectroscopy'
        :param sub_title: (Optional) additional information about the plot. Default - None
        :param x_axis_title: (Optional) Sets the title of this axis. Default - 'Currents, A'
        :param x_axis_size: size of x axis font
        :param y_axis_title: (Optional) Sets the title of this axis. Default - 'Frequencies, Hz'
        :param y_axis_size: size of y axis font
        :param colorbar_text: (Optional) Sets the title of color bar. Default - None
        :param logo: (Optional) If True - shows the FMN logo at the upper right corner. Default - True
        :param logo_local: (Optional) If not None - puts <filename> file as a logo
        :param y_logo_pos: sets y logo position
        :param x_logo_pos: sets x logo position
        :param logo_size: logo scale parameter
        :param qubit_number: qubit information
        :param exponent_mode: if False - ignore exponential formatting
        :param width: the width of the plot (in pixels)
        :param height: the height of the plot (in pixels)
        :return:
        """

        if data is not None:
            fig = go.Figure(data=go.Heatmap(
                z=data.iloc[:, 2],
                x=data.iloc[:, 0],
                y=data.iloc[:, 1],
                colorscale=cmap))
        else:
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=cmap))

        """Adding subtitle"""
        Sub_title = '' if sub_title is None else '<br>' + '<i>' + sub_title + '</i>'
        Title = title + Sub_title if qubit_number is None else title + ' (qubit ' + str(qubit_number) + ')' + Sub_title

        """Choosing colors"""
        if theme.lower() == 'white':
            fig.update_layout(template="plotly_white")
            title_clr = '#000000'
            bg_clr = '#F5F5F5'
            paper_clr = '#FAFAFA'
            grid_clr = '#757575'
        elif theme.lower() == 'dark':
            fig.update_layout(template="plotly_dark")
            title_clr = '#ffffff'
            bg_clr = '#1c1c1c'
            paper_clr = '#1c1c1c'
            grid_clr = '#757575'
        elif theme.lower() == 'pure_white':
            fig.update_layout(template="plotly_white")
            title_clr = '#000000'
            bg_clr = '#ffffff'
            paper_clr = '#ffffff'
            grid_clr = '#757575'
        else:
            fig.update_layout(template="plotly_dark")
            raise NameError

        """layout settings. See more https://plotly.com/python/reference/layout/#layout-title"""

        fig.update_layout(title={
            'text': Title,
            'font': {
                'color': title_clr
            },
            'x': 0.086
        },
            font=dict(family='Open Sans', color=title_clr),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            autosize=False,
            separators='.',
            paper_bgcolor=paper_clr,
            plot_bgcolor=bg_clr)

        fig.update_yaxes(title_font={"size": y_axis_size}, tickfont_size=y_axis_size)
        fig.update_xaxes(title_font={"size": x_axis_size}, tickfont_size=x_axis_size)

        """figure size"""
        # if width is not None:
        #     fig.update_layout(width=width)
        # elif height is not None:
        #     fig.update_layout(height=height)
        # elif (width is not None) and (height is not None):
        #     fig.update_layout(width=width, height=height)
        # else:
        #     pass
        fig.update_layout(width=width, height=height)

        """upd numbers formatting. See more https://plotly.com/python/tick-formatting/"""
        if not exponent_mode:
            fig.update_layout(yaxis=dict(showexponent='none', exponentformat='e'))
            fig.update_traces(zhoverformat='.2f')
            # fig.update_layout(xaxis=dict(showexponent='none', exponentformat='e'))
        else:
            fig.update_layout(yaxis_tickformat='.2e', xaxis_tickformat='.2e')
            fig.update_traces(zhoverformat='.2f')

        """add color bar text"""
        if colorbar_text is not None:
            fig.update_traces(colorbar_title_text=colorbar_text)
        else:
            pass

        """adding logo. See more https://plotly.com/python/images/"""
        if logo:
            fig.add_layout_image(
                dict(
                    source='https://raw.githubusercontent.com/ikaryss/pyquac/Master/images/logo_sign.png',
                    xref="paper", yref="paper",
                    x=x_logo_pos, y=y_logo_pos,
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="right", yanchor="bottom",
                    opacity=1,
                    layer="above"))
        else:
            pass

        if logo_local is not None:
            img = Image.open('logo_sign.png')
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="paper", yref="paper",
                    x=x_logo_pos, y=y_logo_pos,
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="right", yanchor="bottom",
                    opacity=1,
                    layer="above"))
        else:
            pass

        """grid color. See more https://plotly.com/python/axes/"""
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor=grid_clr, gridwidth=0.1,
                                     zeroline=True, zerolinewidth=0.1, zerolinecolor=grid_clr),
                          yaxis=dict(showgrid=True, gridcolor=grid_clr, gridwidth=0.1,
                                     zeroline=True, zerolinewidth=0.1, zerolinecolor=grid_clr))

        return fig

    @staticmethod
    def decorate_figure(fig, *,
                        theme: str = 'dark',
                        cmap: Union[str, Iterable] = ['#ffd200', '#cb2d3e'],
                        title: str = 'Two Tone Spectroscopy',
                        sub_title: str = None,
                        x_axis_title: str = 'Voltages, V',
                        x_axis_size: int = 13,
                        y_axis_title: str = 'Frequencies, GHz',
                        y_axis_size: int = 13,
                        colorbar_text: str = None,
                        logo: bool = False,
                        logo_local=False,
                        y_logo_pos: float = 1.09,
                        x_logo_pos: float = 1,
                        logo_size: float = 0.09,
                        qubit_number: Union[str, float] = None,
                        exponent_mode=False,
                        width=None,
                        height=None
                        ):
        """
        Built a complete plot formatter for your go.Fig object.
        :param fig: plotly.graph_objs._figure.Figure object
        :param theme: ['white', 'dark'] - plot style. default - 'dark'
        :param cmap: The 'colorscale' property is a colorscale and may be specified as:
                      - A list of colors that will be spaced evenly to create the colorscale.
                        Many predefined colorscale lists are included in the sequential, diverging,
                        and cyclical modules in the plotly.colors package.
                      - A list of 2-element lists where the first element is the
                        normalized color level value (starting at 0 and ending at 1),
                        and the second item is a valid color string.
                        (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
                      - One of the following named colorscales:
                            ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
                             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
                             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
                             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
                             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
                             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
                             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
                             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
                             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
                             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
                             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
                             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
                             'ylorrd'].
        :param title: (Optional) Sets the plot's title. Default - 'Two Tone Spectroscopy'
        :param sub_title: (Optional) additional information about the plot. Default - None
        :param x_axis_title: (Optional) Sets the title of this axis. Default - 'Currents, A'
        :param y_axis_title: (Optional) Sets the title of this axis. Default - 'Frequencies, Hz'
        :param colorbar_text: (Optional) Sets the title of color bar. Default - None
        :param logo: (Optional) If True - shows the FMN logo at the upper right corner. Default - True
        :param y_logo_pos: y position of the logo. Default = 1.09. Usually belongs to [0.9, 1.1] interval
        """

        """More figure|plot settings here: https://plotly.com/python/reference/layout/"""

        """Adding subtitle"""
        Sub_title = '' if sub_title is None else '<br>' + '<i>' + sub_title + '</i>'
        Title = title + Sub_title if qubit_number is None else title + ' (qubit ' + str(qubit_number) + ')' + Sub_title

        """Choosing colors"""
        if theme.lower() == 'white':
            fig.update_layout(template="plotly_white")
            title_clr = '#000000'
            bg_clr = '#F5F5F5'
            paper_clr = '#FAFAFA'
            grid_clr = '#757575'
        elif theme.lower() == 'dark':
            fig.update_layout(template="plotly_dark")
            title_clr = '#ffffff'
            bg_clr = '#1c1c1c'
            paper_clr = '#1c1c1c'
            grid_clr = '#757575'
        elif theme.lower() == 'pure_white':
            fig.update_layout(template="plotly_white")
            title_clr = '#000000'
            bg_clr = '#ffffff'
            paper_clr = '#ffffff'
            grid_clr = '#757575'
        else:
            fig.update_layout(template="plotly_dark")
            raise NameError

        """Color scale plotly. See more https://plotly.com/python/builtin-colorscales/"""
        fig.update_traces(colorscale=cmap)

        """layout settings. See more https://plotly.com/python/reference/layout/#layout-title"""

        fig.update_layout(title={
            'text': Title,
            'font': {
                'color': title_clr
            },
            'x': 0.086
        },
            font=dict(family='Open Sans', color=title_clr),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            autosize=False,
            separators='.',
            paper_bgcolor=paper_clr,
            plot_bgcolor=bg_clr)

        fig.update_yaxes(title_font={"size": y_axis_size}, tickfont_size=y_axis_size)
        fig.update_xaxes(title_font={"size": x_axis_size}, tickfont_size=x_axis_size)

        """figure size"""
        if width is not None:
            fig.update_layout(width=width)
        elif height is not None:
            fig.update_layout(height=height)
        elif (width is not None) and (height is not None):
            fig.update_layout(width=width, height=height)
        else:
            pass

        """upd numbers formatting. See more https://plotly.com/python/tick-formatting/"""
        if not exponent_mode:
            fig.update_layout(yaxis=dict(showexponent='none', exponentformat='e'))
            # fig.update_layout(xaxis=dict(showexponent='none', exponentformat='e'))
            fig.update_traces(zhoverformat='.2f')
        else:
            fig.update_layout(yaxis_tickformat='.2e', xaxis_tickformat='.2e')
            fig.update_traces(zhoverformat='.2f')

        """add color bar text"""
        if colorbar_text is not None:
            fig.update_traces(colorbar_title_text=colorbar_text)
        else:
            pass

        """adding logo. See more https://plotly.com/python/images/"""
        if logo:
            fig.add_layout_image(
                dict(
                    source='https://raw.githubusercontent.com/ikaryss/pyquac/Master/images/logo_sign.png',
                    xref="paper", yref="paper",
                    x=x_logo_pos, y=y_logo_pos,
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="right", yanchor="bottom",
                    opacity=1,
                    layer="above"))
        else:
            pass

        if logo_local:
            img = Image.open('logo_sign.png')
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="paper", yref="paper",
                    x=x_logo_pos, y=y_logo_pos,
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="right", yanchor="bottom",
                    opacity=1,
                    layer="above"))
        else:
            pass

        """grid color. See more https://plotly.com/python/axes/"""
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor=grid_clr, gridwidth=0.1,
                                     zeroline=True, zerolinewidth=0.1, zerolinecolor=grid_clr),
                          yaxis=dict(showgrid=True, gridcolor=grid_clr, gridwidth=0.1,
                                     zeroline=True, zerolinewidth=0.1, zerolinecolor=grid_clr))

        pass

    """
    info https://plotly.com/python/static-image-export/
    requirements:
    conda install -c conda-forge python-kaleido
    
    """

    @staticmethod
    def save_fig_as(fig, name: str = 'figure.svg', method: str = 'image', create_folder: bool = False,
                    folder_name: str = 'images'):
        """

        :param fig:
        :param name: export format options for images: [.svg (default), .pdf, .png, .jpeg, .webp]
                     HTML export
        :param method: exporting method [image (default), html]
        :param create_folder:
        :param folder_name:
        :return:
        """
        if create_folder == True:
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            directory = folder_name + '/' + name

        else:
            directory = name

        if method == 'image':
            fig.write_image(directory)

        elif method == 'html':
            fig.write_html(directory)


class Dash_app:

    def __init__(self, data, fig, qubit_id=None, chip_id=None, additional_info: str = '', spectroscopy_type='tts'):
        """

        :param data: entity of Tts | Sts or TwoToneSpectroscopy class
        :param fig: displayed figure
        :param qubit_id: id of qubit
        :param chip_id: id of chip
        :param spectroscopy_type: tts or sts
        """
        self.interval = None
        self.port = None
        self.mode = None
        self.data = data
        self.fig = fig
        self.qubit_id = qubit_id
        self.chip_id = chip_id
        self.additional_info = '_' + additional_info + '_'
        self.spectroscopy_type = spectroscopy_type
        self._ts = 'TTS_' if self.spectroscopy_type == 'tts' else 'STS_'

    def __middle_name(self, time):
        if self.qubit_id is not None:
            return f'q{self.qubit_id}{self.additional_info}{str(time)}'
        else:
            return f'q_untitled{self.additional_info}{str(time)}'

    def configure_app(self, mode='inline', port=8051, interval=4e3):
        """
        Configures Dashboard app
        :param mode: 'inline', 'external'
        :param port:
        :param interval: default update time in ms
        :return:
        """
        self.mode = mode
        self.interval = interval
        self.port = port

        disabled_btn = False
        disabled_input = False
        maxx_interval = -1

        # Build App
        app = JupyterDash(__name__)
        app.layout = html.Div(
            [
                html.Button('raw CSV', id='btn-nclicks-1', n_clicks=0),
                html.Button('CSV', id='btn-nclicks-6', n_clicks=0),
                html.Button('PDF', id='btn-nclicks-2', n_clicks=0),
                html.Button('HTML', id='btn-nclicks-3', n_clicks=0),
                html.Button('SVG', id='btn-nclicks-5', n_clicks=0),
                dcc.Checklist(id='checkbox',
                              options=[{'label': 'stop live upd', 'value': 'NO'}],
                              value=['YES', 'NO'],
                              labelStyle={'display': 'inline-block'}
                              ),
                html.Button('Manual upd', id='btn-nclicks-4', n_clicks=0),
                dcc.Input(
                    id="input_time",
                    type='number',
                    placeholder="Auto upd in, ms",
                    min=800,
                ),
                dcc.Graph(id="heatmap", figure=self.fig),
                dcc.Interval(id="animateInterval", interval=self.interval, n_intervals=0,
                             max_intervals=maxx_interval),
            ],
        )

        @app.callback(
            Output("heatmap", "figure"),
            Output("animateInterval", "max_intervals"),
            Output("animateInterval", "interval"),
            Output('btn-nclicks-4', 'disabled'),
            Output('input_time', 'disabled'),
            Input('btn-nclicks-1', 'n_clicks'),
            Input('btn-nclicks-6', 'n_clicks'),
            Input('btn-nclicks-2', 'n_clicks'),
            Input('btn-nclicks-3', 'n_clicks'),
            Input('btn-nclicks-5', 'n_clicks'),
            Input('checkbox', 'value'),
            Input('btn-nclicks-4', 'n_clicks'),
            Input('input_time', 'value'),
            Input("animateInterval", "n_intervals"),
        )
        def doUpdate(btn1, btn6, btn2, btn3, btn5, chkbx, btn4, time_val, i):

            if chkbx[-1] == 'NO':
                maxx_interval = 0
                disabled_btn = False
                disabled_input = False
            else:
                maxx_interval = -1
                disabled_btn = True
                disabled_input = True

            changed_id = [p['prop_id'] for p in callback_context.triggered][0]

            # CSV
            if 'btn-nclicks-1' in changed_id:
                file_name = f'{self._ts}raw_{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.csv'
                self.data.get_result().to_csv(
                    _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # CSV FORMATTED
            elif 'btn-nclicks-6' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.csv'
                self.data.get_raw_result().to_csv(
                    _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # PDF
            elif 'btn-nclicks-2' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.pdf'
                self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # HTML
            elif 'btn-nclicks-3' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.html'
                self.fig.write_html(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # SVG
            elif 'btn-nclicks-5' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.svg'
                self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            elif 'btn-nclicks-4' in changed_id:
                z = self.data.njit_result
                self.fig.update_traces(z=z)
            else:
                pass

            # Change time interval for updating heatmap
            if time_val:
                self.interval = time_val

            z = self.data.njit_result

            return self.fig.update_traces(z=z), maxx_interval, self.interval, disabled_btn, disabled_input

        return app.run_server(mode=self.mode, port=self.port)

    def save_all(self,
                 *, csv=True, pdf=True, html=True, svg=True):
        """
        Saves data in different formats
        :param csv: if true - saves csv
        :param pdf: if true - saves pdf
        :param html: if true - saves html
        :param svg: if true - saves svg
        :return:
        """

        # CSV
        if csv:
            file_name = f'{self._ts}raw_{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.csv'
            self.data.get_result().to_csv(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type),
                                          index=False)

        # CSV FORMATTED
        if csv:
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.csv'
            self.data.get_raw_result().to_csv(
                _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

        # PDF
        if pdf:
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.pdf'
            self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

        # HTML
        if html:
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.html'
            self.fig.write_html(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

        # SVG
        if svg:
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"))}.svg'
            self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))


class Dash_app_2Q:
    def __init__(self, *,
                 data1,
                 fig,
                 data2=None, data3=None,
                 data4=None, data5=None, data6=None,
                 Q1_id=None, Q2_id=None, Coupler_id=None,
                 chip_id=None,
                 check=False,
                 additional_info: str = '', spectroscopy_type='sts'
                 ):
        self.data11, self.data12, self.data13 = data1, data2, data3
        self.data21, self.data22, self.data23 = data4, data5, data6

        self.data_zoo = np.array([self.data11, self.data12, self.data13,
                                  self.data21, self.data22, self.data23])

        'block for dropdown menu'
        none_count = 0
        for data in self.data_zoo:
            if data is None:
                none_count += 1

        self.existing_data = self.data_zoo[:-none_count]
        self.disabled_drpdwn = False if none_count < 5 else True

        self.data_zoo_names = []
        data_zoo_names_temp = np.array([self.get_label_name(d) for d in self.data_zoo[:-none_count]]) if none_count > 0 \
            else np.array([self.get_label_name(d) for d in self.data_zoo])
        for name in data_zoo_names_temp:
            self.data_zoo_names.append(name)

        while len(self.data_zoo_names) < len(self.data_zoo):
            self.data_zoo_names.append('undefined')
        'end of block for dropdown menu'

        self.data_zoo_names = np.array(self.data_zoo_names)

        self.disabled_markers = []
        for name in self.data_zoo_names:
            if name == 'undefined':
                self.disabled_markers.append(True)
            else:
                self.disabled_markers.append(False)

        self.fig_zoo = {
            self.data_zoo_names[0]: 0, self.data_zoo_names[1]: 1, self.data_zoo_names[2]: 2,
            self.data_zoo_names[3]: 3, self.data_zoo_names[4]: 4, self.data_zoo_names[5]: 5,
        }

        self.main_data = self.data11
        self.picked_data = None

        self.fig = fig

        self.Q1_id, self.Q2_id, self.Coupler_id = Q1_id, Q2_id, Coupler_id
        self.chip_id = chip_id

        self.additional_info = '_' + additional_info + '_'
        self.spectroscopy_type = spectroscopy_type
        self._ts = 'STS_' if self.spectroscopy_type == 'sts' else 'TTS_'

        self.check = check

        self.__dict_of_data = dict(fig11=self.data11, fig12=self.data12, fig13=self.data13,
                                   fig21=self.data21, fig22=self.data22, fig23=self.data23)

        pass

    def __middle_name(self, time, data):
        if not self.check:
            mid_name = f'{data.readout_qubit}_{data.flux_qubit}_sweep{self.additional_info}{str(time)}'
        else:
            mid_name = f'check_{data.readout_qubit}_{data.flux_qubit}_sweep{self.additional_info}{str(time)}'
        return mid_name

    def __get_id(self, qubit):
        if qubit == 'Q1':
            qid = self.Q1_id
        elif qubit == 'Q2':
            qid = self.Q2_id
        elif qubit == 'Coupler':
            qid = self.Coupler_id
        else:
            qid = 'Nan'
        return str(qid)

    def get_label_name(self, data):
        readout = data.readout_qubit
        flux = data.flux_qubit
        if isinstance(flux, str):
            return f'{readout}_{flux}_sweep'
        else:
            global_str = str(readout) + '_'
            for f in flux:
                global_str += str(f) + '+'
            global_str = global_str[:-1]
            global_str += '_sweep'
            return global_str

    def configure_app(self, mode='inline', port=8051, interval=4e3):
        """
        Configures Dashboard app
        :param mode: 'inline', 'external'
        :param port:
        :param interval: default update time in ms
        :return:
        """
        self.mode = mode
        self.interval = interval
        self.port = port

        disabled_btn = False
        disabled_input = False
        maxx_interval = -1

        # Build App
        app = JupyterDash(__name__)
        app.layout = html.Div(
            [
                html.Button('raw CSV', id='btn-nclicks-1', n_clicks=0),
                html.Button('CSV', id='btn-nclicks-6', n_clicks=0),
                html.Button('PDF', id='btn-nclicks-2', n_clicks=0),
                html.Button('HTML', id='btn-nclicks-3', n_clicks=0),
                html.Button('SVG', id='btn-nclicks-5', n_clicks=0),
                dcc.Dropdown(
                    options=[{'label': value, 'value': value, 'disabled': disabl} for value, disabl in
                             zip(self.data_zoo_names, self.disabled_markers)],
                    value=self.get_label_name(self.data11), id='fig-dropdown',
                    disabled=self.disabled_drpdwn,
                    clearable=False
                ),

                dcc.Checklist(id='checkbox',
                              options=[{'label': 'stop live upd', 'value': 'NO'}],
                              value=['YES', 'NO'],
                              labelStyle={'display': 'inline-block'}
                              ),
                html.Button('Manual upd', id='btn-nclicks-4', n_clicks=0),
                dcc.Input(
                    id="input_time",
                    type='number',
                    placeholder="Auto upd in, ms",
                    min=800,
                ),

                dcc.Graph(id="heatmap", figure=self.fig),

                dcc.Interval(id="animateInterval", interval=self.interval, n_intervals=0,
                             max_intervals=maxx_interval),
            ],
        )

        @app.callback(
            Output("heatmap", "figure"),
            Output("animateInterval", "max_intervals"),
            Output("animateInterval", "interval"),
            Output('btn-nclicks-4', 'disabled'),
            Output('input_time', 'disabled'),
            Input('btn-nclicks-1', 'n_clicks'),
            Input('btn-nclicks-6', 'n_clicks'),
            Input('btn-nclicks-2', 'n_clicks'),
            Input('btn-nclicks-3', 'n_clicks'),
            Input('btn-nclicks-5', 'n_clicks'),
            Input('fig-dropdown', 'value'),
            Input('checkbox', 'value'),
            Input('btn-nclicks-4', 'n_clicks'),
            Input('input_time', 'value'),
            Input("animateInterval", "n_intervals"),
        )
        def doUpdate(btn1, btn6, btn2, btn3, btn5, fig_dropd_value, chkbx, btn4, time_val, i):

            self.picked_data = self.data_zoo[
                self.fig_zoo[fig_dropd_value]] if self.disabled_drpdwn is False else self.data11

            if chkbx[-1] == 'NO':
                maxx_interval = 0
                disabled_btn = False
                disabled_input = False
            else:
                maxx_interval = -1
                disabled_btn = True
                disabled_input = True

            changed_id = [p['prop_id'] for p in callback_context.triggered][0]

            # CSV
            if 'btn-nclicks-1' in changed_id:
                file_name = f'{self._ts}raw_{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), self.picked_data)}' \
                            f'.csv'
                self.picked_data.get_result().to_csv(
                    _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # CSV FORMATTED
            elif 'btn-nclicks-6' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), self.picked_data)}' \
                            f'.csv'
                self.picked_data.get_raw_result().to_csv(
                    _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # PDF
            elif 'btn-nclicks-2' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), self.picked_data)}' \
                            f'.pdf'
                self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # HTML
            elif 'btn-nclicks-3' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), self.picked_data)}' \
                            f'.html'
                self.fig.write_html(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # SVG
            elif 'btn-nclicks-5' in changed_id:
                file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), self.picked_data)}' \
                            f'.svg'
                self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            if time_val:
                self.interval = time_val

            z = self.picked_data.njit_result
            x = self.picked_data.x_1d
            y = self.picked_data.y_1d

            return self.fig.update_traces(z=z, x=x, y=y), maxx_interval, self.interval, disabled_btn, disabled_input

        return app.run_server(mode=self.mode, port=self.port)

    def save_all(self):

        for data in self.existing_data:
            # CSV
            file_name = f'{self._ts}raw_{self.__middle_name(datetime.now().strftime("_%H-%M-%S"), data)}' \
                        f'.csv'
            data.get_result().to_csv(
                _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # CSV FORMATTED
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"),data)}.csv'
            data.get_raw_result().to_csv(
                _save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type), index=False)

            # PDF
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"),data)}.pdf'
            self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # HTML
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"),data)}.html'
            self.fig.write_html(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))

            # SVG
            file_name = f'{self._ts}{self.__middle_name(datetime.now().strftime("_%H-%M-%S"),data)}.svg'
            self.fig.write_image(_save_path(file_name, self.chip_id, spectroscopy_type=self.spectroscopy_type))
