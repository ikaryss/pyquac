# built-in libraries
from time import perf_counter
from typing import Iterable, Union
import time
import random
# installable libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import numba as nb

# working with 2nd qubit
from scipy.optimize import curve_fit

# devices
import zhinst
import zhinst.ziPython
from drivers.M9290A import *
from drivers.N5183B import *
from drivers.k6220 import *
from drivers.znb_besedin import *
from resonator_tools import circlefit
from resonator_tools.circuit import notch_port

# family class
from pyquac.fmn_datatools import TwoToneSpectroscopy, timer, mrange


def _fit_cos(t, A1, A0, omega, teta):
    return A1 * np.cos(2 * np.pi * omega * t + teta) + A0


class Tts(TwoToneSpectroscopy):
    """
    Two Tone Spectroscopy for 1 qubit measurements
    """

    def __init__(self,
                 *,
                 fr_min: float, fr_max: float,
                 x_min=None, x_max=None, y_min=None, y_max=None,
                 nx_points=None, x_step=None,
                 y_step=None, ny_points=None,
                 x_arr=None, y_arr=None,
                 hdawg_host: str = '127.0.0.1', hdawg_port: int = 8004, hdawg_mode: int = 6,
                 hdawg_device: str = 'dev8210', hdawg_channel: int = 5,
                 LO_port: str = 'TCPIP0::192.168.180.143::hislip0::INSTR',
                 LO_res_port: str = 'TCPIP0::192.168.180.110::inst0::INSTR',
                 LO_set_power: int = 5,
                 LO_res_set_bandwidth: int = 20, LO_res_set_power: int = -10, LO_res_set_nop=101,
                 base_bandwidth=40, LO_res_set_averages=1, LO_res_meas_averages=1,
                 cs_mode=False,
                 cs_device: str = 'GPIB0::12::INSTR'
                 ):
        """
        Class provides methods for working with live data for Two Tone Spectroscopy
        :param x_arr: array of x values
        :param y_arr: array of y values
        :param x_min: x minimum value (int | float)
        :param x_max: x maximum value (int | float)
        :param y_min: y minimum value (int | float)
        :param y_max: y maximum value (int | float)
        :param nx_points: x count value (int)
        :param x_step: x step value (float)
        :param ny_points: y count value (int)
        :param y_step: y step value (float)
        :param fr_min: min frequency for find resonator
        :param fr_max: max frequency for find resonator
        :param hdawg_host: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_port: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_mode: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param LO_port: qubit LO = N5183B('N5183B', LO_port)
        :param LO_res_port: resonator LO_res = Znb(LO_res_port)
        :param hdawg_device: 'dev8210' by default
        :param hdawg_channel: hdawg.setInt('/' + hdawg_device + '/sigouts/' + str(hdawg_channel) + '/on', 1)
        :param LO_set_power: base LO power (default 5)
        :param LO_res_set_bandwidth: bandwidth during resonator tuning
        :param LO_res_set_power: base resonator LO power (default -10)
        :param LO_res_set_nop: number of points during resonator scanning (default 101)
        :param base_bandwidth: (int) bandwidth during measurements
        :param LO_res_set_averages: set averages for resonator LO parameter
        :param LO_res_meas_averages: measure averages for resonator LO parameter
        """

        super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min, y_max=y_max, ny_points=ny_points,
                         x_step=x_step, y_step=y_step, x_arr=x_arr, y_arr=y_arr)

        # if True - than use CS instead of HDAWG
        self.cs_mode = cs_mode

        self.fr_min = fr_min
        self.fr_max = fr_max
        self.hdawg_channel = hdawg_channel
        self.hdawg_device = hdawg_device

        # HDAWG or CS init
        if not self.cs_mode:
            self.hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
            self.hdawg_setDouble = '/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/offset'
            # open HDAWG ZI
            _ = self.hdawg.awgModule()
        else:
            self.cs = K6220(cs_device)

        # freq generator init
        self.LO = N5183B('N5183B', LO_port)  # qubit
        self.LO_res = Znb(LO_res_port)  # resonator

        # set base parameters
        self.LO_set_power = LO_set_power
        self.LO_res_set_nop = LO_res_set_nop
        self.LO_res_set_bandwidth = LO_res_set_bandwidth
        self.LO_res_set_power = LO_res_set_power
        self.base_bandwidth = base_bandwidth
        self.LO_res_set_averages = LO_res_set_averages
        self.LO_res.set_averages(LO_res_set_averages)
        self.LO_res_meas_averages = LO_res_meas_averages

        pass

    def spectroscopy_preset_on(self, ch_I, ch_Q):
        # readout
        self.LO_res.set_power(-10)
        self.LO_res.set_power_on()
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/0/offset', 0.6)
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/1/offset', 0.6)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 1)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 1)
        # qubit
        self.LO.set_status(0)
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/{ch_I}/offset', 0.4)
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/{ch_Q}/offset', 0.4)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{ch_I}/on', 1)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{ch_Q}/on', 1)

    def spectroscopy_preset_off(self, *channels):
        """
        turn off readout channels (0 and 1) and also turn of selected qubit channels
        :param channels: qubit channels to turn off
        """
        # readout
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 0)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 0)
        # qubit
        for channel in channels:
            self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{channel}/on', 0)

    @property
    def __find_resonator(self):

        self.LO.set_status(0)

        # prior bandwidth
        bandwidth = self.LO_res.get_bandwidth()
        x_lim = self.LO_res.get_freq_limits()

        self.LO_res.set_bandwidth(self.LO_res_set_bandwidth)
        self.LO_res.set_nop(self.LO_res_set_nop)
        self.LO_res.set_freq_limits(self.fr_min, self.fr_max)
        self.LO_res.set_averages(self.LO_res_set_averages)

        # measure S21
        freqs = self.LO_res.get_freqpoints()
        notch = notch_port(freqs, self.LO_res.measure()['S-parameter'])
        notch.autofit(electric_delay=60e-9)
        result = round(notch.fitresults['fr'])

        # Resetting to the next round of measurements
        self.LO_res.set_bandwidth(bandwidth)
        self.LO_res.set_freq_limits(*x_lim)
        self.LO_res.set_nop(1)
        self.LO.set_status(1)

        return result

    def run_measurements(self, *, x_key: Union[float, int, Iterable] = None, y_key: Union[float, int, Iterable] = None,
                         x_min=None, x_max=None, y_min=None, y_max=None,
                         sleep=0.007, timeout=None):

        # Turn on HDAWG or CS
        if not self.cs_mode:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/on', 1)
        else:
            self.cs.output_on()

        self.iter_setup(x_key=x_key, y_key=y_key,
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # Set power
        self.LO.set_power(self.LO_set_power)
        self.LO_res.set_power(self.LO_res_set_power)
        # base bandwidth
        self.LO_res.set_bandwidth(self.base_bandwidth)
        try:
            for i in range(len(self.load)):
                if (i == 0) or (self.load[i] != self.load[i - 1]):
                    self.LO_res.set_center(float(self.__find_resonator))

                    if timeout is not None:
                        timer.sleep(timeout)
                    else:
                        pass
                # measurement averages
                self.LO_res.set_averages(self.LO_res_meas_averages)

                if not self.cs_mode:
                    self.hdawg.setDouble(self.hdawg_setDouble, self.load[i])  # Voltage write
                else:
                    self.cs.write("CURRent {}".format(self.load[i]))  # Current write

                self.LO.set_frequency(self.frequency[i])  # Frequency write

                result = self.LO_res.measure()['S-parameter']

                self.write(x=self.load[i],
                           y=self.frequency[i],
                           heat=20 * np.log10(abs(result)[0])
                           )
                timer.sleep(sleep)

        except KeyboardInterrupt:
            pass

        # Turn off LO
        self.LO.set_status(0)

        # Turn off HDAWG or CS
        if not self.cs_mode:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/on', 0)
        else:
            self.cs.output_off()


class Sts(TwoToneSpectroscopy):
    """
    Single Tone Spectroscopy for 1 qubit measurements
    """

    def __init__(self,
                 *,
                 x_min=None, x_max=None, y_min=None, y_max=None,
                 nx_points=None, x_step=None, y_step=None, ny_points=None,
                 x_arr=None, y_arr=None,
                 hdawg_host: str = '127.0.0.1', hdawg_port: int = 8004, hdawg_mode: int = 6,
                 hdawg_channel: int = 5, hdawg_device: str = 'dev8210',
                 LO_res_port: str = 'TCPIP0::192.168.180.110::inst0::INSTR',
                 LO_res_set_bandwidth: int = 20, LO_res_set_power: int = -10,
                 LO_res_set_averages=1,
                 cs_mode=False,
                 cs_device: str = 'GPIB0::12::INSTR'
                 ):
        """
        :param x_arr: array of x values
        :param y_arr: array of y values
        :param x_min: x minimum value (int | float)
        :param x_max: x maximum value (int | float)
        :param y_min: y minimum value (int | float)
        :param y_max: y maximum value (int | float)
        :param x_step: x step value (float)
        :param nx_points: x count value (int)
        :param y_step: y step value (float)
        :param ny_points: y count value (int)
        :param hdawg_host: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_port: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_mode: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param LO_res_port: resonator LO_res = Znb(LO_res_port)
        :param hdawg_channel: hdawg.setInt('/' + hdawg_device + '/sigouts/' + str(hdawg_channel) + '/on', 1)
        :param hdawg_device: 'dev8210' by default
        :param LO_res_set_bandwidth: base bandwidth (default 20)
        :param LO_res_set_power: base LO_res power (default -10)
        :param LO_res_set_averages: set averages for resonator LO parameter
        """

        super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min, y_max=y_max, ny_points=ny_points,
                         x_step=x_step, y_step=y_step, x_arr=x_arr, y_arr=y_arr)

        # if True - than use CS instead of HDAWG
        self.cs_mode = cs_mode

        self.ny_points = ny_points if self.y_step is None else len(self.y_list)

        # HDAWG init
        self.hdawg_device = hdawg_device
        self.hdawg_channel = hdawg_channel

        # HDAWG or CS init
        if not self.cs_mode:
            self.hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
            self.hdawg_setDouble = '/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/offset'
            _ = self.hdawg.awgModule()
        else:
            self.cs = K6220(cs_device)

        # LO connect
        self.LO_res = Znb(LO_res_port)  # resonator

        # set base parameters of LO res
        self.LO_res_set_bandwidth = LO_res_set_bandwidth
        self.LO_res_set_power = LO_res_set_power

        self.LO_res_set_averages = LO_res_set_averages
        self.LO_res.set_averages(LO_res_set_averages)

        pass

    def run_measurements(self, *, sleep=0.0007):

        self.iter_setup(x_key=None, y_key=None,
                        x_min=None, x_max=None, y_min=None, y_max=None)

        # enable channel
        if not self.cs_mode:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/on', 1)
        else:
            self.cs.output_on()

        # Set power
        self.LO_res.set_power(self.LO_res_set_power)

        # Set LO parameters
        self.LO_res.set_bandwidth(self.LO_res_set_bandwidth)
        self.LO_res.set_nop(self.ny_points)
        self.LO_res.set_freq_limits(self.y_min, self.y_max)

        freq_len = len(self.y_list)
        try:
            for i in range(len(self.load)):
                if (i == 0) or (self.load[i] != self.load[i - 1]):

                    if not self.cs_mode:
                        self.hdawg.setDouble(self.hdawg_setDouble, self.load[i])  # Voltage write
                    else:
                        self.cs.write("CURRent {}".format(self.load[i]))  # Current write

                    self.LO_res.set_averages(self.LO_res_set_averages)
                    result = self.LO_res.measure()['S-parameter']

                    for j in range(freq_len):
                        self.write(x=self.load[i],
                                   y=self.y_list[j],
                                   heat=20 * np.log10(abs(result[j]))
                                   )

                timer.sleep(sleep)

        except KeyboardInterrupt:
            if (self.x_raw[-1] == self.x_list[-1]) and (self.y_raw[-1] == self.y_list[-1]):
                pass
            else:
                # drop the last column if interrupt for stable data saving
                self.drop(x=self.x_raw[-1])
            pass

        # Turn off HDAWG or CS
        if not self.cs_mode:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.hdawg_channel) + '/on', 0)
        else:
            self.cs.output_off()


class Sts2Q(TwoToneSpectroscopy):
    """
    Single Tone Spectroscopy for 2 qubit measurements
    """

    def __init__(self,
                 *,
                 flux_qubit: str, readout_qubit: str,
                 x_min=None, x_max=None, y_min=None, y_max=None,
                 nx_points=None, x_step=None, y_step=None, ny_points=None,
                 x_min_target=None, x_max_target=None, x_step_target=None, nx_points_target=None,
                 x_min_control=None, x_max_control=None, x_step_control=None, nx_points_control=None,

                 x_arr=None, y_arr=None, x_arr_target=None, x_arr_control=None,

                 Q1_ch: int = None,
                 Q2_ch: int = None,
                 Coupler_ch: int = None,

                 hdawg_host: str = '127.0.0.1', hdawg_port: int = 8004, hdawg_mode: int = 6,
                 hdawg_device: str = 'dev8210',
                 hdawg_device_coupler: str = 'dev8307',

                 LO_res_port: str = 'TCPIP0::192.168.180.110::inst0::INSTR',
                 LO_res_set_bandwidth: int = 20, LO_res_set_power: int = -10,
                 LO_res_set_averages=1,
                 compensation_matrix=None
                 ):
        """

        :param flux_qubit: str|list of strings. names of qubits that we drive
        :param readout_qubit: str. name of readout qubit
        :param x_arr: array of x values
        :param y_arr: array of y values
        :param x_arr_coupler: array of x values for coupler
        :param x_arr_control: array of x values for control
        :param x_min: min x value of target qubit or base x_min while checking
        :param x_max: max x value of target qubit or base x_max while checking
        :param y_min: min y value of readout qubit
        :param y_max: max y value of readout qubit
        :param nx_points: nx_points value of target qubit or base hdawg_port nx_points while checking
        :param x_step: x_step value of target qubit or base x_step while checking
        :param y_step: y_step value of readout qubit
        :param ny_points: ny_points value of readout qubit
        :param x_min_coupler: min x value of coupler qubit
        :param x_max_coupler: max x value of coupler qubit
        :param x_step_coupler: x_step value of coupler qubit
        :param nx_points_coupler: nx_points value of coupler qubit
        :param x_min_control: min x value of control qubit
        :param x_max_control: max x value of control qubit
        :param x_step_control: x_step value of control qubit
        :param nx_points_control: nx_points value of control qubit
        :param Q1_ch: target channel
        :param Q2_ch: control channel
        :param Coupler_ch: coupler channel
        :param hdawg_host: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_port: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_mode: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param LO_res_port: resonator LO_res = Znb(LO_res_port)
        :param hdawg_channel: hdawg.setInt('/' + hdawg_device + '/sigouts/' + str(hdawg_channel) + '/on', 1)
        :param hdawg_device: 'dev8210' by default
        :param LO_res_set_bandwidth: base bandwidth (default 20)
        :param LO_res_set_power: base LO_res power (default -10)
        :param LO_res_set_averages: set averages for resonator LO parameter
        """
        self.target_STS = None
        self.control_STS = None

        "configuring arrays for compensation"
        if compensation_matrix is not None:
            if x_arr is None:
                if nx_points is not None:
                    _, x_step = np.linspace(x_min, x_max, nx_points, retstep=True)
                    x_step = round(x_step, 10)
                else:
                    x_step = float(x_step)

                __x_step_DP = len(str(x_step).split(".")[1])
                x_list = mrange.orange(x_min, x_max + x_step, x_step)
                x_list = np.around(x_list, decimals=__x_step_DP)
                x_lists = np.array([x_list, x_list, x_list])

                new_x_matrix = np.dot(compensation_matrix, x_lists)
                target_x_list = new_x_matrix[0]
                control_x_list = new_x_matrix[1]
            else:
                x_arr = np.array(x_arr)
                x_lists = np.array([x_arr, x_arr, x_arr])
                new_x_matrix = np.dot(compensation_matrix, x_lists)
                target_x_list = new_x_matrix[0]
                control_x_list = new_x_matrix[1]
        else:
            pass

        if compensation_matrix is None:
            "if there is no compensation matrix and differ voltage arrays"
            super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min, y_max=y_max,
                             ny_points=ny_points,
                             x_step=x_step, y_step=y_step, x_arr=x_arr, y_arr=y_arr)
            self.ny_points = ny_points if self.y_step is None else len(self.y_list)

            "for differ voltage arrays - create different TwoToneSpectroscopy classes"
            if (x_min_target is not None) or (x_arr_target is not None):
                self.target_STS = TwoToneSpectroscopy(x_min=x_min_target, x_max=x_max_target,
                                                      nx_points=nx_points_target, y_min=y_min,
                                                      y_max=y_max,
                                                      ny_points=ny_points,
                                                      x_step=x_step_target, y_step=y_step, x_arr=x_arr_target,
                                                      y_arr=y_arr)
            if (x_min_control is not None) or (x_arr_control is not None):
                self.control_STS = TwoToneSpectroscopy(x_min=x_min_control, x_max=x_max_control,
                                                       nx_points=nx_points_control, y_min=y_min, y_max=y_max,
                                                       ny_points=ny_points,
                                                       x_step=x_step_control, y_step=y_step, x_arr=x_arr_control,
                                                       y_arr=y_arr)
        else:
            "if compensation matrix - create different TwoToneSpectroscopy classes"
            self.target_STS = TwoToneSpectroscopy(y_min=y_min, y_max=y_max, ny_points=ny_points, y_step=y_step,
                                                  y_arr=y_arr, x_arr=target_x_list)

            self.control_STS = TwoToneSpectroscopy(y_min=y_min, y_max=y_max, ny_points=ny_points, y_step=y_step,
                                                   y_arr=y_arr, x_arr=control_x_list)

            super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min,
                             y_max=y_max, ny_points=ny_points, x_step=x_step, y_step=y_step,
                             x_arr=x_arr, y_arr=y_arr)
            self.ny_points = ny_points if self.y_step is None else len(self.y_list)

        # define channels
        self.target_ch = Q1_ch
        self.coupler_ch = Coupler_ch
        self.control_ch = Q2_ch

        # HDAWG init for Q1 Q2
        self.hdawg_device = hdawg_device
        self.hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        _ = self.hdawg.awgModule()

        # HDAWG init for coupler
        self.hdawg_deviceC = hdawg_device_coupler

        # LO connect
        self.LO_res = Znb(LO_res_port)  # resonator

        # set base parameters of LO res
        self.LO_res_set_bandwidth = LO_res_set_bandwidth
        self.LO_res_set_power = LO_res_set_power

        self.LO_res_set_averages = LO_res_set_averages
        self.LO_res.set_averages(LO_res_set_averages)

        self.flux_qubit = flux_qubit
        self.readout_qubit = readout_qubit

        self.finished = False
        self.active = False
        self.start_for_fit_LP = (6, 0.02, 0.1, 0)
        self.start_for_fit_HP = (10, 0.2, 0.5, 0)

        self.fit = None
        self.x_offset = None
        self.x_offset_Q1_Q2, self.x_offset_Q1_C = None, None
        self.x_offset_Q2_Q1, self.x_offset_Q2_C = None, None
        pass

    def spectroscopy_preset_on(self, *channels):
        """
        turn on readout channels (0 and 1) and also turn on selected qubit channels
        :param channels: qubit channels to turn off
        """
        # readout
        self.LO_res.set_power(-10)
        self.LO_res.set_power_on()

        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/0/offset', 0.6)
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/1/offset', 0.6)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 1)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 1)
        # qubit
        for channel in channels:
            self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/{channel}/offset', 0.4)
            self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{channel}/on', 1)

    def spectroscopy_preset_off(self, *channels):
        """
        turn off readout channels (0 and 1) and also turn of selected qubit channels
        :param channels: qubit channels to turn off
        """
        # readout
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 0)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 0)
        # qubit
        for channel in channels:
            self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{channel}/on', 0)

    def run_measurements(self, *, sleep=0.0007):

        self.iter_setup(x_key=None, y_key=None,
                        x_min=None, x_max=None, y_min=None, y_max=None)

        if self.target_STS is not None:
            self.target_STS.iter_setup(x_key=None, y_key=None,
                                       x_min=None, x_max=None, y_min=None, y_max=None)
        if self.control_STS is not None:
            self.control_STS.iter_setup(x_key=None, y_key=None,
                                        x_min=None, x_max=None, y_min=None, y_max=None)

        self.active = True

        # channel switch on
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 1)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 1)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 1)

        # Set power
        self.LO_res.set_power(self.LO_res_set_power)

        # Set LO parameters
        self.LO_res.set_bandwidth(self.LO_res_set_bandwidth)
        self.LO_res.set_nop(self.ny_points)
        self.LO_res.set_freq_limits(self.y_min, self.y_max)

        freq_len = len(self.y_list)
        try:
            for i in range(len(self.load)):
                if (i == 0) or (self.load[i] != self.load[i - 1]):
                    "target qubit"
                    if (self.target_ch is not None) and (self.target_STS is None):
                        self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                             + str(self.target_ch) + '/offset', self.load[i])
                    elif (self.target_ch is not None) and (self.target_STS is not None):
                        self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                             + str(self.target_ch) + '/offset', self.target_STS.load[i])

                    "coupler qubit"
                    if self.coupler_ch is not None:
                        self.hdawg.setDouble('/' + self.hdawg_deviceC + '/sigouts/'
                                              + str(self.coupler_ch) + '/offset', self.load[i])

                    "control qubit"
                    if (self.control_ch is not None) and (self.control_STS is None):
                        self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                             + str(self.control_ch) + '/offset', self.load[i])
                    elif (self.control_ch is not None) and (self.control_STS is not None):
                        self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                             + str(self.target_ch) + '/offset', self.control_STS.load[i])

                    self.LO_res.set_averages(self.LO_res_set_averages)
                    result = self.LO_res.measure()['S-parameter']

                    for j in range(freq_len):
                        self.write(x=self.load[i],
                                   y=self.y_list[j],
                                   heat=20 * np.log10(abs(result[j]))
                                   )
                timer.sleep(sleep)

        except KeyboardInterrupt:
            if (self.x_raw[-1] == self.x_list[-1]) and (self.y_raw[-1] == self.y_list[-1]):
                pass
            else:
                # drop the last column if interrupt for stable data saving
                self.drop(x=self.x_raw[-1])
            self.active = False
            pass

        if (self.x_raw[-1] == self.x_list[-1]) and (self.y_raw[-1] == self.y_list[-1]):
            self.finished = True
        else:
            pass

        # channel switch off
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 0)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 0)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 0)

        self.active = False

    def fit_and_offset(self, start, fit_nop: int,
                       freq_for_offset=None, voltage_boundary=[0, 1]):

        X = self.raw_frame.copy()
        x_set = np.unique(X['x_value'].values)  # get an array of unique values of x

        glob_samp, glob_h_mi, glob_h_ma = self._TwoToneSpectroscopy__define_heat_sample_on(x_set,
                                                                                           info=False, plot=False,
                                                                                           q_max=95,
                                                                                           q_min=5)
        self.glob_samp = glob_samp
        self.glob_h_mi = glob_h_mi
        self.glob_h_ma = glob_h_ma

        tuple_list = ()
        for xx in x_set:
            y_min, y_max = self._TwoToneSpectroscopy__find_min_max_y_on_(xx, glob_samp)

            temp_y_idx = (X[(X.x_value == xx) & (X.y_value >= y_min) & (X.y_value <= y_max)
                            ].heat_value - glob_samp).abs().idxmin()
            temp_max_row = tuple(X.iloc[temp_y_idx])
            tuple_list += (temp_max_row,)

        # rotating array
        # !!!!
        # WHILE DEBUGING REURN X Y FOR CHECKING HOW FIND MIN MAX Y WORKS
        # !!!!
        tuple_of_max_z_values = np.array(tuple_list).T
        x = tuple_of_max_z_values[0]
        y = tuple_of_max_z_values[1]

        # fitting cosine
        fitted_params, cov = curve_fit(_fit_cos, x, y, start)
        A1, A0, omega, teta = fitted_params

        # generating curve
        x_list = np.linspace(self.x_min, self.x_max, fit_nop)
        fit = A1 * np.cos(2 * np.pi * omega * x_list + teta) + A0
        fit_dict = dict(x=x_list, y=fit)

        # offset
        if freq_for_offset is not None:
            volt = x_list[(x_list >= voltage_boundary[0]) & (x_list <= voltage_boundary[1])]
            delta_f = abs(fit - freq_for_offset)[(x_list >= voltage_boundary[0]) & (x_list <= voltage_boundary[1])]
            offset_idx = np.argmin(delta_f)
            x_offset = volt[offset_idx]
        else:
            fit = fit[(x_list >= voltage_boundary[0]) & (x_list <= voltage_boundary[1])]
            x_list = x_list[(x_list >= voltage_boundary[0]) & (x_list <= voltage_boundary[1])]
            x_offset = x_list[np.argmax(fit)]

        self.fit = fit
        self.x_offset = x_offset

        return fit_dict, x_offset, fitted_params


class Sts2QContainer:

    def __init__(self, *,
                 Q1_ch: int,
                 Q2_ch: int,
                 Coupler_ch: int,

                 y_minQ1=None, y_maxQ1=None, y_minQ2=None, y_maxQ2=None,

                 x_min=None, x_max=None, nx_points=None, x_step=None,
                 x_arr=None, y_arr=None,

                 x_minQ1=None, x_maxQ1=None, nx_pointsQ1=None, x_stepQ1=None,
                 ny_pointsQ1=None, y_stepQ1=None,
                 x_arrQ1=None, y_arrQ1=None,

                 x_minC=None, x_maxC=None, nx_pointsC=None, x_stepC=None,
                 x_arrC=None,

                 x_minQ2=None, x_maxQ2=None, nx_pointsQ2=None, x_stepQ2=None,
                 ny_pointsQ2=None, y_stepQ2=None,
                 x_arrQ2=None, y_arrQ2=None,

                 hdawg_host: str = '127.0.0.1', hdawg_port: int = 8004, hdawg_mode: int = 6,
                 hdawg_device: str = 'dev8210',
                 hdawg_device_coupler: str = 'dev8307',

                 LO_res_port: str = 'TCPIP0::192.168.180.110::inst0::INSTR',
                 LO_res_set_bandwidth: int = 20, LO_res_set_power: int = -10,
                 LO_res_set_averages=1
                 ):

        if (x_min is not None) and (x_max is not None):

            x_minQ1 = x_min if x_minQ1 is None else x_minQ1
            x_minQ2 = x_min if x_minQ2 is None else x_minQ2
            x_minC = x_min if x_minC is None else x_minC

            x_maxQ1 = x_max if x_maxQ1 is None else x_maxQ1
            x_maxQ2 = x_max if x_maxQ2 is None else x_maxQ2
            x_maxC = x_max if x_maxC is None else x_maxC

            nx_pointsQ1 = nx_points if nx_pointsQ1 is None else nx_pointsQ1
            nx_pointsQ2 = nx_points if nx_pointsQ2 is None else nx_pointsQ2
            nx_pointsC = nx_points if nx_pointsC is None else nx_pointsC

            x_stepQ1 = x_step if x_stepQ1 is None else x_stepQ1
            x_stepQ2 = x_step if x_stepQ2 is None else x_stepQ2
            x_stepC = x_step if x_stepC is None else x_stepC

        elif x_arr is not None:
            x_arrQ1 = np.array(x_arr)
            x_arrQ2 = np.array(x_arr)
            x_arrC = np.array(x_arr)
        else:
            pass

        if y_arr is not None:
            y_arrQ1 = np.array(y_arr)
            y_arrQ2 = np.array(y_arr)

        "Flux on Q1 | Coupler | Q2; Readout - Q1"
        self.Q1 = Sts2Q(flux_qubit='Q1', readout_qubit='Q1', x_min=x_minQ1, x_max=x_maxQ1, y_min=y_minQ1, y_max=y_maxQ1,
                        nx_points=nx_pointsQ1, x_step=x_stepQ1, y_step=y_stepQ1, ny_points=ny_pointsQ1, x_arr=x_arrQ1,
                        y_arr=y_arrQ1, Q1_ch=Q1_ch, hdawg_host=hdawg_host, hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                        hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                        LO_res_port=LO_res_port,
                        LO_res_set_bandwidth=LO_res_set_bandwidth, LO_res_set_power=LO_res_set_power,
                        LO_res_set_averages=LO_res_set_averages)

        self.Q1_Coupler = Sts2Q(flux_qubit='Coupler', readout_qubit='Q1', x_min=x_minC, x_max=x_maxC, y_min=y_minQ1,
                                y_max=y_maxQ1, nx_points=nx_pointsC, x_step=x_stepC, y_step=y_stepQ1,
                                ny_points=ny_pointsQ1, x_arr=x_arrC, y_arr=y_arrQ1, Coupler_ch=Coupler_ch,
                                hdawg_host=hdawg_host, hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                                hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                                LO_res_port=LO_res_port,
                                LO_res_set_bandwidth=LO_res_set_bandwidth, LO_res_set_power=LO_res_set_power,
                                LO_res_set_averages=LO_res_set_averages)

        self.Q1_Q2 = Sts2Q(flux_qubit='Q2', readout_qubit='Q1', x_min=x_minQ2, x_max=x_maxQ2, y_min=y_minQ1,
                           y_max=y_maxQ1, nx_points=nx_pointsQ2, x_step=x_stepQ2, y_step=y_stepQ1,
                           ny_points=ny_pointsQ1, x_arr=x_arrQ2, y_arr=y_arrQ1, Q2_ch=Q2_ch, hdawg_host=hdawg_host,
                           hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                           hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                           LO_res_port=LO_res_port, LO_res_set_bandwidth=LO_res_set_bandwidth,
                           LO_res_set_power=LO_res_set_power, LO_res_set_averages=LO_res_set_averages)

        "Flux on Q1 | Coupler | Q2; Readout - Q2"
        self.Q2 = Sts2Q(flux_qubit='Q2', readout_qubit='Q2', x_min=x_minQ2, x_max=x_maxQ2, y_min=y_minQ2, y_max=y_maxQ2,
                        nx_points=nx_pointsQ2, x_step=x_stepQ2, y_step=y_stepQ2, ny_points=ny_pointsQ2, x_arr=x_arrQ2,
                        y_arr=y_arrQ2, Q1_ch=Q2_ch, hdawg_host=hdawg_host, hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                        hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                        LO_res_port=LO_res_port,
                        LO_res_set_bandwidth=LO_res_set_bandwidth, LO_res_set_power=LO_res_set_power,
                        LO_res_set_averages=LO_res_set_averages)

        self.Q2_Coupler = Sts2Q(flux_qubit='Coupler', readout_qubit='Q2', x_min=x_minC, x_max=x_maxC, y_min=y_minQ2,
                                y_max=y_maxQ2, nx_points=nx_pointsC, x_step=x_stepC, y_step=y_stepQ2,
                                ny_points=ny_pointsQ2, x_arr=x_arrC, y_arr=y_arrQ2, Coupler_ch=Coupler_ch,
                                hdawg_host=hdawg_host, hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                                hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                                LO_res_port=LO_res_port,
                                LO_res_set_bandwidth=LO_res_set_bandwidth, LO_res_set_power=LO_res_set_power,
                                LO_res_set_averages=LO_res_set_averages)

        self.Q2_Q1 = Sts2Q(flux_qubit='Q1', readout_qubit='Q2', x_min=x_minQ1, x_max=x_maxQ1, y_min=y_minQ2,
                           y_max=y_maxQ2, nx_points=nx_pointsQ1, x_step=x_stepQ1, y_step=y_stepQ2,
                           ny_points=ny_pointsQ2, x_arr=x_arrQ1, y_arr=y_arrQ2, Q2_ch=Q1_ch, hdawg_host=hdawg_host,
                           hdawg_port=hdawg_port, hdawg_mode=hdawg_mode,
                           hdawg_device=hdawg_device, hdawg_device_coupler=hdawg_device_coupler,
                           LO_res_port=LO_res_port, LO_res_set_bandwidth=LO_res_set_bandwidth,
                           LO_res_set_power=LO_res_set_power, LO_res_set_averages=LO_res_set_averages)

        pass

    @property
    def final_matrix_old(self):

        # First row
        Q1_offset_base = self.Q1.x_offset if self.Q1.finished else None
        a11 = 1 if self.Q1.finished else None

        Q2_offset = self.Q1_Q2.x_offset if self.Q1_Q2.finished else None
        a12 = Q1_offset_base / Q2_offset if self.Q1_Q2.finished and self.Q1.finished else None

        Q1_coupler_offset = self.Q1_Coupler.x_offset if self.Q1_Coupler.finished else None
        a13 = Q1_offset_base / Q1_coupler_offset if self.Q1_Coupler.finished and self.Q1.finished else None

        # Second row
        Q2_offset_base = self.Q2.x_offset if self.Q2.finished else None
        Q1_offset = self.Q2_Q1.x_offset if self.Q2_Q1.finished else None
        a21 = Q2_offset_base / Q1_offset if self.Q2_Q1.finished and self.Q2.finished else None

        a22 = 1 if self.Q2.finished else None

        Q2_coupler_offset = self.Q2_Coupler.x_offset if self.Q2_Coupler.finished else None
        a23 = Q2_offset_base / Q2_coupler_offset if self.Q2_Coupler.finished and self.Q2.finished else None

        # Third row
        a31 = 0
        a32 = 0
        a33 = 1

        matrix = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        return np.linalg.inv(matrix)

    @property
    def final_matrix(self):

        # First row
        last_point_Q1 = self.Q1.x_list[-1] if self.Q1.finished else None
        a11 = 1 if self.Q1.finished else None

        Q1_Q2_offset = self.Q1.x_offset_Q1_Q2 if self.Q1_Q2.finished else None
        a12 = Q1_Q2_offset / last_point_Q1 if self.Q1_Q2.finished and self.Q1.finished else None

        Q1_coupler_offset = self.Q1.x_offset_Q1_C if self.Q1_Coupler.finished else None
        a13 = Q1_coupler_offset / last_point_Q1 if self.Q1_Coupler.finished and self.Q1.finished else None

        # Second row
        last_point_Q2 = self.Q2.x_list[-1] if self.Q2.finished else None

        Q2_Q1_offset = self.Q2.x_offset_Q2_Q1 if self.Q2_Q1.finished else None
        a21 = Q2_Q1_offset / last_point_Q2 if self.Q2_Q1.finished and self.Q2.finished else None

        a22 = 1 if self.Q2.finished else None

        Q2_coupler_offset = self.Q2.x_offset_Q2_C if self.Q2_Coupler.finished else None
        a23 = Q2_coupler_offset / last_point_Q2 if self.Q2_Coupler.finished and self.Q2.finished else None

        # Third row
        a31 = 0
        a32 = 0
        a33 = 1

        matrix = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        return matrix, np.linalg.inv(matrix)


class Tts2Q(TwoToneSpectroscopy):
    """
    Two Tone Spectroscopy for 2 qubit measurements
    """

    def __init__(self,
                 *,
                 flux_qubit: str, readout_qubit: str, readout_q: int,
                 fr_min: float, fr_max: float,
                 x_min=None, x_max=None, y_min=None, y_max=None,
                 nx_points=None, x_step=None,
                 y_step=None, ny_points=None,
                 x_min_target=None, x_max_target=None, x_step_target=None, nx_points_target=None,
                 x_min_control=None, x_max_control=None, x_step_control=None, nx_points_control=None,

                 x_arr=None, y_arr=None, x_arr_target=None, x_arr_control=None,

                 Q1_ch: int = None,
                 Q2_ch: int = None,
                 Coupler_ch: int = None,

                 hdawg_host: str = '127.0.0.1', hdawg_port: int = 8004, hdawg_mode: int = 6,
                 hdawg_device: str = 'dev8210',
                 hdawg_device_coupler: str = 'dev8307',

                 LO_port: str = 'TCPIP0::192.168.180.143::hislip0::INSTR',
                 LO_res_port: str = 'TCPIP0::192.168.180.110::inst0::INSTR',
                 LO_set_power: int = 5,
                 LO_res_set_bandwidth: int = 20, LO_res_set_power: int = -10, LO_res_set_nop=101,
                 base_bandwidth=40, LO_res_set_averages=1, LO_res_meas_averages=1,
                 compensation_matrix=None
                 ):
        """
        Class provides methods for working with live data for Two Tone Spectroscopy
        :param x_arr: array of x values
        :param y_arr: array of y values
        :param x_min: x minimum value (int | float)
        :param x_max: x maximum value (int | float)
        :param y_min: y minimum value (int | float)
        :param y_max: y maximum value (int | float)
        :param nx_points: x count value (int)
        :param x_step: x step value (float)
        :param ny_points: y count value (int)
        :param y_step: y step value (float)
        :param fr_min: min frequency for find resonator
        :param fr_max: max frequency for find resonator
        :param hdawg_host: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_port: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param hdawg_mode: hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        :param LO_port: qubit LO = N5183B('N5183B', LO_port)
        :param LO_res_port: resonator LO_res = Znb(LO_res_port)
        :param hdawg_device: 'dev8210' by default
        :param hdawg_channel: hdawg.setInt('/' + hdawg_device + '/sigouts/' + str(hdawg_channel) + '/on', 1)
        :param LO_set_power: base LO power (default 5)
        :param LO_res_set_bandwidth: bandwidth during resonator tuning
        :param LO_res_set_power: base resonator LO power (default -10)
        :param LO_res_set_nop: number of points during resonator scanning (default 101)
        :param base_bandwidth: (int) bandwidth during measurements
        :param LO_res_set_averages: set averages for resonator LO parameter
        :param LO_res_meas_averages: measure averages for resonator LO parameter
        """
        self.Q1_STS = None
        self.Q2_STS = None

        self.fr_min = fr_min
        self.fr_max = fr_max

        "configuring arrays for compensation"
        if compensation_matrix is not None:
            if x_arr is None:
                if nx_points is not None:
                    _, x_step = np.linspace(x_min, x_max, nx_points, retstep=True)
                    x_step = round(x_step, 10)
                else:
                    x_step = float(x_step)

                __x_step_DP = len(str(x_step).split(".")[1])
                x_list = mrange.orange(x_min, x_max + x_step, x_step)
                x_list = np.around(x_list, decimals=__x_step_DP)

                zeros = np.zeros(len(x_list))
                if readout_q == 1:
                    x_lists = np.array([zeros, x_list, x_list])
                elif readout_q == 2:
                    x_lists = np.array([x_list, zeros, x_list])

                new_x_matrix = np.dot(compensation_matrix, x_lists)
                target_x_list = new_x_matrix[0]
                control_x_list = new_x_matrix[1]
            else:
                x_arr = np.array(x_arr)

                zeros = np.zeros(len(x_arr))
                if readout_q == 1:
                    x_lists = np.array([zeros, x_arr, x_arr])
                elif readout_q == 2:
                    x_lists = np.array([x_arr, zeros, x_arr])

                new_x_matrix = np.dot(compensation_matrix, x_lists)
                target_x_list = new_x_matrix[0]
                control_x_list = new_x_matrix[1]
        else:
            pass

        if compensation_matrix is None:
            "if there is no compensation matrix and differ voltage arrays"
            super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min, y_max=y_max,
                             ny_points=ny_points,
                             x_step=x_step, y_step=y_step, x_arr=x_arr, y_arr=y_arr)
            self.ny_points = ny_points if self.y_step is None else len(self.y_list)

            "for differ voltage arrays - create different TwoToneSpectroscopy classes"
            if (x_min_target is not None) or (x_arr_target is not None):
                self.Q1_STS = TwoToneSpectroscopy(x_min=x_min_target, x_max=x_max_target,
                                                  nx_points=nx_points_target, y_min=y_min,
                                                  y_max=y_max,
                                                  ny_points=ny_points,
                                                  x_step=x_step_target, y_step=y_step, x_arr=x_arr_target,
                                                  y_arr=y_arr)
            if (x_min_control is not None) or (x_arr_control is not None):
                self.Q2_STS = TwoToneSpectroscopy(x_min=x_min_control, x_max=x_max_control,
                                                  nx_points=nx_points_control, y_min=y_min, y_max=y_max,
                                                  ny_points=ny_points,
                                                  x_step=x_step_control, y_step=y_step, x_arr=x_arr_control,
                                                  y_arr=y_arr)
        else:
            "if compensation matrix - create different TwoToneSpectroscopy classes"
            self.Q1_STS = TwoToneSpectroscopy(y_min=y_min, y_max=y_max, ny_points=ny_points, y_step=y_step,
                                              y_arr=y_arr, x_arr=target_x_list)

            self.Q2_STS = TwoToneSpectroscopy(y_min=y_min, y_max=y_max, ny_points=ny_points, y_step=y_step,
                                              y_arr=y_arr, x_arr=control_x_list)

            super().__init__(x_min=x_min, x_max=x_max, nx_points=nx_points, y_min=y_min,
                             y_max=y_max, ny_points=ny_points, x_step=x_step, y_step=y_step,
                             x_arr=x_arr, y_arr=y_arr)
            self.ny_points = ny_points if self.y_step is None else len(self.y_list)

        # define channels
        self.target_ch = Q1_ch
        self.coupler_ch = Coupler_ch
        self.control_ch = Q2_ch

        # HDAWG init for Q1 Q2
        self.hdawg_device = hdawg_device
        self.hdawg = zhinst.ziPython.ziDAQServer(hdawg_host, hdawg_port, hdawg_mode)
        _ = self.hdawg.awgModule()

        # HDAWG init for coupler
        self.hdawg_deviceC = hdawg_device_coupler

        # freq generator init
        self.LO = N5183B('N5183B', LO_port)  # qubit
        self.LO_res = Znb(LO_res_port)  # resonator

        # set base parameters
        self.LO_set_power = LO_set_power
        self.LO_res_set_nop = LO_res_set_nop
        self.LO_res_set_bandwidth = LO_res_set_bandwidth
        self.LO_res_set_power = LO_res_set_power
        self.base_bandwidth = base_bandwidth
        self.LO_res_set_averages = LO_res_set_averages
        self.LO_res.set_averages(LO_res_set_averages)
        self.LO_res_meas_averages = LO_res_meas_averages

        self.flux_qubit = flux_qubit
        self.readout_qubit = readout_qubit
        pass

    def run_measurements(self, *, x_key: Union[float, int, Iterable] = None, y_key: Union[float, int, Iterable] = None,
                         x_min=None, x_max=None, y_min=None, y_max=None,
                         sleep=0.007, timeout=None):

        self.iter_setup(x_key=x_key, y_key=y_key,
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        if self.Q1_STS is not None:
            self.Q1_STS.iter_setup(x_key=None, y_key=None,
                                   x_min=None, x_max=None, y_min=None, y_max=None)
        if self.Q2_STS is not None:
            self.Q2_STS.iter_setup(x_key=None, y_key=None,
                                   x_min=None, x_max=None, y_min=None, y_max=None)

        # channel switch on
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 1)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 1)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 1)

        # Set power
        self.LO.set_power(self.LO_set_power)
        self.LO_res.set_power(self.LO_res_set_power)
        # base bandwidth
        self.LO_res.set_bandwidth(self.base_bandwidth)
        try:
            for i in range(len(self.load)):
                if (i == 0) or (self.load[i] != self.load[i - 1]):
                    self.LO_res.set_center(float(self.__find_resonator))

                    if timeout is not None:
                        timer.sleep(timeout)
                    else:
                        pass
                # measurement averages
                self.LO_res.set_averages(self.LO_res_meas_averages)

                if (self.target_ch is not None) and (self.Q1_STS is None):
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.target_ch) + '/offset', self.load[i])
                elif (self.target_ch is not None) and (self.Q1_STS is not None):
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.target_ch) + '/offset', self.Q1_STS.load[i])

                if self.coupler_ch is not None:
                    self.hdawg.setDouble('/' + self.hdawg_deviceC + '/sigouts/'
                                          + str(self.coupler_ch) + '/offset', self.load[i])

                if (self.control_ch is not None) and (self.Q2_STS is None):
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.control_ch) + '/offset', self.load[i])
                elif (self.control_ch is not None) and (self.Q2_STS is not None):
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.control_ch) + '/offset', self.Q2_STS.load[i])

                self.LO.set_frequency(self.frequency[i])  # Frequency write

                result = self.LO_res.measure()['S-parameter']

                self.write(x=self.load[i],
                           y=self.frequency[i],
                           heat=20 * np.log10(abs(result)[0])
                           )
                timer.sleep(sleep)

        except KeyboardInterrupt:
            pass

        # Turn off LO
        self.LO.set_status(0)

        # channel switch off
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 0)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 0)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 0)

    def run_compensation_measurements(self, *, x_key: Union[float, int, Iterable] = None,
                                      y_key: Union[float, int, Iterable] = None,
                                      x_min=None, x_max=None, y_min=None, y_max=None,
                                      sleep=0.007, timeout=None):

        self.iter_setup(x_key=x_key, y_key=y_key,
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        self.Q1_STS.iter_setup(x_key=x_key, y_key=y_key,
                               x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        self.Q2_STS.iter_setup(x_key=x_key, y_key=y_key,
                               x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # channel switch on
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 1)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 1)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 1)

        # Set power
        self.LO.set_power(self.LO_set_power)
        self.LO_res.set_power(self.LO_res_set_power)
        # base bandwidth
        self.LO_res.set_bandwidth(self.base_bandwidth)
        try:
            for i in range(len(self.load)):
                if (i == 0) or (self.load[i] != self.load[i - 1]):
                    self.LO_res.set_center(float(self.__find_resonator))

                    if timeout is not None:
                        timer.sleep(timeout)
                    else:
                        pass
                # measurement averages
                self.LO_res.set_averages(self.LO_res_meas_averages)

                if self.target_ch is not None:
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.target_ch) + '/offset', self.Q1_STS.load[i])

                if self.coupler_ch is not None:
                    self.hdawg.setDouble('/' + self.hdawg_deviceC + '/sigouts/'
                                          + str(self.coupler_ch) + '/offset', self.load[i])

                if self.control_ch is not None:
                    self.hdawg.setDouble('/' + self.hdawg_device + '/sigouts/'
                                         + str(self.control_ch) + '/offset', self.Q2_STS.load[i])

                self.LO.set_frequency(self.frequency[i])  # Frequency write

                result = self.LO_res.measure()['S-parameter']

                self.write(x=self.load[i],
                           y=self.frequency[i],
                           heat=20 * np.log10(abs(result)[0])
                           )
                timer.sleep(sleep)

        except KeyboardInterrupt:
            pass

        # Turn off LO
        self.LO.set_status(0)

        # channel switch off
        if self.target_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.target_ch) + '/on', 0)

        if self.coupler_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_deviceC + '/sigouts/' + str(self.coupler_ch) + '/on', 0)

        if self.control_ch is not None:
            self.hdawg.setInt('/' + self.hdawg_device + '/sigouts/' + str(self.control_ch) + '/on', 0)

    def spectroscopy_preset_on(self, *channels):
        """
        turn on readout channels (0 and 1) and also turn on selected qubit channels
        :param channels: qubit channels to turn off
        """
        # readout
        self.LO_res.set_power(-10)
        self.LO_res.set_power_on()

        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/0/offset', 0.6)
        self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/1/offset', 0.6)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 1)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 1)
        # qubit
        for channel in channels:
            self.hdawg.setDouble(f'/{self.hdawg_device}/sigouts/{channel}/offset', 0.4)
            self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{channel}/on', 1)

    def spectroscopy_preset_off(self, *channels):
        """
        turn off readout channels (0 and 1) and also turn of selected qubit channels
        :param channels: qubit channels to turn off
        """
        # readout
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/0/on', 0)
        self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/1/on', 0)
        # qubit
        for channel in channels:
            self.hdawg.setInt(f'/{self.hdawg_device}/sigouts/{channel}/on', 0)

    @property
    def __find_resonator(self):

        self.LO.set_status(0)

        # prior bandwidth
        bandwidth = self.LO_res.get_bandwidth()
        x_lim = self.LO_res.get_freq_limits()

        self.LO_res.set_bandwidth(self.LO_res_set_bandwidth)
        self.LO_res.set_nop(self.LO_res_set_nop)
        self.LO_res.set_freq_limits(self.fr_min, self.fr_max)
        self.LO_res.set_averages(self.LO_res_set_averages)

        # measure S21
        freqs = self.LO_res.get_freqpoints()
        notch = notch_port(freqs, self.LO_res.measure()['S-parameter'])
        notch.autofit(electric_delay=60e-9)
        result = round(notch.fitresults['fr'])

        # Resetting to the next round of measurements
        self.LO_res.set_bandwidth(bandwidth)
        self.LO_res.set_freq_limits(*x_lim)
        self.LO_res.set_nop(1)
        self.LO.set_status(1)

        return result
