# main file for impulse spectroscopy measurements

# USEFUL LINKS
# 1. ПЕРВОНАЧАЛЬНАЯ НАСТРОЙКА UHFQA https://docs.zhinst.com/zhinst-toolkit/en/latest/examples/uhfqa_result_unit.html
# 2. PULSED RESONATOR SPECTROSCOPY https://docs.zhinst.com/shfqc_user_manual/tutorials/tutorial_pulsed_spectroscopy.html
# 3. MULTIPLEXED QUBIT READOUT https://docs.zhinst.com/shfqa_user_manual/tutorial_qubit_readout.html

from typing import Iterable
import numpy as np
from pyquac.datatools import Spectroscopy, timer
from pyquac.main.fmn_pulse_config import xy_control, z_readout_combined

import zhinst
from zhinst.toolkit import Session, Sequence

from drivers.M9290A import *
from drivers.N5183B import *
from drivers.k6220 import *
from drivers.znb_besedin import *
from resonator_tools import circlefit
from resonator_tools.circuit import notch_port


class Sts_pulsed(Spectroscopy):
    """Single tone pulsed spectroscopy"""

    def __init__(
        self,
        *,
        x_arr: Iterable = None,
        y_arr: Iterable = None,
        x_min: np.float32 = None,
        x_max: np.float32 = None,
        y_min: np.float32 = None,
        y_max: np.float32 = None,
        x_step: np.float32 = None,
        y_step: np.float32 = None,
        nx_points: int = None,
        ny_points: int = None,
        hdawg_host: str = "127.0.0.1",
        hdawg_device: str = "dev8210",
        hdawg_port: int = 8004,
        hdawg_mode: int = 6,
        hdwag_avg_count: int,
        hdawg_wait_time,
        uhfqa_host: str,
        uhfqa_device: str,
        uhfqa_port: int,
        uhfqa_mode: int,
        uhfqa_monitor_length: int = 1024,
        uhfqa_readout_ch_number: int = 1,  # применяется в примере https://docs.zhinst.com/zhinst-toolkit/en/latest/examples/uhfqa_result_unit.html
        r_amp,
        r_length,
        r_fade_length,
        r_channel_I,
        r_channel_Q,
        z_amp,
        z_length,
        z_fade_length,
        z_channel,
        pi_amp,
        pi_length,
        pi_channel_I,
        pi_channel_Q,
        LO_res_port: str = "TCPIP0::192.168.180.110::inst0::INSTR",
        LO_res_set_bandwidth: int = 20,
        LO_res_set_power: int = -10,
        LO_res_set_averages=1,
    ):
        super().__init__(
            x_arr=x_arr,
            y_arr=y_arr,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_step=x_step,
            y_step=y_step,
            nx_points=nx_points,
            ny_points=ny_points,
        )

        # Channels class variables
        self._ch_z_, self._ch_r_i_, self._ch_r_q_ = z_channel, r_channel_I, r_channel_Q
        self._ch_pi_i_, self._ch_pi_q_ = pi_channel_I, pi_channel_Q

        # HDAWG init
        session = Session(server_host=hdawg_host, server_port=hdawg_port)
        self.hdawg = session.connect_device(serial=hdawg_device)

        # UHFQA init
        session_u = Session(server_host=uhfqa_host, server_port=uhfqa_port)
        self.uhfqa = session_u.connect_device(serial=uhfqa_device)
        self.N_READOUT_CHANNELS = uhfqa_readout_ch_number

        # context manager reduces network overhead and increases speed
        with self.uhfqa.set_transaction():
            # 1. Parameters for QAmonitor
            """
            self.uhfqa.qas[0].monitor.averages(uhfqua_monitor_averages)
            """
            self.uhfqa.qas[0].monitor.reset(True)
            self.uhfqa.qas[0].monitor.enable(True)
            self.uhfqa.qas[0].monitor.length(uhfqa_monitor_length)

            # 2. Delay in measurement start
            self.uhfqa.qas[0].delay(100)

            # 3. Configure integration weights
            for ch in range(self.N_READOUT_CHANNELS):
                self.uhfqa.qas[0].integration.weights[ch].real(
                    np.zeros(2 * r_fade_length + r_length)
                )
                self.uhfqa.qas[0].integration.weights[ch].imag(
                    np.zeros(2 * r_fade_length + r_length)
                )

                # 3.1 Subscribing to nodes
                node = self.uhfqa.qas[0].result.data[ch].wave
                node.subscribe()

                # 3.2 Getting results
                "dataset = session_u.poll()"

        # Config Z pulse and Readout pulse program to the 1st core of HDAWG
        self.z_r_program = z_readout_combined(
            amplitude_z=z_amp,
            channel_z=self.ch_z,
            length_z=z_length,
            fade_length_z=z_fade_length,
            amplitude_r=r_amp,
            channel_i_r=self.ch_r_i,
            channel_q_r=self.ch_r_q,
            length_r=r_length,
            fade_length_r=r_fade_length,
            averages_count=hdwag_avg_count,
            wait_time=hdawg_wait_time,
        )

        self.z_r_compile_info = self.hdawg.awgs[0].load_sequencer_program(
            self.z_r_program
        )  # returns name of embedded ELF filename

        # Config XY pulse program to the 2nd core of HDAWG (Every core can operate 2 ports)
        self.xy_program = xy_control(
            amplitude=pi_amp,
            length=pi_length,
            channel_I=self.ch_pi_i,
            channel_Q=self.ch_pi_q,
            z_length=z_fade_length + z_length / 2,
        )

        self.xy_compile_info = self.hdawg.awgs[1].load_sequencer_program(
            self.xy_program
        )

        # LO connect
        self.LO_res = Znb(LO_res_port)  # resonator

        # set base parameters of LO res
        self.LO_res_set_bandwidth = LO_res_set_bandwidth
        self.LO_res_set_power = LO_res_set_power
        self.LO_res.set_averages(LO_res_set_averages)

    def run_measurements(self, sleep: float = 0.0007):
        self.iter_setup()

        # enable channels
        for ch in [
            self._ch_z_,
            self._ch_r_i_,
            self._ch_r_q_,
            self._ch_pi_i_,
            self._ch_pi_q_,
        ]:
            self.hdawg.sigouts[ch].on(True)

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
                    # Send pulse to Z-control
                    """
                    self.hdawg.awgs[0].ready.wait_for_state_change(1)
                    self.hdawg.awgs[0].enable(True)
                    """
                    # single - Flag if the sequencer should be disabled after finishing
                    self.hdawg.awgs[0].enable_sequencer(single=True)

                    # Send modulated Pi-pulse to XY control
                    self.hdawg.awgs[1].enable_sequencer(single=True)

                    # Wait until the AWG is finished.
                    # timeout (float) – The maximum waiting time in seconds for the generator (default: 10)
                    # sleep_time (float) – Time in seconds to wait between requesting generator state (default: 0.005)
                    self.hdawg.awgs[0].wait_done(timeout=10, sleep_time=0.005)

                    """
                    wave = self.uhfqa.qas[0].result.data[self.uhfqa_ch].wave
                    """

                    ###
                    # Where should we send load[i]
                    ###

                    self.LO_res.set_averages(self.LO_res_set_averages)
                    result = self.LO_res.measure()["S-parameter"]

                    for j in range(freq_len):
                        self.write(
                            x=self.load[i],
                            y=self.y_list[j],
                            z=20 * np.log10(abs(result[j])),
                        )

                timer.sleep(sleep)

        except KeyboardInterrupt:
            if (self.x_raw[-1] == self.x_list[-1]) and (
                self.y_raw[-1] == self.y_list[-1]
            ):
                pass
            else:
                # drop the last column if interrupt for stable data saving
                self.drop(
                    x=[
                        self.x_raw[-1],
                    ]
                )
            pass

        # disable channels
        for ch in [
            self._ch_z_,
            self._ch_r_i_,
            self._ch_r_q_,
            self._ch_pi_i_,
            self._ch_pi_q_,
        ]:
            self.hdawg.sigouts[ch].on(False)
