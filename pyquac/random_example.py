# built-in libraries
from typing import Iterable, Union

# installable libraries
import numpy as np

# family class
from pyquac.datatools import Spectroscopy, timer


class Random_spectroscopy(Spectroscopy):
    """
    Two Tone Spectroscopy for 1 qubit measurements
    """

    def __init__(
        self,
        *,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        nx_points=None,
        x_step=None,
        y_step=None,
        ny_points=None,
        x_arr=None,
        y_arr=None,
    ):
        super().__init__(
            x_min=x_min,
            x_max=x_max,
            nx_points=nx_points,
            y_min=y_min,
            y_max=y_max,
            ny_points=ny_points,
            x_step=x_step,
            y_step=y_step,
            x_arr=x_arr,
            y_arr=y_arr,
        )
        pass

    def run_measurements(
        self,
        *,
        x_key: Union[float, int, Iterable] = None,
        y_key: Union[float, int, Iterable] = None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        sleep=0.007,
    ):
        self.iter_setup(
            x_key=x_key, y_key=y_key, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        iterations = len(self.load)
        try:
            for i in range(iterations):
                self.write(x=self.load[i], y=self.frequency[i], z=np.random.random())
                timer.sleep(sleep)

        except KeyboardInterrupt:
            pass

    def dump_data(self):
        print("hurray")
