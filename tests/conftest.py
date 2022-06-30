import pytest
import numpy as np


@pytest.fixture(scope='session')
def init_spectroscopy():
    x_arr1, y_arr1 = np.linspace(-1, 1, 100), np.linspace(-10e9, 50e9, 600)
    len_x1, len_y1 = len(x_arr1), len(y_arr1)

    x_arr2, y_arr2 = np.linspace(-1.05, 1, 122), np.linspace(-10.002e9, 50e9, 638)
    len_x2, len_y2 = len(x_arr2), len(y_arr2)

    x_arr3, y_arr3 = np.linspace(-256.1242, 1, 122), np.linspace(-10.002, 50, 638)
    len_x3, len_y3 = len(x_arr3), len(y_arr3)

    return [
        (x_arr1, y_arr1, len_x1, len_y1),
        (x_arr2, y_arr2, len_x2, len_y2),
        (x_arr3, y_arr3, len_x3, len_y3)
    ]
