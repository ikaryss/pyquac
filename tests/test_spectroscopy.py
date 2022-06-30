import numpy as np
from pyquac.datatools import Spectroscopy
import pytest

x_arr1, y_arr1 = np.linspace(-1, 1, 100), np.linspace(-10e9, 50e9, 600)
len_x1, len_y1 = len(x_arr1), len(y_arr1)

x_arr2, y_arr2 = np.linspace(-1.05, 1, 122), np.linspace(-10.002e9, 50e9, 638)
len_x2, len_y2 = len(x_arr2), len(y_arr2)

x_arr3, y_arr3 = np.linspace(-256.1242, 1, 122), np.linspace(-10.002, 50, 638)
len_x3, len_y3 = len(x_arr3), len(y_arr3)


@pytest.mark.parametrize('x_array, y_array', [
    (x_arr1, y_arr1),
    (x_arr2, y_arr2),
    (x_arr3, y_arr3)
])
def test_array_init(x_array, y_array):
    data = Spectroscopy(x_arr=x_array, y_arr=y_array)
    assert (len(data.x_list) == len(x_array)) and (len(data.y_list) == len(y_array))
