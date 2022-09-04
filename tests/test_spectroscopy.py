import numpy as np
from typing import Iterable
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


# Check for iter_setup (all true for i in range (xmin xmax))
x_key1 = float(np.random.uniform(-1, 1, 1))
y_key2 = float(np.random.uniform(4e9, 5e9, 1))
x_key3 = float(np.random.uniform(-1, 1, 1))
y_key3 = float(np.random.uniform(4e9, 5e9, 1))

x_min4 = float(np.random.uniform(-1, 1, 1))

x_min5 = float(np.random.uniform(-1, -0.95, 1))
x_max5 = float(np.random.uniform(0, 1, 1))

y_min6 = float(np.random.uniform(4e9, 5e9, 1))

y_min7 = float(np.random.uniform(4e9, 4.5e9, 1))
y_max7 = float(np.random.uniform(4.6e9, 5e9, 1))

x_key_arr = np.random.uniform(-1, 1, 5)
y_key_arr = np.random.uniform(4e9, 5e9, 100)


@pytest.mark.parametrize('x_key, y_key, x_min, x_max, y_min, y_max', [
    (x_key1, None, None, None, None, None), # +
    (None, y_key2, None, None, None, None),# +
    (x_key3, y_key3, None, None, None, None),# +
    (None, None, x_min4, None, None, None),# +
    (None, None, None, x_max5, None, None),# +
    (None, None, x_min5, x_max5, None, None),# +
    (None, None, None, None, y_min6, None),# +
    (None, None, None, None, None, y_max7),# +
    (None, None, None, None, y_min7, y_max7),# +
    (None, None, x_min5, x_max5, y_min7, y_max7),# +
    (None, None, x_min5, None, y_min7, None),# +
    (None, None, None, x_max5, None, y_max7),# +
    (None, None, x_min5, None, None, y_max7),# +
    (None, None, None, x_max5, y_min7, None),# +

    (x_key1, None, None, None, y_min7, None),# +
    (x_key1, None, None, None, None, y_max7),# +
    (x_key1, None, None, None, y_min7, y_max7),# +

    (None, y_key2, x_min5, None, None, None),
    (None, y_key2, None, x_max5, None, None),
    (None, y_key2, x_min5, x_max5, None, None),

    (x_key_arr, None, None, None, None, None),
    (None, y_key_arr, None, None, None, None),
    (x_key_arr, y_key_arr, None, None, None, None),

    (x_key_arr, None, None, None, y_min7, None),
    (x_key_arr, None, None, None, None, y_max7),
    (x_key_arr, None, None, None, y_min7, y_max7),

    (None, y_key_arr, x_min5, None, None, None),
    (None, y_key_arr, None, x_max5, None, None),
    (None, y_key_arr, x_min5, x_max5, None, None),
])
def test_iter_setup(x_key, y_key, x_min, x_max, y_min, y_max):
    y_min_d = 4e9
    y_max_d = 5e9
    ny_points_d = 401

    x_min_d = -1
    x_max_d = 1
    x_step_d = 0.004
    data = Spectroscopy(x_min=x_min_d, x_max=x_max_d, x_step=x_step_d, y_min=y_min_d, y_max=y_max_d,
                        ny_points=ny_points_d)
    data.iter_setup(x_key=x_key, y_key=y_key, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    # X mask set
    if x_key is not None:
        x_key = data._Spectroscopy__config_closest_values(x_key, data.x_list)

        msk = np.isin(data.x_1d, x_key)
        if (y_min is not None) and (y_max is None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                msk = ((np.isin(data.x_1d, x_key)) & (data.y_1d >= y_min))
                y_mask = data.y_1d[msk]
        elif (y_max is not None) and (y_min is None):
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (np.isin(data.x_1d, x_key)) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif (y_min is not None) and (y_max is not None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (np.isin(data.x_1d, x_key)) & (data.y_1d >= y_min) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif y_key is not None:
                y_key = data._Spectroscopy__config_closest_values(y_key, data.y_list)
                msk = np.isin(data.x_1d, x_key) & np.isin(data.y_1d, y_key)
                y_mask = data.y_1d[msk]
        else:
                y_mask = None
                pass
        x_mask = data.x_1d[msk]

    elif (x_min is not None) and (x_max is None):
        x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
        msk = data.x_1d >= x_min

        if (y_min is not None) and (y_max is None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                msk = (data.x_1d >= x_min) & (data.y_1d >= y_min)
                y_mask = data.y_1d[msk]
        elif (y_max is not None) and (y_min is None):
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (data.x_1d >= x_min) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif (y_min is not None) and (y_max is not None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)

                msk = (data.x_1d >= x_min) & (data.y_1d >= y_min) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif y_key is not None:
                y_key = data._Spectroscopy__config_closest_values(y_key, data.y_list)
                msk = (data.x_1d >= x_min) & np.isin(data.y_1d, y_key)
                y_mask = data.y_1d[msk]
        else:
                y_mask = None
                pass

        x_mask = data.x_1d[msk]

    elif (x_max is not None) and (x_min is None):
        x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
        msk = data.x_1d <= x_max

        if (y_min is not None) and (y_max is None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                msk = (data.x_1d <= x_max) & (data.y_1d >= y_min)
                y_mask = data.y_1d[msk]
        elif (y_max is not None) and (y_min is None):
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (data.x_1d <= x_max) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif (y_min is not None) and (y_max is not None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (data.x_1d <= x_max) & (data.y_1d >= y_min) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif y_key is not None:
                y_key = data._Spectroscopy__config_closest_values(y_key, data.y_list)
                msk = (data.x_1d <= x_max) & np.isin(data.y_1d, y_key)
                y_mask = data.y_1d[msk]
        else:
                y_mask = None
                pass
        x_mask = data.x_1d[msk]


    elif (x_min is not None) and (x_max is not None):
        x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
        x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
        msk = (data.x_1d >= x_min) & (data.x_1d <= x_max)

        if (y_min is not None) and (y_max is None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                msk = (data.x_1d >= x_min) & (data.x_1d <= x_max) & (data.y_1d >= y_min)
                y_mask = data.y_1d[msk]
        elif (y_max is not None) and (y_min is None):
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (data.x_1d >= x_min) & (data.x_1d <= x_max) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif (y_min is not None) and (y_max is not None):
                y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
                y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
                msk = (data.x_1d >= x_min) & (data.x_1d <= x_max) & (data.y_1d >= y_min) & (data.y_1d <= y_max)
                y_mask = data.y_1d[msk]
        elif y_key is not None:
                y_key = data._Spectroscopy__config_closest_values(y_key, data.y_list)
                msk = (data.x_1d >= x_min) & (data.x_1d <= x_max) & np.isin(data.y_1d, y_key)
                y_mask = data.y_1d[msk]
        else:
                y_mask = None
                pass
        x_mask = data.x_1d[msk]

    else:
        x_mask = None
        print('x mask is undefined')

    # Y mask set
    if y_key is not None:
        y_key = data._Spectroscopy__config_closest_values(y_key, data.y_list)

        msk = np.isin(data.y_1d, y_key)
        if (x_min is not None) and (x_max is None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                msk = ((np.isin(data.y_1d, y_key)) & (data.x_1d >= x_min))
                x_mask = data.x_1d[msk]
        elif (x_max is not None) and (x_min is None):
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (np.isin(data.y_1d, y_key)) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif (x_min is not None) and (x_max is not None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (np.isin(data.y_1d, y_key)) & (data.x_1d >= x_min) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif x_key is not None:
                x_key = data._Spectroscopy__config_closest_values(x_key, data.x_list)
                msk = np.isin(data.y_1d, y_key) & np.isin(data.x_1d, x_key)
                x_mask = data.x_1d[msk]
        else:
                x_mask = None
                pass
        y_mask = data.y_1d[msk]
            
    elif (y_min is not None) and (y_max is None):

        y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
        msk = data.y_1d >= y_min

        if (x_min is not None) and (x_max is None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                msk = (data.y_1d >= y_min) & (data.x_1d >= x_min)
                x_mask = data.x_1d[msk]
        elif (x_max is not None) and (x_min is None):
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (data.y_1d >= y_min) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif (x_min is not None) and (x_max is not None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)

                msk = (data.y_1d >= y_min) & (data.x_1d >= x_min) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif x_key is not None:
                x_key = data._Spectroscopy__config_closest_values(x_key, data.x_list)
                msk = (data.y_1d >= y_min) & np.isin(data.x_1d, x_key)
                x_mask = data.x_1d[msk]
        else:
                x_mask = None
                pass

        y_mask = data.y_1d[msk]

    elif (y_max is not None) and (y_min is None):
        y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
        msk = data.y_1d <= y_max

        if (x_min is not None) and (x_max is None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                msk = (data.y_1d <= y_max) & (data.x_1d >= x_min)
                x_mask = data.x_1d[msk]
        elif (x_max is not None) and (x_min is None):
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (data.y_1d <= y_max) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif (x_min is not None) and (x_max is not None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (data.y_1d <= y_max) & (data.x_1d >= x_min) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif x_key is not None:
                x_key = data._Spectroscopy__config_closest_values(x_key, data.x_list)
                msk = (data.y_1d <= y_max) & np.isin(data.x_1d, x_key)
                x_mask = data.x_1d[msk]
        else:
                x_mask = None
                pass
        y_mask = data.y_1d[msk]


    elif (y_min is not None) and (y_max is not None):
        y_min = data._Spectroscopy__config_closest_values(y_min, data.y_list)
        y_max = data._Spectroscopy__config_closest_values(y_max, data.y_list)
        msk = (data.y_1d >= y_min) & (data.y_1d <= y_max)

        if (x_min is not None) and (x_max is None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                msk = (data.y_1d >= y_min) & (data.y_1d <= y_max) & (data.x_1d >= x_min)
                x_mask = data.x_1d[msk]
        elif (x_max is not None) and (x_min is None):
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (data.y_1d >= y_min) & (data.y_1d <= y_max) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif (x_min is not None) and (x_max is not None):
                x_min = data._Spectroscopy__config_closest_values(x_min, data.x_list)
                x_max = data._Spectroscopy__config_closest_values(x_max, data.x_list)
                msk = (data.y_1d >= y_min) & (data.y_1d <= y_max) & (data.x_1d >= x_min) & (data.x_1d <= x_max)
                x_mask = data.x_1d[msk]
        elif x_key is not None:
                x_key = data._Spectroscopy__config_closest_values(x_key, data.x_list)
                msk = (data.y_1d >= y_min) & (data.y_1d <= y_max) & np.isin(data.x_1d, x_key)
                x_mask = data.x_1d[msk]
        else:
                x_mask = None
                pass
        y_mask = data.y_1d[msk]

    else:
        y_mask = None
        print('y mask is undefined')

    if (x_mask is not None) and (y_mask is not None):
        data_load = np.array(sorted(np.copy(data.load))) if isinstance(data.load, Iterable) else np.copy(data.load)
        data_frequency = np.array(sorted(np.copy(data.frequency))) if isinstance(data.frequency, Iterable) else np.copy(data.frequency)
        x_mask = np.array(sorted(x_mask)) if isinstance(x_mask, Iterable) else np.copy(x_mask)
        y_mask = np.array(sorted(y_mask)) if isinstance(y_mask, Iterable) else np.copy(y_mask)
        # print(all(data_frequency == y_mask) and all(data_load == x_mask))
        assert all(data_frequency == y_mask) and all(data_load == x_mask)

    elif (x_mask is not None) and (y_mask is None):
        data_load = np.array(sorted(np.copy(data.load))) if isinstance(data.load, Iterable) else np.copy(data.load)
        x_mask = np.array(sorted(x_mask)) if isinstance(x_mask, Iterable) else np.copy(x_mask)
        # print(all(data_load == x_mask))
        assert all(data_load == x_mask)

    if (x_mask is None) and (y_mask is not None):
        data_frequency = np.array(sorted(np.copy(data.frequency))) if isinstance(data.frequency, Iterable) else np.copy(data.frequency)
        y_mask = np.array(sorted(y_mask)) if isinstance(y_mask, Iterable) else np.copy(y_mask)
        # print(all(data_frequency == y_mask))
        assert all(np.equal(data_frequency, y_mask))
