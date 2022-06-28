# built-in libraries
from time import perf_counter
from typing import Iterable, Union
import time
# installable libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import numba as nb
import peakutils


@nb.jit(nopython=True)
def _complicated(raw_array_x, raw_array_y, x_min_, x_step_, y_min_, y_step_,
                 len_y_):
    ind_array = np.zeros(len(raw_array_x))

    if len(ind_array) >= 2:
        for k in range(len(raw_array_x)):
            x_ind = np.around((raw_array_x[k] - x_min_) / x_step_)
            y_ind = np.around((raw_array_y[k] - y_min_) / y_step_)

            ind_array[k] = x_ind * len_y_ + y_ind
            pass

        return ind_array
    else:
        pass


class SortingTools:

    def __init__(self):
        pass

    @staticmethod
    def k_max_idxs(arr, k):
        """
            Returns the indices of the k first largest elements of arr
            (in descending order in values)
        """
        assert k <= arr.size, 'k should be smaller or equal to the array size'
        arr_ = arr.astype(float)  # make a copy of arr
        max_idxs = []
        ful_max_idxs = []
        for _ in range(k):
            max_element = np.max(arr_)
            if np.isinf(max_element):
                break
            else:
                idx = np.where(arr_ == max_element)[0]
            max_idxs.append(idx)
            arr_[idx] = -np.inf
            ful_max_idxs += list(idx)
        return max_idxs, ful_max_idxs


class mrange:

    @classmethod
    def cust_range(cls, *args, rtol=1e-05, atol=1e-08, include=[True, False]):
        """
        Combines numpy.arange and numpy.isclose to mimic
        open, half-open and closed intervals.
        Avoids also floating point rounding errors as with
        numpy.arange(1, 1.3, 0.1)
        array([1. , 1.1, 1.2, 1.3])

        args: [start, ]stop, [step, ]
            as in numpy.arange
        rtol, atol: floats
            floating point tolerance as in numpy.isclose
        include: boolean list-like, length 2
            if start and end point are included
        """
        # process arguments
        if len(args) == 1:
            start = 0
            stop = args[0]
            step = 1
        elif len(args) == 2:
            start, stop = args
            step = 1
        else:
            assert len(args) == 3
            start, stop, step = tuple(args)

        # determine number of segments
        n = (stop - start) / step + 1

        # do rounding for n
        if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
            n = np.round(n)

        # correct for start/end is exluded
        if not include[0]:
            n -= 1
            start += step
        if not include[1]:
            n -= 1
            stop -= step

        return np.linspace(start, stop, int(n))

    @classmethod
    def crange(cls, *args, **kwargs):
        return mrange.cust_range(*args, **kwargs, include=[True, True])

    @classmethod
    def orange(cls, *args, **kwargs):
        return mrange.cust_range(*args, **kwargs, include=[True, False])


class timer:
    """
    timer class provides accurate variation of time.sleep() method. Example:
    timer.sleep(1) #  sleeps for 1. second

    Why to use this? If you use time.sleep() for 1 iteration, it wouldn't bring you problem, but if your script
    requires a loop with time.sleep() inside, Windows OS will ruin your script execution. To explain why this
    happened, Windows OS has its default timer resolution set to 15.625 ms or 64 Hz, which is decently enough for
    most of the applications. However, for applications that need very short sampling rate or time delay, then 15.625
    ms is not sufficient.
    """

    @staticmethod
    def sleep(sec: float):
        deadline = perf_counter() + sec
        while perf_counter() < deadline:
            pass


class Spectroscopy:
    __slots__ = ['x_min', 'x_max', 'x_step', '__x_step_DP', 'x_list', 'nx_steps',
                 'y_min', 'y_max', 'y_step', '__y_step_DP', 'y_list', 'ny_steps',
                 'load', 'frequency',
                 'x_raw', 'y_raw', 'z_raw',
                 '__z_2d', '__x_container', '__y_container', '__z_container',
                 'x_1d', 'y_1d', '__n_steps', '__len_y', '__x_for_approximate_idxs', '__approximation_y_keys']

    def __init__(self, *,
                 x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                 x_step: float = None, y_step: float = None, nx_points: float = None, ny_points: float = None,
                 x_arr: float = None, y_arr: float = None
                 ):
        """
        Class provides methods for working with live data for heatmap measurements
        :param x_min: x minimum value
        :type x_min int | float
        :param x_max: x maximum value (int | float)
        :type x_max int | float
        :param x_step: x step value (int)
        :type x_step int | float
        :param y_min: y minimum value (int | float)
        :type y_min int | float
        :param y_max: y maximum value (int | float)
        :type y_max int | float
        :param y_step: y step value (int)
        :type y_step int | float
        """

        "generating x list with proper step"
        if x_arr is None:
            self.x_min = x_min
            self.x_max = x_max

            if nx_points is not None:
                _, self.x_step = np.linspace(x_min, x_max, nx_points, retstep=True)
                self.x_step = round(self.x_step, 10)
            else:
                self.x_step = float(x_step)
            self.__x_step_DP = len(str(self.x_step).split(".")[1])

            self.x_list = mrange.orange(self.x_min, self.x_max + self.x_step,
                                        self.x_step)
            self.x_list = np.around(self.x_list, decimals=self.__x_step_DP)

        else:
            self.x_min = x_arr[0]
            self.x_max = x_arr[-1]
            _, self.x_step = np.linspace(self.x_min, self.x_max, len(x_arr), retstep=True)
            self.x_step = round(self.x_step, 10)
            self.__x_step_DP = len(str(self.x_step).split(".")[1])
            self.x_list = mrange.orange(self.x_min, self.x_max + self.x_step,
                                        self.x_step)
            self.x_list = np.around(self.x_list, decimals=self.__x_step_DP)

        "generating y list with proper step"
        if y_arr is None:
            self.y_min = y_min
            self.y_max = y_max

            if ny_points is not None:
                _, self.y_step = np.linspace(y_min, y_max, ny_points, retstep=True)
                self.y_step = round(self.y_step, 2)
            else:
                self.y_step = float(y_step)
            self.__y_step_DP = len(str(self.y_step).split(".")[1])

            self.y_list = mrange.orange(self.y_min, self.y_max + self.y_step,
                                        self.y_step)
            self.y_list = np.around(self.y_list, decimals=self.__y_step_DP)

        else:
            self.y_min = y_arr[0]
            self.y_max = y_arr[-1]
            _, self.y_step = np.linspace(self.y_min, self.y_max, len(y_arr), retstep=True)
            self.y_step = round(self.y_step, 10)
            self.__y_step_DP = len(str(self.y_step).split(".")[1])
            self.y_list = mrange.orange(self.y_min, self.y_max + self.y_step,
                                        self.y_step)
            self.y_list = np.around(self.y_list, decimals=self.__y_step_DP)

        self.load = None
        "x values that need to be send"
        self.frequency = None
        "y values that need to be send"

        self.x_raw = []
        self.y_raw = []
        self.z_raw = []

        self.nx_steps = len(self.x_list)
        self.ny_steps = len(self.y_list)

        # service variables
        self.__z_2d = np.empty((self.ny_steps, self.nx_steps))
        self.__z_2d[:, :] = np.nan

        self.__z_container = np.zeros(len(self.x_list) * len(self.y_list))
        self.__z_container[:] = np.nan
        self.__x_container = np.zeros(len(self.x_list) * len(self.y_list))
        self.__y_container = np.zeros(len(self.x_list) * len(self.y_list))

        self.x_1d = np.repeat(self.x_list, len(self.y_list))
        self.y_1d = np.tile(self.y_list, len(self.x_list))
        self.__n_steps = int(0.04 * len(self.y_list))
        self.__len_y = len(self.y_list)
        self.__x_for_approximate_idxs = None
        self.__approximation_y_keys = None

    def __getitem__(self, item):
        return self.raw_array[item]

    def __len__(self):
        return len(self.raw_array)

    def __call__(self):
        print('Base spectroscopy data class with parameters:\n')
        print(f'x_min={self.x_min};\tx_max={self.x_max}\tx_step={self.x_step}\n'
              f'y_min={self.y_min};\ty_max={self.y_max}\ty_step={self.y_step}\n')
        print(f'heatmap filling percentage: {len(self.raw_frame) / len(self.x_1d) * 100}, %')

    def __getattr__(self, name):
        raise AttributeError(f"Class Spectroscopy doesn't have {str(name)} attribute")

    def iter_setup(self, *, x_key: Union[float, int, Iterable] = None, y_key: Union[float, int, Iterable] = None,
                   x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None):
        """measurement setup. if all optional params are None then setup self.load and self.frequency for measuring all
        data
        :param x_key: x value(s) for measurement
        :type x_key: float | int | Iterable
        :param y_key: y value(s) for measurement
        :type y_key: float | int | Iterable
        :param x_min: x min value of measurement
        :type x_min: float
        :param x_max: x max value of measurement
        :type x_max: float
        :param y_min: y min value of measurement
        :type y_min: float
        :param y_max: y max value of measurement
        :type y_max: float
        :return:
        """
        if x_key is not None:
            x_key = self.__config_closest_values(x_key, self.x_list)

        if y_key is not None:
            if np.array(y_key).dtype == object:
                pass
            else:
                y_key = self.__config_closest_values(y_key, self.y_list)

        "set x_min x_max arrays"
        if (y_min is not None) | (y_max is not None) | (x_min is not None) | (x_max is not None):
            x_key = self.__config_arrays_from(min_value=x_min, max_value=x_max, step=self.x_step,
                                              array=self.x_list, array_set_input_in_func=x_key, column='x_value')

            y_key = self.__config_arrays_from(min_value=y_min, max_value=y_max, step=self.y_step,
                                              array=self.y_list, array_set_input_in_func=y_key, column='y_value')

        if x_key is not None:
            x_key = x_key if isinstance(x_key, Iterable) else [x_key, ]

        x = np.array(x_key) if x_key is not None else np.array(self.x_list)

        """config frequencies"""
        if y_key is not None:
            y_key = y_key if isinstance(y_key, Iterable) else [y_key, ]
            if np.array(y_key).dtype != object:
                y_key = np.tile(y_key, [len(x), 1])
            elif np.array(y_key).dtype == object and x_key is not None:
                ind_list = []
                for x_k in x_key:
                    ind_list.append(list(self.x_list).index(x_k))
                y_key = y_key[ind_list]

            else:
                pass

            if np.array(y_key).dtype == object:
                freqs = []
                for i in range(len(y_key)):
                    freqs += list(y_key[i])
                freqs = np.array(freqs)

            else:
                freqs = y_key.ravel()

        else:
            freqs = np.tile(self.y_list, len(x))
            pass

        """config currents"""
        if y_key is not None:
            if np.array(y_key).dtype == object:
                current_encapsulated = []
                for i, y_i in enumerate(y_key):
                    current_encapsulated.append([x[i], ] * len(y_i))

                if np.array(y_key).dtype == object:
                    currents = []
                    for i in range(len(current_encapsulated)):
                        currents += list(current_encapsulated[i])
                    currents = np.array(currents)

                else:
                    currents = np.array(current_encapsulated).ravel()

            else:
                current_encapsulated = []
                for i, y_i in enumerate(y_key):
                    current_encapsulated.append([x[i], ] * len(y_i))
                currents = np.array(current_encapsulated).ravel()
                pass

        else:
            currents = np.repeat(x, len(self.y_list))

        "collect all together"
        temp_df = pd.DataFrame({'x_value': currents, 'y_value': freqs})
        raw_frame_without_nans = self.raw_frame.dropna(axis=0)

        index1 = pd.MultiIndex.from_arrays([temp_df[col] for col in ['x_value', 'y_value']])
        index2 = pd.MultiIndex.from_arrays([raw_frame_without_nans[col] for col in ['x_value', 'y_value']])
        temp_df = temp_df.loc[~index1.isin(index2)]

        self.load = temp_df['x_value'].values
        self.frequency = temp_df['y_value'].values

        pass

    def write(self, *, x: Union[float, int] = None, y: Union[float, int] = None,
              z: Union[float, int] = None):
        """writes x coord value, y coord value and z coord value to class entity
        :param z: z value
        :type z: float | int
        :param x: x value
        :type x: float | int
        :param y: y value
        :type y: float | int
        """
        self.x_raw.append(x)  # Current write to DF
        self.y_raw.append(y)  # Frequency write to DF
        self.z_raw.append(z)  # Response write to DF
        pass

    def check_stop_on_iter(self, i: int):
        if len(self.x_raw) == len(self.y_raw) == len(self.z_raw):
            pass
        else:
            max_l = max([len(self.x_raw), len(self.y_raw), len(self.z_raw)])
            i = i + 1 if i < len(self.frequency) else i
            if len(self.y_raw) < max_l:
                self.y_raw.append(self.frequency[i])
            if len(self.z_raw) < max_l:
                self.z_raw.append(np.nan)

    @property
    def raw_frame(self):
        """generates raw Data Frame with columns [x_value, y_value, z_value]
        :return: dataframe
        """
        if len(self.x_raw) == len(self.y_raw) == len(self.z_raw):
            return pd.DataFrame({'x_value': self.x_raw, 'y_value': self.y_raw, 'z_value': self.z_raw})
        else:
            pass

    @property
    def raw_array(self):
        """generates raw 2-d numpy array np.array([y_value, x_value])
        :return: ndarray
        """
        if len(self.x_raw) == len(self.y_raw) == len(self.z_raw):
            return np.array([self.x_raw, self.y_raw, self.z_raw])
        else:
            pass

    def get_result(self, *, imshow: bool = False) -> pd.DataFrame:
        """return resulting Data Frame
        :param imshow: (optional) if True then result returns dataframe for matplotlib.pyplot.pcolormesh()
        :type imshow: bool
        :return: Pandas Data Frame. column.names: ['x_value', 'y_value', 'z_value']
        """
        l_h = len(self.__z_container[:len(self.z_raw)])
        l_x = len(self.__x_container[:len(self.x_raw)])
        l_y = len(self.__y_container[:len(self.y_raw)])

        if l_h == l_x == l_y:
            self.__z_container[:len(self.z_raw)] = self.z_raw
            self.__x_container[:len(self.x_raw)] = self.x_raw
            self.__y_container[:len(self.y_raw)] = self.y_raw
        else:
            pass

        for i in range(len(self.x_raw)):
            self.__z_2d[round((self.__y_container[i] - self.y_min) / self.y_step),
                        round((self.__x_container[i] - self.x_min) / self.x_step)] = self.__z_container[i]

        z_1d = self.__z_2d.ravel(order='F')

        if not imshow:
            df = pd.DataFrame({'x_value': self.x_1d, 'y_value': self.y_1d, 'z_value': z_1d})
        else:
            df = pd.DataFrame(data=self.__z_2d, columns=self.x_list, index=self.y_list)

        return df

    @property
    def non_njit_result(self):
        """generates resulting 2-d array of z values
        :return: numpy.ndarray
        """

        if len(self.x_raw) >= 2:
            array = np.array(self.raw_array)
            z_val = array[2, :]
            x_val = array[0, :]
            y_val = array[1, :]

            for i in range(len(z_val)):
                self.__z_2d[round((y_val[i] - self.y_min) / self.y_step),
                            round((x_val[i] - self.x_min) / self.x_step)] = z_val[i]

            return self.__z_2d

        else:
            pass

    @property
    def njit_result(self):
        """generates CPU accelerated resulting 2-d array of z values
        :return: numpy.ndarray
        """
        # array = np.copy(self.raw_array)
        array_to_njit = np.copy(self.raw_array)
        x_val = array_to_njit[0, :]
        y_val = array_to_njit[1, :]
        z_val = array_to_njit[2, :]

        ind_array = _complicated(x_val, y_val,
                                 self.x_min, self.x_step, self.y_min, self.y_step,
                                 self.__len_y)
        if len(self.x_raw) >= 2:
            array_to_process = np.copy(self.__z_container)
            array_to_process[ind_array.astype(int)] = z_val

            return array_to_process
        else:
            pass

    def xyz_peak(self, x_key: Iterable = None, thres: float = 0.7,
                 min_dist: float = 75, n_last: int = 20):
        """fit the heatmap curve by finding peak values
        :param x_key:
        :type x_key: Iterable
        :param thres:
        :type thres: float
        :param min_dist:
        :type min_dist: float
        :param n_last:
        :type n_last: int
        :return: x, y, z tuple of peak values
        """
        if x_key is not None:
            x_key = self.__config_closest_values(x_key, self.x_list)
            X = self.raw_frame[self.raw_frame.x_value.isin(x_key)].copy()
            X = X.reset_index(drop=True)
            x_set = x_key
        else:
            X = self.raw_frame.copy()
            """rows that we are looking for (every x_value)"""
            x_set = np.unique(X['x_value'].values)  # get an array of unique values of x

        tuple_list = ()
        for xx in x_set:
            z = X[X.x_value == xx].z_value.values
            y = X[X.x_value == xx].y_value.values

            mode_res = stats.mode(np.around(z), axis=None)[0][0]
            deltas = abs(abs(z) - abs(mode_res))
            _, n_last_idxs = SortingTools.k_max_idxs(deltas, n_last)

            peak_idxs = peakutils.indexes(abs(z), thres=thres, min_dist=min_dist)

            peak_and_delta = np.where(np.isin(peak_idxs, n_last_idxs))[0]

            if isinstance(peak_and_delta[0], np.ndarray):
                max_delta_idx = peak_and_delta[0][-1]
            else:
                max_delta_idx = peak_and_delta[0]

            max_delta_idx = n_last_idxs[max_delta_idx]

            y_min = y[max_delta_idx] - self.__n_steps * self.y_step
            y_max = y[max_delta_idx] + self.__n_steps * self.y_step

            z_sample = z[n_last_idxs[0]]

            if len(X[X.x_value == xx].y_value) <= 0.7 * len(self.y_list):
                temp_y_idx = (X[X.x_value == xx].z_value - z_sample).abs().idxmin()
            else:
                temp_y_idx = (X[(X.x_value == xx) & (X.y_value >= y_min) & (X.y_value <= y_max)
                                ].z_value - z_sample).abs().idxmin()

            temp_max_row = tuple(X.iloc[temp_y_idx])

            tuple_list += (temp_max_row,)
        # rotating array
        tuple_of_max_z_values = np.array(tuple_list).T
        return tuple_of_max_z_values

    def approximate(self, poly_nop: int = 1000, resolving_zone: float = 0.1, *,
                    x_key: Iterable = None, thres: float = 0.7, min_dist: float = 75, n_last: int = 20,
                    deg: int = 2):
        """approximating measured data with polyline of chosen degree
        :param poly_nop: number of points in fitted curve
        :type poly_nop: int
        :param resolving_zone: (0:1) - resolving zone for plot
        :type resolving_zone: float
        :param x_key: (optional) list of x values that will be used in approximation
        :type x_key: Iterable
        :param thres: [0:1] threshold of the normalized peak. default 0.7
        :type thres: float
        :param min_dist: minimum distant from one peak to another. default 75
        :type min_dist: float
        :param n_last: nop of max values
        :type n_last: int
        :param deg: Degree of the fitting polynomial
        :type deg: int
        :return: dict with keys x_key, y_key, mask, imshow_mask, poly_line, poly_coef
        """
        if x_key is not None:
            x_key = self.__config_closest_values(x_key, self.x_list)
            X = self.raw_frame[self.raw_frame.x_value.isin(x_key)].copy()
            X = X.reset_index(drop=True)
            x_set = x_key
        else:
            X = self.raw_frame.copy()
            """rows that we are looking for (every x_value)"""
            x_set = np.unique(X['x_value'].values)  # get an array of unique values of x

        tuple_of_max_z_values = self.xyz_peak(x_key=x_set, thres=thres, min_dist=min_dist, n_last=n_last)
        print('x', tuple_of_max_z_values[0])
        print('y', tuple_of_max_z_values[1])

        # creating poly curve
        poly = np.poly1d(np.polyfit(x=tuple_of_max_z_values[0], y=tuple_of_max_z_values[1], deg=deg))
        poly_x = np.linspace(self.x_min, self.x_max, poly_nop)

        """getting freq values for approximation"""
        y_for_approximate = []
        self.__x_for_approximate_idxs = []
        x_for_approximate_ind = 0
        for value in poly(self.x_list):
            y_for_approximate.append(self.__find_nearest(value))
            if (value <= self.y_min) or (value >= self.y_max):
                self.__x_for_approximate_idxs.append(x_for_approximate_ind)
            x_for_approximate_ind += 1

        """masking"""

        min_z_sample = X.z_value.mean() if fillna is True else np.nan
        max_z_sample = tuple_of_max_z_values[2].mean()

        get_result_df = self.get_result()
        get_result_df.loc[:, 'z_value'] = min_z_sample

        for i in range(len(y_for_approximate)):
            get_result_mask = (get_result_df['x_value'] == self.x_list[i]) & (
                    get_result_df['y_value'] == y_for_approximate[i])
            get_result_df.loc[get_result_mask, 'z_value'] = max_z_sample

        """approximation"""
        y_keys = []

        for xx in self.x_list:
            idx = get_result_df[get_result_df.x_value == xx].loc[get_result_df.z_value == max_z_sample].index[0]

            count_of_resolve_idx = len(self.y_list) * resolving_zone
            get_result_df.iloc[idx + 1:idx + round(count_of_resolve_idx / 2), 2] = max_z_sample / 2

            get_result_df.iloc[idx - round(count_of_resolve_idx / 2):idx, 2] = max_z_sample / 2
            y_keys.append(
                get_result_df.iloc[idx - int(count_of_resolve_idx / 2):
                                   idx + int(count_of_resolve_idx / 2), 1].to_list())

            self.__approximation_y_keys = np.array(y_keys, dtype=object)

        """checking for zero-length list. and fix it."""
        for i in range(len(y_keys)):
            if len(y_keys[i]) == 0:
                if i == len(y_keys) - 1:
                    y_keys[i] = list(y_keys[i - 1])
                else:
                    y_keys[i] = list(y_keys[i + 1])

        """config returning values"""
        """imshow"""
        z_list = []
        for xx in self.x_list:
            z_list.append(get_result_df[get_result_df.x_value == xx].loc[:, 'z_value'].values)
        get_imshow_result_df = pd.DataFrame(data=np.array(z_list).T, columns=self.x_list, index=self.y_list)

        "deleting bad x approximation"
        x_keys = np.delete(self.x_list, self.__x_for_approximate_idxs)

        return dict(y_key=np.array(y_keys, dtype=object),
                    x_key=x_keys,
                    mask=get_result_df,
                    imshow_mask=get_imshow_result_df,
                    poly_line={'x': poly_x, 'y': poly(poly_x)},
                    poly_coef=poly.c)

    def clean_up(self):
        """
        cleans data after approximation
        """
        i = 0
        for x in np.array(self.x_list, dtype=float):
            array2 = np.array(self.__approximation_y_keys[i], dtype=float)
            mask_arr = ~np.isclose(self.raw_frame['y_value'].values[:, None], array2, atol=.1).any(axis=1)
            idx = self.raw_frame[mask_arr & (self.raw_frame['x_value'] == x)].index

            self.x_raw = np.delete(self.x_raw, idx)
            self.y_raw = np.delete(self.y_raw, idx)
            self.z_raw = np.delete(self.z_raw, idx)
            self.__z_2d[:, i] = self.__z_2d[:, i] * np.nan
            i += 1

        self.x_raw = list(self.x_raw)
        self.y_raw = list(self.y_raw)
        self.z_raw = list(self.z_raw)

        self.__z_container[:] = np.nan
        self.__x_container[:] = 0
        self.__y_container[:] = 0

    def cls(self):
        """
        clean all heatmap
        """
        self.x_raw = []
        self.y_raw = []
        self.z_raw = []

        self.__z_2d = np.empty((self.ny_steps, self.nx_steps))
        self.__z_2d[:, :] = np.nan

        self.__z_container[:] = np.nan
        self.__x_container[:] = 0
        self.__y_container[:] = 0

    def drop(self, x: Union[float, int, Iterable] = None, y: Union[float, int, Iterable] = None,
             x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None):
        """
        drops specific values (x, y)
        :param x: x value(s)
        :type x: float | int | Iterable
        :param y: y value(s) [float, int, Iterable]
        :type y: float | int | Iterable
        :param x_min: minimum x value [float, int]
        :type x_min: float | int
        :param x_max: maximum x value [float, int]
        :type x_max: float | int
        :param y_min: minimum y value [float, int]
        :type y_min: float | int
        :param y_max: maximum y value [float, int]
        :type y_max: float | int
        """
        """if x list is given then find nearest values in self.x_list"""
        if x is not None:
            x = self.__config_closest_values(x, self.raw_array[0, :])

        """if y list is given then find nearest values in self.y_list"""
        if y is not None:
            y = self.__config_closest_values(y, self.raw_array[1, :])

        "set min max arrays"
        if (y_min is not None) | (y_max is not None) | (x_min is not None) | (x_max is not None):
            x = self.__config_arrays_from(min_value=x_min, max_value=x_max, step=self.x_step,
                                          array=self.raw_array[0, :], array_set_input_in_func=x, column='x_value')

            y = self.__config_arrays_from(min_value=y_min, max_value=y_max, step=self.y_step,
                                          array=self.raw_array[1, :], array_set_input_in_func=y, column='y_value')

        if (x is not None) and (y is None):
            self.__drop_the('x_value', x)
            pass

        if (y is not None) and (x is None):
            self.__drop_the('y_value', y)
            pass

        if (x is not None) and (y is not None):
            self.__drop_the_cols(x, y)
            pass

    def load_data(self, raw_file: str):
        raw_csv = pd.read_csv(raw_file)

        self.x_raw = list(raw_csv.x_value.values)
        self.y_raw = list(raw_csv.y_value.values)
        self.z_raw = list(raw_csv.z_value.values)

    def __drop_the(self, column: str, value_s: Union[float, int, Iterable]):

        decimals = self.__x_step_DP if column == 'x_value' else self.__y_step_DP

        value_s = list(value_s) if isinstance(value_s, Iterable) else [value_s, ]
        value_s = np.around(value_s, decimals=decimals)
        idx = self.raw_frame[self.raw_frame[column].isin(value_s)].index

        self.x_raw = list(np.delete(self.x_raw, idx))
        self.y_raw = list(np.delete(self.y_raw, idx))
        self.z_raw = list(np.delete(self.z_raw, idx))

        self.__z_container[:] = np.nan
        self.__x_container[:] = 0
        self.__y_container[:] = 0

        column_var = list(self.x_list) if column == 'x_value' else list(self.y_list)

        col_idx_list = [list(column_var).index(value_s[i]) for i in range(len(value_s))]

        if column == 'x_value':
            for i in col_idx_list:
                self.__z_2d[:, i] = self.__z_2d[:, i] * np.nan
        else:
            for i in col_idx_list:
                self.__z_2d[i] = self.__z_2d[i] * np.nan

        pass

    def __drop_the_cols(self, x_values: Union[float, int, Iterable], y_values: Union[float, int, Iterable]):

        x_decimals, y_decimals = self.__x_step_DP, self.__y_step_DP

        # x mask
        x_value_s = list(x_values) if isinstance(x_values, Iterable) else [x_values, ]
        x_value_s = np.around(x_value_s, decimals=x_decimals)
        x_mask = np.logical_or.reduce(np.isclose(self.raw_frame['x_value'].to_numpy()[None, :], x_value_s[:, None]))

        # y mask
        y_value_s = list(y_values) if isinstance(y_values, Iterable) else [y_values, ]
        y_value_s = np.around(y_value_s, decimals=y_decimals)
        y_mask = np.logical_or.reduce(np.isclose(self.raw_frame['y_value'].to_numpy()[None, :], y_value_s[:, None]))

        # idx
        msk = pd.DataFrame({'y': y_mask, 'x': x_mask}).all(axis=1)
        idx = self.raw_frame[msk].index

        self.x_raw = list(np.delete(self.x_raw, idx))
        self.y_raw = list(np.delete(self.y_raw, idx))
        self.z_raw = list(np.delete(self.z_raw, idx))

        self.__z_container[:] = np.nan
        self.__x_container[:] = 0
        self.__y_container[:] = 0

        x_col_idx_list = [list(self.x_list).index(x_value_s[i]) for i in range(len(x_value_s))]
        y_col_idx_list = [list(self.y_list).index(y_value_s[i]) for i in range(len(y_value_s))]

        for i in x_col_idx_list:
            for j in y_col_idx_list:
                self.__z_2d[j, i] = self.__z_2d[j, i] * np.nan
        pass

    def __config_closest_values(self, input_value: Union[float, int, Iterable], base_array: Iterable):

        """
        if x list is given then find the nearest values in self.x_list
        :param input_value:
        :param base_array:
        :return:
        """

        if isinstance(input_value, float) or isinstance(input_value, int):
            input_value = self.__find_nearest_universal(base_array, input_value)
        else:
            input_value_temp = []
            for value in input_value:
                input_value_temp.append(self.__find_nearest_universal(base_array, value))
            input_value = np.array(input_value_temp)
        return input_value

    def __config_arrays_from(self, min_value: float, max_value: float, step: float,
                             array: Union[float, int, Iterable], array_set_input_in_func: Union[float, int, Iterable],
                             column: str = 'x_value'):

        """configurate arrays from given values"""

        if (min_value is not None) and (max_value is not None):
            min_value = self.__find_nearest_universal(array, min_value)
            max_value = self.__find_nearest_universal(array, max_value)
            array = mrange.orange(min_value, max_value + step, step)
            return array

        elif (min_value is not None) and (max_value is None):
            if column == 'x_value':
                max_v = self.x_max
                step_v = self.x_step
            else:
                max_v = self.y_max
                step_v = self.y_step

            min_value = self.__find_nearest_universal(array, min_value)
            array = mrange.orange(min_value, max_v + step_v, step_v)
            return array
        elif (min_value is None) and (max_value is not None):
            if column == 'x_value':
                min_v = self.x_min
                step_v = self.x_step
            else:
                min_v = self.y_min
                step_v = self.y_step

            max_value = self.__find_nearest_universal(array, max_value)
            array = mrange.orange(min_v, max_value + step_v, step_v)
            return array

        else:
            return array_set_input_in_func

    def __smooth_list_gaussian(self, list1, degree=5):
        window = degree * 2 - 1
        weight = np.array([1.0] * window)
        weightGauss = []
        for i in range(window):
            i = i - degree + 1
            frac = i / float(window)
            gauss = 1 / (np.exp((4 * frac) ** 2))
            weightGauss.append(gauss)
        weight = np.array(weightGauss) * weight
        smoothed = [0.0] * (len(list1) - window)
        for i in range(len(smoothed)):
            smoothed[i] = sum(np.array(list1[i:i + window]) * weight) / sum(weight)
        return np.array(smoothed)

    def __find_nearest_universal(self, arr, value):
        array = np.asarray(arr)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def __find_nearest(self, value):
        array = np.asarray(self.y_list)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
