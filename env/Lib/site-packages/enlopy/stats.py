"""This modules contains methods which correspond to estimation of statistics (features) for timeseries."""

import numpy as np
from scipy.signal import find_peaks_cwt, ricker
from scipy.stats import ks_2samp
from itertools import groupby

# make it work only with ndarray?

def get_mean(x, trunced=False):
    if trunced:
        x = x[x > 0]
    return np.mean(x)


def get_lf(x, trunced=False):
    """Load factor"""
    if trunced:
        x = x[x > 0]
    try:
        return np.mean(x) / np.max(x)
    except ZeroDivisionError:
        return np.nan


def get_trend(x, deg=1):
    # Assumes equally spaced series
    xx = np.arange(len(x))
    coeff = np.polyfit(xx, x, deg) #fails if empty (e.g. pd.Series([0]*800)
    return coeff[-2]


def get_rle(x, a):
    """Run length encoding algorithm.
    >>> get_rle("aaaaaaaaaaaaaaaaahhhh,,,,iuuuuuuiiaaalllaaa","a")
    [17, 3, 3]
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in groupby(x) if value == a]
        return res if len(res) > 0 else [0]


def largest_dur_of_zero(x):
    return max(get_rle(x, 0))


def get_peaks(x, n):
    return find_peaks_cwt(x, widths=np.arange(1, n + 1), wavelet=ricker)


def get_dur_val(x, a):
    """Get duration of periods that the timeseries is above one value.
    E.g It can be used to check the non-zero load duration"""
    return len(x[x == a])  # or sum(x[x>0]) ?

def get_percentile(x, a, trunced=False):
    if trunced:
        x = x[x > 0]
    return np.nanpercentile(x, a)


def get_ramp_rates(x, a=95):
    """Retrieve ramp rate of a percentile.
    a=100 for maximum-minimum. """
    ramps = np.diff(x)
    return np.round(np.nanpercentile(ramps, 100 - a),3), np.round(np.nanpercentile(ramps, a),3)


def get_highest_periodicity(x):  # highest peaks of fft
    from scipy.signal import welch

    length = len(x)
    w = welch(x, scaling="spectrum", nperseg=length // 2)
    peak_ind = get_peaks(w[1], 5)
    periods = [np.round(1 / w[0][peak]) for peak in peak_ind]
    # filter periods. Remove too small and too high
    max_p = length // 2
    min_p = 3  # Do not show short term AR(1) - AR(3)
    return tuple(period for period in periods if min_p < period < max_p)


def get_load_ratio(x):
    if np.min(x) != 0:
        return np.max(x) / np.min(x)


def get_autocorr(x, lag=1):
    # We redefine AR with numpy to avoid importing statsmodels
    if np.isclose(np.min(x), np.max(x)):
        # if x vector is flat, autocorrelation is not defined
        return np.nan
    else:
        return np.corrcoef(np.array([x[0 : len(x) - lag], x[lag : len(x)]]))[0, 1]

#def get_hurst(x):
#    np.std(x)

### Similarity stats


def get_similarity(x_obs, x_pred, method="nash-sutcliffe"):
    """ Timeseries Similarity factors

    :param x_obs:
    :param x_pred:
    :param method: Implemented: nash-sutcliffe (https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient)
        Distance correlation (https://en.wikipedia.org/wiki/Distance_correlation)
        dtw: https://en.wikipedia.org/wiki/Dynamic_time_warping
    :return:
    """
    if method == "euclidean":
        out = np.linalg.norm(x_obs - x_pred)
    elif method == "nash-sutcliffe":
        x_obs_mean = np.mean(x_obs)
        out = 1 - np.sum((x_pred - x_obs) ** 2) / np.sum((x_obs - x_obs_mean) ** 2)
    elif method == "dcorr":
        out = np.nan  # TODO !!
    elif method == "kolmogorov-smirnov":
        out = ks_2samp(x_obs, x_pred)[0]
    elif method == "dtw":
        out = np.nan  # TODO copy from https://github.com/talcs/simpledtw
    else:
        out = np.nan
    return np.round(out,3)

# TODO: 2d: parcorr

from functools import partial

## Templates of stat collection can be found below:
all_stats_desc = {
    "Sum": np.sum,
    "Average": get_mean,
    "Max": np.max,
    "Load Factor (peakiness)": get_lf,
    "Total Zero load duration": partial(get_dur_val, a=0),
    "Biggest duration of consecutive zero load": lambda x: np.max(get_rle(x, a=0)),
    "Ramps (98%)": partial(get_ramp_rates, a=98),
    "Min (2%)": partial(get_percentile, a=0.02, trunced=False),
    "Periodicity": lambda x: get_highest_periodicity(x)[0:2],
    "Autocorrelation(1)": partial(get_autocorr, lag=1),
    "Trend": get_trend,
    "Load ratio (max/min)": get_load_ratio,
}
# to add more...
