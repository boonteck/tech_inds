# External Dependencies
import numpy as np
import numba

@numba.jit(nopython = True)
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# rolling standard deviation
def stdev(arr, period, ddof = 1):
    rolling_windows = rolling_window(arr,period).std(axis = 1, ddof = ddof) # corrected sample standard deviation
    results = np.concatenate((np.array([np.nan] * (period-1)), rolling_windows))
    return(results)

# Simple Moving Average
def SMA(arr, period):
    results = rolling_window(arr,period).mean(axis = 1)
    results = np.concatenate((np.array([np.nan] * (period-1)), results))
    return(results)

# Bollinger Bands
def Boll(dfHigh, dfLow, dfClose, period = 20, stdev_dist = 2):
    typicalPrice = (dfHigh + dfClose + dfLow) / 3
    middle = SMA(typicalPrice, period)
    stdev = stdev(typicalPrice, period, ddof = 0)
    upper = middle + (stdev_dist * stdev)
    lower = middle - (stdev_dist * stdev)
    return ((upper, middle, lower))

# Simple Average - With numba optimization
@numba.jit(nopython = True)
def SMA_numba(arr, period):
    results = np.zeros(len(arr))
    for i in range(period - 1, len(results)):
        window = arr[i - period + 1 : i + 1]
        results[i] = np.sum(window)
    results[:period-1] = np.nan
    return(results / period)

# Exponential Moving Average
@numba.jit(nopython = True)
def EMA(arr,period, alpha = False):
    if (alpha == True):
        alpha = 1 / period
    elif (alpha == False):
        alpha = 2 / (period + 1)

    exp_weights = np.zeros(len(arr))
    exp_weights[period - 1] = np.mean(arr[:period])
    for i in range(period,len(exp_weights) + 1):
        exp_weights[i] = exp_weights[i-1]*(1-alpha) + ((alpha)*(arr[i]))
    exp_weights[:period-1] = np.nan
    return(exp_weights)

# Average True Range
@numba.jit(nopython = True)
def ATR(dfHigh, dfLow, dfClose, period = 14):
    high_minus_low = (dfHigh - dfLow)[1:]
    high_minus_close_prev = np.abs(dfHigh[1:] - dfClose[:-1])
    low_minus_close_prev = np.abs(dfLow[1:] - dfClose[:-1])
    arr = np.zeros(len(high_minus_low))
    for i in range(len(high_minus_low)):
        arr[i] = max(high_minus_low[i],
                     high_minus_close_prev[i],
                     low_minus_close_prev[i]
                    )
    results = EMA(arr, period, alpha = True)
    return (results)

# Moving Average Convergence Divergence
@numba.jit(nopython = True)
def MACD(arr, fast_period=12, slow_period=26, signal_period=9, percent = True):
    fast_EMA = EMA(arr, fast_period)
    slow_EMA = EMA(arr,slow_period)
    macd = fast_EMA - slow_EMA
    if (percent):
        macd *= 100 / slow_EMA
    signal = np.concatenate((macd[:slow_period-1],EMA(macd[slow_period-1:],signal_period)))
    hist = macd - signal
    return((macd, signal, hist))

# Relative Strength Index
@numba.jit(nopython = True)
def RSI(arr, period = 21):
    delta = np.diff(arr)
    up, down = np.copy(delta), np.copy(delta)
    up[up < 0] = 0
    down[down > 0] = 0

    # Exponential Weighted windows mean with centre of mass = period - 1 -> alpha = 1 / (period)
    alpha = 1 / (period)
    rUp = EMA(up, period, alpha = alpha)
    rDown = np.abs(EMA(down, period, alpha = alpha))
    result = 100 - (100 / (1+ rUp / rDown))

    #append nan that was lost in np.diff
    result = np.concatenate((np.array([np.nan]), result))
    return(result)

# Commodity Channel Index
@numba.jit()
def CCI(dfHigh, dfLow, dfClose, period = 20, scaling = 0.015):
    '''
    Similar to TTR package in R, central tendency measure uses mean
    '''
    typicalPrice = (dfHigh + dfClose + dfLow) / 3
    rolling_windows = rolling_window(typicalPrice,period)
    central_tendency_arr = SMA_numba(typicalPrice, period)[period - 1:]
    abs_deviation_arr = np.abs((rolling_windows.T - central_tendency_arr).T)
    mean_abs_deviation = np.zeros(len(abs_deviation_arr))

    # once numba has a way of reducing along axes, can switch this away
    for i in range(len(rolling_windows)):
        mean_abs_deviation[i] = np.mean(abs_deviation_arr[i])

    result = (typicalPrice[period-1:] - central_tendency_arr) / (mean_abs_deviation * scaling)
    result = np.concatenate((np.array([np.nan] * (period-1)), result))
    return(result)

# Stochastic Momentum Indicator
@numba.jit(nopython = True)
def SMI(dfHigh, dfLow, dfClose, period = 13, fast_period = 2, slow_period = 25, signal_period = 9):
    rolling_high = rolling_window(dfHigh,period)
    rolling_low = rolling_window(dfLow,period)

    centre = np.zeros(len(rolling_high))
    HL_diff = np.copy(centre)
    for i in range(len(centre)):
        centre[i] = (np.max(rolling_high[i]) + np.min(rolling_low[i])) / 2
        HL_diff[i] = np.max(rolling_high[i]) - np.min(rolling_low[i])

    centre = np.concatenate((((dfHigh[:period-1] + dfLow[:period-1]) / 2),centre))
    HL_diff = np.concatenate(((dfHigh[:period-1] - dfLow[:period-1]),HL_diff))

    c_diff = dfClose - centre
    num1 = EMA(c_diff,slow_period)
    den1 = EMA(HL_diff,slow_period)
    num2 = EMA(num1[slow_period-1:], fast_period)
    den2 = EMA(den1[slow_period-1:], fast_period) / 2
    SMI = 100 * (num2 / den2)
    signal = EMA(SMI[1:], signal_period)
    SMI = np.concatenate((np.array([np.nan] * (slow_period-1)), SMI))
    signal = np.concatenate((np.array([np.nan] * (slow_period)), signal))
    return((SMI,signal))

# Rate of Change
@numba.jit(nopython = True)
def ROC(arr, period = 1, continuous_compound = True):
    if (continuous_compound is True):
        result = np.log(arr[period:] / arr[:-period]) * 100
    else:
        result = (arr[period:] - arr[:-period]) /arr[:-period] * 100
    result = np.concatenate((np.array([np.nan] * period), result))
    return(result)
