import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mutual_info_score

def cross_correlation(series1, series2, max_lag=20):
    """
    Compute cross-correlation of two series for lags in [-max_lag, max_lag].
    Positive lag means series1 leads series2.
    """
    n = len(series1)
    assert len(series2) == n, "Series must be same length"

    ccf = []
    lags = range(-max_lag, max_lag+1)

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(series1[:lag], series2[-lag:])[0,1]
        elif lag > 0:
            corr = np.corrcoef(series1[lag:], series2[:-lag])[0,1]
        else:
            corr = np.corrcoef(series1, series2)[0,1]
        ccf.append(corr)

    return lags, ccf

def mutual_information(series1, series2, max_lag=20, bins=20):
    """
    Compute mutual information between series1 and series2 at lags -max_lag...max_lag.
    Series must be same length and aligned by date.
    bins: number of bins to discretize continuous values for MI calculation.
    """
    n = len(series1)
    mi_vals = []
    lags = range(-max_lag, max_lag+1)
    
    series1_binned = pd.cut(series1, bins=bins, labels=False)
    series2_binned = pd.cut(series2, bins=bins, labels=False)

    for lag in lags:
        if lag < 0:
            s1 = series1_binned[:lag]
            s2 = series2_binned[-lag:]
        elif lag > 0:
            s1 = series1_binned[lag:]
            s2 = series2_binned[:-lag]
        else:
            s1 = series1_binned
            s2 = series2_binned

        mi = mutual_info_score(s1, s2)
        mi_vals.append(mi)
    return lags, mi_vals

def check_stationarity(series):
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < 0.05

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_acf(series, seasonal_lag=52, plot_title=None, seasonal_decomposition=False):
    """
    Analyze autocorrelation of a time series with optional seasonal differencing,
    plot ACF and seasonal decomposition if enough data points.

    Parameters:
    - series: pd.Series of time series data (e.g., weekly sales)
    - seasonal_lag: int, seasonal lag period (default 52 for weekly data with yearly seasonality)
    - plot_title: str or None, title for ACF plot; if None, no plot shown
    """

    stationary = check_stationarity(series)

    if not stationary:
        series_diff = series.diff(seasonal_lag).dropna()
    else:
        series_diff = series

    max_lag = min(seasonal_lag, len(series_diff) - 1)
    if plot_title:
        plt.figure(figsize=(12, 4))
        plot_acf(series_diff, lags=max_lag)
        plt.title(plot_title)
        plt.grid(True)
        plt.show()

    if seasonal_decomposition:
      try:
          if len(series) >= 2 * seasonal_lag:
              decomposition = seasonal_decompose(series, model='additive', period=seasonal_lag)
              decomposition.plot()
              plt.suptitle('Seasonal Decomposition', fontsize=16)
              plt.show()
          else:
              print(f"Not enough data points for seasonal decomposition (need at least {2 * seasonal_lag}).")
      except Exception as e:
          print(f"Seasonal decomposition failed. Reason: {e}")

    return stationary, series_diff



def analyze_pacf(series, seasonal_lag=52, plot_title=None, seasonal_decomposition=False):
    """
    Analyze partial autocorrelation of a time series with optional seasonal differencing,
    plot PACF and seasonal decomposition if enough data points.

    Parameters:
    - series: pd.Series of time series data (e.g., weekly sales)
    - seasonal_lag: int, seasonal lag period (default 52 for weekly data with yearly seasonality)
    - plot_title: str or None, title for PACF plot; if None, no plot shown

    Returns:
    - stationary: bool, whether original series is stationary
    - series_diff: pd.Series, differenced series if non-stationary, otherwise original
    """
    
    stationary = check_stationarity(series)

    if not stationary:
        series_diff = series.diff(seasonal_lag).dropna()
    else:
        series_diff = series

    max_lag = min(seasonal_lag, len(series_diff) // 2)
    if plot_title:
        plt.figure(figsize=(12, 4))
        plot_pacf(series_diff, lags=max_lag, method='ywm')
        plt.title(plot_title)
        plt.grid(True)
        plt.show()

    if seasonal_decomposition:
      try:
          if len(series) >= 2 * seasonal_lag:
              decomposition = seasonal_decompose(series, model='additive', period=seasonal_lag)
              decomposition.plot()
              plt.suptitle('Seasonal Decomposition', fontsize=16)
              plt.show()
          else:
              print(f"Not enough data points for seasonal decomposition (need at least {2 * seasonal_lag}).")
      except Exception as e:
          print(f"Seasonal decomposition failed. Reason: {e}")

    return stationary, series_diff

def aggregate_dept_sales(data, dept_id):
    dept_data = data[data['Dept'] == dept_id]
    sales = dept_data.groupby('Date')['Weekly_Sales'].sum().sort_index()
    return sales