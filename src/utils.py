import numpy as np
import pandas as pd

def wmae(y_true, y_pred, is_holiday):
    """
    Compute the Weighted Mean Absolute Error (WMAE).

    Parameters:
    - y_true: array-like of true values
    - y_pred: array-like of predicted values
    - is_holiday: array-like of bools or 0/1, where 1 indicates a holiday week

    Returns:
    - WMAE (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.where(np.array(is_holiday), 5, 1)

    abs_errors = np.abs(y_true - y_pred)
    weighted_errors = weights * abs_errors

    return weighted_errors.sum() / weights.sum()


def difference_series(series, lag=1):
    return series.diff(lag).dropna()
