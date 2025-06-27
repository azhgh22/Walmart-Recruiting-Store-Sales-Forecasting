def difference_series(series, lag=1):
    return series.diff(lag).dropna()