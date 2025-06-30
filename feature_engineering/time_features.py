import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

HOLIDAY_DATES = {
    # Super Bowl 
    '2010-02-12': 'SuperBowl', '2011-02-11': 'SuperBowl', '2012-02-10': 'SuperBowl', '2013-02-08': 'SuperBowl',
    # Labor Day
    '2010-09-10': 'LaborDay', '2011-09-09': 'LaborDay', '2012-09-07': 'LaborDay', '2013-09-06': 'LaborDay',
    # Thanksgiving
    '2010-11-26': 'Thanksgiving', '2011-11-25': 'Thanksgiving', '2012-11-23': 'Thanksgiving', '2013-11-29': 'Thanksgiving',
    # Christmas
    '2010-12-31': 'Christmas', '2011-12-30': 'Christmas', '2012-12-28': 'Christmas', '2013-12-27': 'Christmas',
}

class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer to engineer features for the Walmart sales forecasting problem.
    It can be configured to add different types of features.

    Args:
        add_week_num (bool): If True, adds the week of the year.
        add_holiday_flags (bool): If True, adds specific binary flags for each holiday.
        add_holiday_proximity (bool): If True, adds features for days until/since a holiday.
        add_holiday_windows (bool): If True, adds flags for weeks before/after holidays.
        add_fourier_features (bool): If True, adds sine/cosine features for cyclical time data.
        lags (list): A list of integer lags to create for the 'Weekly_Sales' column.
        rolling_windows (list): A list of integer window sizes for rolling statistics.
    """
    def __init__(self,
                 add_week_num=True,
                 add_holiday_flags=True,
                 add_holiday_proximity=True,
                 add_holiday_windows=True,
                 add_fourier_features=True,
                 lags=[52],
                 holiday_dates=HOLIDAY_DATES,
                 rolling_windows=[4, 12]):

        self.holiday_dates = holiday_dates
        self.add_week_num = add_week_num
        self.add_holiday_flags = add_holiday_flags
        self.add_holiday_proximity = add_holiday_proximity
        self.add_holiday_windows = add_holiday_windows
        self.add_fourier_features = add_fourier_features
        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['Date'] = pd.to_datetime(X_['Date'])

        if self.add_week_num:
            self._add_week_number(X_)

        if self.add_holiday_flags or self.add_holiday_proximity or self.add_holiday_windows:
            self._add_holiday_name_column(X_)

        if self.add_holiday_flags:
            self._add_specific_holiday_flags(X_)
            
        if self.add_holiday_proximity:
            self._add_proximity_to_holidays(X_)

        if self.add_holiday_windows:
            self._add_pre_post_holiday_windows(X_)
        
        if self.add_fourier_features and 'WeekOfYear' in X_.columns:
            self._add_month_and_year(X_)
            self._add_fourier_features(X_)
            
        if self.lags:
            self._add_lag_features(X_)
            
        if self.rolling_windows:
            self._add_rolling_window_features(X_)
            
        if 'HolidayName' in X_.columns:
            X_ = X_.drop(columns=['HolidayName'])
            
        return X_

    def _add_week_number(self, df):
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    def _add_month_and_year(self, df):
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

    def _add_holiday_name_column(self, df):
        df['HolidayName'] = df['Date'].dt.strftime('%Y-%m-%d').map(self.holiday_dates).fillna('NoHoliday')

    def _add_specific_holiday_flags(self, df):
        dummies = pd.get_dummies(df['HolidayName'], prefix='Is')
        if 'Is_NoHoliday' in dummies.columns:
            dummies = dummies.drop(columns=['Is_NoHoliday'])
        df[dummies.columns] = dummies

    def _add_proximity_to_holidays(self, df):
        holiday_dates = sorted([pd.to_datetime(d) for d in self.holiday_dates.keys()])
        indices = np.searchsorted(holiday_dates, df['Date'].values)

        next_holiday_dates = [holiday_dates[i] if i < len(holiday_dates) else pd.NaT for i in indices]
        df['Days_until_next_holiday'] = (pd.to_datetime(next_holiday_dates) - df['Date']).dt.days

        last_holiday_dates = [holiday_dates[i-1] if i > 0 else pd.NaT for i in indices]
        df['Days_since_last_holiday'] = (df['Date'] - pd.to_datetime(last_holiday_dates)).dt.days
        
        df.fillna({'Days_until_next_holiday': 999, 'Days_since_last_holiday': 999}, inplace=True)

    def _add_pre_post_holiday_windows(self, df):
        holiday_cols = [col for col in df.columns if col.startswith('Is_')]
        for col in holiday_cols:
            holiday_name = col.split('Is_')[1]
            df[f'Week_before_{holiday_name}'] = df.groupby('Store')[col].shift(-1).fillna(0)
            df[f'Week_after_{holiday_name}'] = df.groupby('Store')[col].shift(1).fillna(0)
    
    def _add_fourier_features(self, df):
        df['week_sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    def _add_lag_features(self, df):
        if 'Weekly_Sales' not in df.columns:
            return
        for lag in self.lags:
            df[f'sales_lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)

    def _add_rolling_window_features(self, df):
        if 'Weekly_Sales' not in df.columns:
            return

        for w in self.rolling_windows:
            shifted_sales = df.groupby('Store')['Weekly_Sales'].shift(1)
            df[f'sales_rolling_mean_{w}'] = shifted_sales.rolling(window=w, min_periods=1).mean()
            df[f'sales_rolling_std_{w}'] = shifted_sales.rolling(window=w, min_periods=1).std()



            