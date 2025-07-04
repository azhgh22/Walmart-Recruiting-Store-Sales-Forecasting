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
        add_month_and_year (bool): If True, adds Month and Year features for time data.
        list_of_holiday_proximity (list): If non empty, does same as add_holiday_proximity, but for spacific holiday. 
    """
    def __init__(self,
                 add_week_num=True,
                 add_holiday_flags=True,
                 add_holiday_proximity=True,
                 add_holiday_windows=False,
                 add_fourier_features=True,
                 add_month_and_year=True,
                 list_of_holiday_proximity=list(set(HOLIDAY_DATES.values())),
                 holiday_dates=HOLIDAY_DATES,
                 replace_time_index = True,
                 add_dummy_date = False,
                 start_date = None
                 ):

        self.holiday_dates = holiday_dates
        self.add_week_num = add_week_num
        self.add_month_and_year = add_month_and_year
        self.add_holiday_flags = add_holiday_flags
        self.add_holiday_proximity = add_holiday_proximity
        self.add_holiday_windows = add_holiday_windows
        self.add_fourier_features = add_fourier_features
        self.list_of_holiday_proximity = list_of_holiday_proximity
        self.replace_time_index = replace_time_index
        self.add_dummy_date = add_dummy_date
        self.start_date = start_date

    def fit(self, X, y=None):
        if self.start_date is not None:
          self.start_date_ = self.start_date
        else:
          self.start_date_ = pd.to_datetime(X['Date']).min()
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Date'] = pd.to_datetime(X_['Date'])

        if self.add_dummy_date:
          X_['DateDummy'] = ((X_['Date'] - self.start_date_).dt.days // 7).astype(int)

        if self.add_month_and_year or self.add_fourier_features:
          self._add_month_and_year(X_)

        if self.add_week_num:
            self._add_week_number(X_)

        if self.add_holiday_flags:
            self._add_specific_holiday_flags(X_)
            
        if self.add_holiday_proximity:
            self._add_proximity_to_holidays(X_)

        if self.add_holiday_windows:
            self._add_pre_post_holiday_windows(X_)
        
        if self.add_fourier_features and 'WeekOfYear' in X_.columns:
            self._add_fourier_features(X_)

        if self.list_of_holiday_proximity:
            self._add_proximity_to_specific_holidays(X_)

        if self.replace_time_index:
          self._replace_date_with_time_index(X_)
            
        return X_

    def _add_week_number(self, df):
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    def _add_month_and_year(self, df):
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

    def _add_specific_holiday_flags(self, df):
      date_str = df['Date'].dt.strftime('%Y-%m-%d')

      for holiday_name in set(self.holiday_dates.values()):
          holiday_dates = {
              date for date, name in self.holiday_dates.items() if name == holiday_name
          }
          df[f'Is_{holiday_name}'] = date_str.isin(holiday_dates).astype(int)

    def _replace_date_with_time_index(self, df):
        if self.start_date_ is None:
            raise RuntimeError("The transformer has not been fitted yet. Call .fit() before .transform().")
        dates = pd.to_datetime(df['Date'])
        time_delta_days = (dates - self.start_date_).dt.days
        df.drop(columns=['Date'], inplace=True)
        df['Date'] = (time_delta_days / 7).astype(int)

    def _add_proximity_to_holidays(self, df):
      holiday_dates = sorted([pd.to_datetime(d) for d in self.holiday_dates.keys()])
      safe_dates = pd.to_datetime(df['Date'], errors='coerce')
      indices = np.searchsorted(holiday_dates, safe_dates)

      next_holiday_dates = [holiday_dates[i] if i < len(holiday_dates) else pd.NaT for i in indices]
      df['Days_until_next_holiday'] = (pd.to_datetime(next_holiday_dates) - df['Date']).dt.days

      last_holiday_dates = [holiday_dates[i-1] if i > 0 else pd.NaT for i in indices]
      df['Days_since_last_holiday'] = (df['Date'] - pd.to_datetime(last_holiday_dates)).dt.days
      
      df.fillna({'Days_until_next_holiday': 999, 'Days_since_last_holiday': 999}, inplace=True)

    def _add_proximity_to_specific_holidays(self, df):
      safe_dates = pd.to_datetime(df['Date'], errors='coerce')
      for holiday in self.list_of_holiday_proximity:
        holiday_dates = sorted([pd.to_datetime(d) for d, name in self.holiday_dates.items() if name == holiday])
        if len(holiday_dates) == 0:
          continue
        indices = np.searchsorted(holiday_dates, safe_dates)

        next_holiday_dates = [holiday_dates[i] if i < len(holiday_dates) else pd.NaT for i in indices]
        df[f'Days_until_next_{holiday}'] = (pd.to_datetime(next_holiday_dates) - df['Date']).dt.days

        last_holiday_dates = [holiday_dates[i-1] if i > 0 else pd.NaT for i in indices]
        df[f'Days_since_last_{holiday}'] = (df['Date'] - pd.to_datetime(last_holiday_dates)).dt.days
        
        df.fillna({f'Days_until_next_{holiday}': 999, f'Days_since_last_{holiday}': 999}, inplace=True)

    def _add_pre_post_holiday_windows(self, df):
      unique_holidays = set(self.holiday_dates.values())
      for holiday_name in unique_holidays:
          holiday_specific_dates = pd.to_datetime([
              date_str for date_str, name in self.holiday_dates.items() if name == holiday_name
          ])
          
          df[f'Is_7_Days_Before_{holiday_name}'] = 0
          df[f'Is_7_Days_After_{holiday_name}'] = 0

          for holiday_date in holiday_specific_dates:
              before_mask = (df['Date'] >= holiday_date - pd.Timedelta(days=7)) & (df['Date'] < holiday_date)
              after_mask = (df['Date'] > holiday_date) & (df['Date'] <= holiday_date + pd.Timedelta(days=7))
              df.loc[before_mask, f'Is_7_Days_Before_{holiday_name}'] = 1
              df.loc[after_mask, f'Is_7_Days_After_{holiday_name}'] = 1
    
    def _add_fourier_features(self, df):
        df['week_sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)



            