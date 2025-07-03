from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from feature_engineering.imputers import GroupMeanImputer


class LagAdder(BaseEstimator, TransformerMixin):
 """
 Adds lag features to a time-series dataset and makes predictions step-by-step using a pre-fit model.

 Parameters:
 -----------
 val : pd.Series or pd.DataFrame
     Target variable values for training.
 model : object
     Machine learning model with `.fit()` and `.predict()` methods.
 lag_num : int
     Number of lag features to add (default is 2).
 date_col : str
    Name of the date column (default is 'DateDummy').
 """
    
  def __init__(self, val ,model ,lag_num:int = 2,date_col = 'DateDummy') -> None:
    super().__init__()
    self.lag_num = lag_num
    self.model = model
    self.y_val = val
    self.na_imputer = GroupMeanImputer()
    self.time = {}
    self.date_col = date_col

  def fit(self, x:pd.DataFrame, y:pd.DataFrame):
    x_ = x.copy()
    y_ = pd.DataFrame(y.copy())
    y_['Store'] = x_['Store']
    y_[self.date_col] = x_[self.date_col]
    y_['Dept'] = x_['Dept']
    for i in range(1,self.lag_num+1):
      x_[f'shift{i}'] = y_.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(i)

    self.na_imputer.fit(x_)
    x_ = self.na_imputer.transform(x_)
    self.model.fit(x_,y)

    self.time = {}
    dates = sorted(x[self.date_col].unique())
    for i in dates:
      cur_data = y_[y_[self.date_col] == i][['Store', 'Dept','Weekly_Sales']].copy()
      self.time[i] = cur_data
    return self

  def transform(self, x:pd.DataFrame):
    answer = x.copy()
    answer['WeeklySales'] = 0.0
    x_ = x.copy()

    dates = sorted(x[self.date_col].unique())

    time = self.time.copy()

    for i in dates:
      # print(i)
      cur_data = x_[x_[self.date_col] == i].copy()
      for j in range(1,self.lag_num+1):
        if time.get(i-j) is not None:
          cur_data[f'shift{j}'] = pd.merge(cur_data,time[i-j],how='left',on=['Store','Dept'],sort=False)['Weekly_Sales']
        else:
          cur_data[f'shift{j}'] = np.nan
          print(i)
      
      cur_data = self.na_imputer.transform(cur_data)
      assert cur_data.isna().sum().sum() == 0
      pred = self.model.predict(cur_data)
      assert len(answer.loc[answer[self.date_col] == i,'WeeklySales'])==len(pred)
      answer.loc[answer[self.date_col] == i,'WeeklySales'] = pred.copy()
      t_i = x_.loc[x_[self.date_col] == i][['Store','Dept',self.date_col]]
      t_i['Weekly_Sales'] = pred.copy()
      time[i] = t_i

    return answer['WeeklySales']

  def predict(self, x:pd.DataFrame):
    return self.transform(x)
