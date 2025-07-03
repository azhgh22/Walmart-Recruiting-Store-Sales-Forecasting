from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class GroupMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols=['Store', 'Dept'], target_cols=None, fallback=0):
        self.group_cols = group_cols
        self.target_cols = target_cols
        self.fallback = fallback
        self.group_means_ = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.target_cols is None:
            self.target_cols = X.select_dtypes(include='number').columns[
                X.isna().any()
            ].tolist()

        self.group_means_ = (
            X.groupby(self.group_cols)[self.target_cols]
            .mean()
            .reset_index()
        )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.target_cols:
            # Merge group means
            means = self.group_means_[[*self.group_cols, col]]
            X = X.merge(means, on=self.group_cols, how='left', suffixes=('', '_group_mean'),sort=False)

            # Fill NaN with group mean, then fallback
            X[col] = X[col].fillna(X[f'{col}_group_mean'])
            X[col] = X[col].fillna(self.fallback)

            # Drop helper column
            X.drop(columns=[f'{col}_group_mean'], inplace=True)
        return X


class LagAdder(BaseEstimator, TransformerMixin):
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