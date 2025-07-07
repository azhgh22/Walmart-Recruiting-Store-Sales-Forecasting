from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class NaImputer(BaseEstimator, TransformerMixin):
  def __init__(self, strategy='mean'):
    self.strategy = strategy
    self.imputer = SimpleImputer(strategy=strategy)
    self.na_cols = []

  def fit(self, X, y=None):
    self.na_cols = [col for col in X.columns if X[col].isna().sum() > 0]
    self.imputer.fit(X[self.na_cols])
    return self

  def transform(self, X, y=None):
    x_copy = X.copy()
    x_copy[self.na_cols] = self.imputer.transform(x_copy[self.na_cols])
    return x_copy

class GroupMeanImputer(BaseEstimator, TransformerMixin):
  """
  Imputes missing values in target columns using group-wise means.

  Parameters:
  -----------
  group_cols : list of str
      Columns to group by (e.g., ['Store', 'Dept']).
  target_cols : list of str or None
      Target columns to impute. If None, automatically selects numeric columns with missing values.
  fallback : float
      Value to fill if group mean is not available (default is 0).
  """
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
          means = self.group_means_[[*self.group_cols, col]]
          X = X.merge(means, on=self.group_cols, how='left', suffixes=('', '_group_mean'),sort=False)
        
          X[col] = X[col].fillna(X[f'{col}_group_mean'])
          X[col] = X[col].fillna(self.fallback)
        
          X.drop(columns=[f'{col}_group_mean'], inplace=True)
      return X
