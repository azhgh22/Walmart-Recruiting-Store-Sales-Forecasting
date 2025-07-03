import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'feature_engineering')))
from feature_engineering.grouper import group_and_aggregate
from feature_engineering import time_features, feature_transformers

class GroupStatModel(BaseEstimator, RegressorMixin):
    """
    A scikit-learn-compatible regressor that predicts using group-level mean of the target.
    """

    def __init__(self, store_model=None, dept_model=None, global_model=None, drop_markdown_flag=True):
      self.store_model = store_model or self._default_store_model()
      self.dept_model = dept_model or self._default_dept_model()
      self.global_model = global_model or self._default_global_model()
      self.drop_markdown_flag = drop_markdown_flag
      self.group_stat_adder = {}
      self.feature_adder = {}
      self.make_categorical_ = {}

    def _default_store_model(self):
      return XGBRegressor(
          objective='reg:squarederror',
          enable_categorical=True,
          random_state=42,
          n_estimators=200,
          learning_rate=0.1,
          max_depth=7,
          subsample=0.6,
          colsample_bytree=1.0,
          min_child_weight=5
      )

    def _default_dept_model(self):
      return XGBRegressor(
          objective='reg:squarederror',
          enable_categorical=True,
          random_state=42,
          n_estimators=300,
          learning_rate=0.1,
          max_depth=7,
          subsample=1.0,
          colsample_bytree=0.5,
          min_child_weight=1
      )

    def _default_global_model(self):
      return lgb.LGBMRegressor(
          objective='regression',
          random_state=42,
          verbose=-1,
          n_estimators=1000,
          learning_rate=0.1,
          max_depth=10
      )  

    def _transform_feature_adder(self, X, y=None, train=True, grp='Main'):
      params = {
          'add_week_num' : True,
          'add_holiday_flags' : True,
          'add_holiday_proximity': True,
          'add_holiday_windows': True,
          'add_fourier_features': True,
          'add_month_and_year': True,
          'replace_time_index': True,
      }
      if train:
        self.feature_adder[grp] = time_features.FeatureAdder(**params)
        return self.feature_adder[grp].fit_transform(X)
      else:
        return self.feature_adder[grp].transform(X)

    def _object_to_cat(self, X, y=None, train=True):
      if train:
        self.object_transformer = feature_transformers.ObjectToCategory()
        return self.object_transformer.fit_transform(X)
      else:
        return self.object_transformer.transform(X)

    def _group_stat_adder(self, X, y=None, train=True, group='Some'):
      if train:
        self.group_stat_adder[group] = feature_transformers.GroupStatFeatureAdder(groupby_cols=group,
                                                                                  aggfunc='mean')
        return self.group_stat_adder[group].fit_transform(X, y)
      else:
        return self.group_stat_adder[group].transform(X)

    def _make_categorical(self, X, y=None, train=True, feature="Some"):
      if train:
        self.make_categorical_[feature] = feature_transformers.MakeCategorical([feature])
        return self.make_categorical_[feature].fit_transform(X, y)
      else:
        return self.make_categorical_[feature].transform(X)

    def _drop_columns(self, X, y=None, train=True):
      if not self.drop_markdown_flag:
        return X
      columns_to_drop=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
      if train:
        self.drop_markdown = feature_transformers.ChangeColumns(columns_to_drop=columns_to_drop)
        return self.drop_markdown.fit_transform(X)
      else:
        return self.drop_markdown.transform(X)
  
    
    def _process_store_data(self, X, y=None, train=True):
      X_store, y_store = group_and_aggregate(X, y, groupby_cols=['Store', 'Date', 'IsHoliday'], y_aggfunc='mean')
      X_store = self._transform_feature_adder(X_store, y_store, train, grp='Store')
      X_store = self._object_to_cat(X_store, y_store, train)
      X_store = self._group_stat_adder(X_store, y_store, train, 'Store')
      X_store = self._make_categorical(X_store, y_store, train, 'Store')
      X_store = self._drop_columns(X_store, y_store, train)
      return X_store, y_store

    def _process_dept_data(self, X, y=None, train=True):
      X_dept, y_dept = group_and_aggregate(X, y, groupby_cols=['Date', 'IsHoliday', 'Dept'], y_aggfunc='mean')
      X_dept = self._transform_feature_adder(X_dept, y_dept, train, grp="Dept")
      X_dept = self._group_stat_adder(X_dept, y_dept, train, 'Dept')
      X_dept = self._object_to_cat(X_dept, y_dept, train)
      X_dept = self._make_categorical(X_dept, y_dept, train, 'Dept')
      return X_dept, y_dept

    def _process_data(self, X, y, train=True):
      X_ = self._transform_feature_adder(X, y, train)
      X_ = self._object_to_cat(X_, y, train)
      X_ = self._make_categorical(X_, y, train, 'Dept')
      X_ = self._make_categorical(X_, y, train, 'Store')
      return X_, y

    def fit(self, X, y):
        X_store, y_store = self._process_store_data(X, y, train=True)
        self.store_model.fit(X_store, y_store)
        store_pred = self.store_model.predict(X_store)

        store_results = X_store[['Store', 'Date']].copy()
        store_results['Store_Prediction'] = store_pred

        X_dept, y_dept = self._process_dept_data(X, y, train=True)
        self.dept_model.fit(X_dept, y_dept)
        dept_pred = self.dept_model.predict(X_dept)

        dept_results = X_dept[['Dept', 'Date']].copy()
        dept_results['Dept_Prediction'] = dept_pred

        X_, y = self._process_data(X, y)


        X_ = X_.merge(dept_results, on=['Date', 'Dept'], how='left')
        X_ = X_.merge(store_results, on=['Date', 'Store'], how='left')

        self.global_model.fit(X_, y)

        return self

    def predict(self, X):
        y = None

        X_store, y_store = self._process_store_data(X, y, train=False)
        store_pred = self.store_model.predict(X_store)

        store_results = X_store[['Store', 'Date']].copy()
        store_results['Store_Prediction'] = store_pred

        X_dept, y_dept = self._process_dept_data(X, y, train=False)
        dept_pred = self.dept_model.predict(X_dept)

        dept_results = X_dept[['Dept', 'Date']].copy()
        dept_results['Dept_Prediction'] = dept_pred

        X_, y = self._process_data(X, y, train=False)

        X_ = X_.merge(dept_results, on=['Date', 'Dept'], how='left')
        X_ = X_.merge(store_results, on=['Date', 'Store'], how='left')
        
        return self.global_model.predict(X_)
