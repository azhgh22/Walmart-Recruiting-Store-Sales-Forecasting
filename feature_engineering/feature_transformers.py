import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ObjectToCategory(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            return X
        X_ = X.copy()
        object_cols = X_.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                X_[col] = X_[col].astype('category')
        return X_

class GroupStatFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_cols, target_col=None, aggfunc='mean', feature_name=None):
        self.groupby_cols = [groupby_cols] if isinstance(groupby_cols, str) else groupby_cols
        self.target_col = target_col
        self.aggfunc = aggfunc
        self.feature_name = feature_name

    def fit(self, X, y=None):
        if self.target_col is not None:
            target = X[self.target_col]
        elif y is not None:
            target = pd.Series(y, index=X.index)
        else:
            raise ValueError("You must provide either `target_col` or `y`.")

        group_df = X[self.groupby_cols].copy()
        group_df["_target_"] = target.values
        grouped = group_df.groupby(self.groupby_cols, observed=True)["_target_"].agg(self.aggfunc).reset_index()

        new_feature_name = self.feature_name or f"{'_'.join(self.groupby_cols)}_{self.aggfunc}"
        grouped = grouped.rename(columns={"_target_": new_feature_name})

        self.grouped_result_ = grouped
        self.new_feature_name_ = new_feature_name

        return self

    def transform(self, X):
        return X.merge(self.grouped_result_, on=self.groupby_cols, how='left')
        
class MakeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype('category')
        return X_copy

class ChangeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, columns_to_keep=None):
        self.columns_to_drop = columns_to_drop
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        if self.columns_to_keep is not None:
            return X_copy[self.columns_to_keep]

        if self.columns_to_drop is not None:
            return X_copy.drop(columns=self.columns_to_drop, errors='ignore')

        return X_copy
