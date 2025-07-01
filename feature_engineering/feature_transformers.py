import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ObjectToCategory(BaseEstimator, TransformerMixin):

    """
    A scikit-learn compatible transformer that automatically converts all
    columns of dtype 'object' in a DataFrame to the 'category' dtype.
    
    This is particularly useful for preparing data for tree-based models like
    LightGBM or XGBoost that can handle the 'category' dtype efficiently.
    """
    
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