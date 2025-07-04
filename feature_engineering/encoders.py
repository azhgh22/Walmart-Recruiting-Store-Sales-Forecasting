import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.ohe_columns_encodings = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.columns:
            unique_vals = X[col].dropna().unique()
            self.ohe_columns_encodings[col] = [f"{col}_{val}" for val in unique_vals]
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            dummies = pd.get_dummies(
                X_transformed[col],
                prefix=col,
                dummy_na=True,
                dtype=int
            )
            for expected_col in self.ohe_columns_encodings[col]:
                if expected_col not in dummies.columns:
                    dummies[expected_col] = 0
            dummies = dummies[self.ohe_columns_encodings[col]]
            X_transformed = pd.concat([
                X_transformed.drop(col, axis=1),
                dummies
            ], axis=1)
        return X_transformed