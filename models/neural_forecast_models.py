import pandas as pd
from neuralforecast import NeuralForecast
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralForecastModels(BaseEstimator, RegressorMixin):
    def __init__(self, models, model_names, freq='W-FRI', group_cols=['Store', 'Dept'], one_model=False, date_col='Date'):
        """
        Args:
            models: list of neuralforecast model instances
            model_names: list of names corresponding to models (must match forecast columns)
            freq: frequency of the time series
            group_cols: columns to form unique_id
            date_col: name of the datetime column
        """
        assert len(models) == len(model_names), "Each model must have a corresponding name."
        self.models = models
        self.model_names = model_names
        self.freq = freq
        self.group_cols = group_cols
        self.date_col = date_col
        self.nf = None
        self.fitted = False
        self.one_model=one_model

    def _prepare_df(self, X, y=None):
        df = X.copy()
        df['ds'] = df[self.date_col]
        df['unique_id'] = df[self.group_cols].astype(str).agg('-'.join, axis=1)
        if y is not None:
            df['y'] = y.values if isinstance(y, pd.Series) else y
            return df[['unique_id', 'ds', 'y']]
        else:
            return df[['unique_id', 'ds']]

    def fit(self, X, y):
        df = self._prepare_df(X, y)
        self.nf = NeuralForecast(models=self.models, freq=self.freq)
        self.nf.fit(df)
        self.fitted = True

    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        test_df = self._prepare_df(X_test)
        forecast = self.nf.predict()

        predictions = {}
        for name in self.model_names:
            merged = test_df.merge(
                forecast[['unique_id', 'ds', name]],
                on=['unique_id', 'ds'],
                how='left'
            )
            merged.fillna(0, inplace=True)
            predictions[name] = merged[name]

        if self.one_model:
            return predictions[self.model_names[0]]

        return predictions

    def forecast(self):
        if not self.fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.nf.predict()
