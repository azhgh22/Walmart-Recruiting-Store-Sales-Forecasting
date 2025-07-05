import pandas as pd
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.statespace.sarimax import SARIMAX

class StoreDeptSARIMAX(BaseEstimator, RegressorMixin):
    """
    SARIMAX-based forecaster trained per (Store, Dept) group.

    This model trains a separate SARIMAX model for each combination of
    'Store' and 'Dept' found in the training data. It is designed to be
    fully silent, suppressing all warnings and errors during model fitting
    and prediction. Failed models or predictions will result in NaN values.

    Parameters:
    -----------
    order : tuple, default=(1, 1, 1)
        The (p, d, q) order of the non-seasonal component of the ARIMA model.
    seasonal_order : tuple, default=(0, 0, 0, 0)
        The (P, D, Q, s) order of the seasonal component. Set s > 1 to
        use seasonal ARIMA.
    use_all_exog : bool, default=False
        If True, all columns in X other than 'Store', 'Dept', and 'Date'
        are treated as exogenous variables.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                 use_all_exog=False):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_all_exog = use_all_exog
        self.exog_cols = None
        self.models_ = {}

    def fit(self, X, y):
        """
        Fits a SARIMAX model for each (Store, Dept) group, suppressing all output.
        """
        data = X.copy()
        data["y"] = y

        if self.use_all_exog:
            exclude_cols = {"Store", "Dept", "Date", "y"}
            self.exog_cols = [col for col in data.columns if col not in exclude_cols]
            if not self.exog_cols: self.exog_cols = None

        for key, group in data.groupby(["Store", "Dept"]):
            group = group.sort_values("Date")
            group['Date'] = pd.to_datetime(group['Date'])
            group = group.set_index('Date')

            ts = group['y']
            if ts.dropna().empty: continue

            exog_data = group[self.exog_cols] if self.exog_cols else None

            # The key fix: The warning suppressor now wraps BOTH the model
            # initialization and the fitting process.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    # Initialize the model inside the suppressor
                    model = SARIMAX(
                        endog=ts, exog=exog_data, order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False
                    )
                    # Fit the model
                    fitted_model = model.fit(disp=False)
                    self.models_[key] = fitted_model
                except Exception:
                    # Silently skip any group that fails to initialize or fit
                    pass
        return self

    def predict(self, X):
        """
        Generates predictions for each row in X, suppressing all errors and warnings.
        """
        if not hasattr(self, 'models_') or not self.models_:
            raise NotFittedError("This model has not been fitted yet. Call fit() before predict().")

        X_pred = X.copy()
        X_pred['Date'] = pd.to_datetime(X_pred['Date'])
        
        predictions = pd.Series(index=X_pred.index, dtype=float)

        for key, group in X_pred.groupby(["Store", "Dept"]):
            if key in self.models_:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        model = self.models_[key]
                        group_for_pred = group.drop_duplicates(subset=['Date']).sort_values("Date").set_index("Date")
                        
                        if group_for_pred.empty: continue
                            
                        start_date = group_for_pred.index.min()
                        end_date = group_for_pred.index.max()
                        exog_data = group_for_pred[self.exog_cols] if self.exog_cols else None
                        
                        group_preds_by_date = model.predict(
                            start=start_date, end=end_date, exog=exog_data
                        )
                        
                        mapped_preds = group['Date'].map(group_preds_by_date)
                        mapped_preds.index = group.index
                        
                        predictions.update(mapped_preds)
                    except Exception:
                        # If prediction fails, the values remain NaN
                        pass
        
        return predictions