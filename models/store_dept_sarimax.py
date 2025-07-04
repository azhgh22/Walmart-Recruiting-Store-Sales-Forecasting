import warnings
import gc
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

class StoreDeptSARIMAX(BaseEstimator, RegressorMixin):
    """
    SARIMAX-based forecaster trained per (Store, Dept) group.

    Parameters:
    -----------
    order : tuple, default=(1,1,1)
        The (p, d, q) order of the ARIMA model.
    seasonal : bool, default=False
        Whether to use seasonal ARIMA.
    seasonal_order : tuple, default=(0,0,0,0)
        The (P, D, Q, s) order of the seasonal component.
    use_all_exog : bool, default=False
        If True, all non-key columns are treated as exogenous.
    verbose : int, default=1
        -1 = silent, 1 = log if Store or Dept divisible by 10, 2 = log all.
    filterwarnings : bool, default=False
        If True, suppress all warnings during model fitting.
    save_to_disk : bool, default=False
        If True, saves each fitted model to disk as 'model_Store_Dept.pkl'.
    """

    def __init__(self, order=(1,1,1), seasonal=False, seasonal_order=(0,0,0,0), 
                 use_all_exog=False, verbose=1, filterwarnings=False, save_to_disk=False):
        self.order = order
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.use_all_exog = use_all_exog
        self.exog_cols = None
        self.models_ = {}
        self.verbose = verbose
        self.filterwarnings = filterwarnings
        self.save_to_disk = save_to_disk

    def fit(self, X, y):
        data = X.copy()
        data["y"] = y.values
        self.models_ = {}

        if self.use_all_exog:
            exclude_cols = {"Store", "Dept", "Date"}
            self.exog_cols = [col for col in data.columns if col not in exclude_cols and col != "y"]
        else:
            self.exog_cols = None

        for (store, dept), group in data.groupby(["Store", "Dept"]):
            group = group.sort_values("Date")
            group['Date'] = pd.to_datetime(group['Date'])
            group = group.set_index('Date')
            group = group.asfreq('W')

            ts = group['y']
            if ts.dropna().empty:
                if self.verbose >= 1:
                    print(f"Skipping Store {store}, Dept {dept} due to empty or all NaN time series")
                continue

            exog = None
            if self.exog_cols:
                exog = group[self.exog_cols]

            model = SARIMAX(
                ts,
                order=self.order,
                seasonal_order=self.seasonal_order if self.seasonal else (0, 0, 0, 0),
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            if self.filterwarnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  
                    fitted_model = model.fit(disp=False)
            else:
                fitted_model = model.fit(disp=False)

            self.models_[(store, dept)] = fitted_model

            if self.save_to_disk:
                import joblib
                filename = f"model_store{store}_dept{dept}.pkl"
                joblib.dump(fitted_model, filename)

            if self.verbose == 2 or (self.verbose == 1 and (store % 10 == 0 or dept % 10 == 0)):
                print(f"Fitted SARIMAX model for Store {store}, Dept {dept}")

            del model, ts, exog, group
            gc.collect()

        return self

    def predict(self, X):
        if self.exog_cols is None and self.use_all_exog:
            exclude_cols = {"Store", "Dept", "Date"}
            self.exog_cols = [col for col in X.columns if col not in exclude_cols]

        preds = []
        for idx, row in X.iterrows():
            key = (row["Store"], row["Dept"])
            date = pd.to_datetime(row["Date"])
            store, dept = key

            if key not in self.models_:
                preds.append(float("nan"))
                continue

            model = self.models_[key]

            exog = None
            if self.exog_cols:
                exog = row[self.exog_cols].values.reshape(1, -1)

            try:
                pred_series = model.predict(start=date, end=date, exog=exog)
                if len(pred_series) == 0:
                    if self.verbose >= 1:
                        print(f"No prediction returned for Store {store}, Dept {dept} on {date}")
                    pred = float("nan")
                else:
                    pred = pred_series.iloc[0]
            except Exception as e:
                if self.verbose == 2:
                    print(f"Prediction error for Store {store}, Dept {dept} on {date}: {e}")
                pred = float("nan")

            preds.append(pred)

            if self.verbose == 2 or (self.verbose == 1 and (store % 10 == 0 or dept % 10 == 0)):
                print(f"Predicted value for Store {store}, Dept {dept} on {date}: {pred}")

            gc.collect()

        return pd.Series(preds, index=X.index)
