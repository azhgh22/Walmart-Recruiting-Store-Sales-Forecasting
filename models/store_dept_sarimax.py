import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

class StoreDeptSARIMAX(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1,1,1), seasonal=False, seasonal_order=(0,0,0,0), 
                 use_all_exog=False, verbose=False, filterwarnings=False):
        self.order = order
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.use_all_exog = use_all_exog
        self.exog_cols = None
        self.models_ = {}
        self.verbose = verbose
        self.filterwarnings = filterwarnings

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
            ts = group.set_index("Date")["y"]

            exog = None
            if self.exog_cols:
                exog = group.set_index("Date")[self.exog_cols]

            model = SARIMAX(
                ts,
                order=self.order,
                seasonal_order=self.seasonal_order if self.seasonal else (0,0,0,0),
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            if self.filterwarnings:
              with warnings.catch_warnings():
                  warnings.filterwarnings(
                      "ignore",
                      message="Too few observations to estimate starting parameters for seasonal ARMA.*"
                  )
                  self.models_[(store, dept)] = model.fit(disp=False)
            else:
                self.models_[(store, dept)] = model.fit(disp=False)

            if self.verbose:
                print(f"Fitted SARIMAX model for Store {store}, Dept {dept}")

        return self

    def predict(self, X):
        if self.exog_cols is None and self.use_all_exog:
            exclude_cols = {"Store", "Dept", "Date"}
            self.exog_cols = [col for col in X.columns if col not in exclude_cols]

        preds = []
        for idx, row in X.iterrows():
            key = (row["Store"], row["Dept"])
            date = row["Date"]

            if key not in self.models_:
                preds.append(float("nan"))
                continue

            model = self.models_[key]

            exog = None
            if self.exog_cols:
                exog = row[self.exog_cols].values.reshape(1, -1)

            try:
                pred = model.predict(start=date, end=date, exog=exog).iloc[0]
            except Exception as e:
                if self.verbose:
                    print(f"Prediction error for Store {key[0]}, Dept {key[1]} on {date}: {e}")
                pred = float("nan")

            preds.append(pred)

            if self.verbose:
                print(f"Predicted value for Store {key[0]}, Dept {key[1]} on {date}: {pred}")

        return pd.Series(preds, index=X.index)
