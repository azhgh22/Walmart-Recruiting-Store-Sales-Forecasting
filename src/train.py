import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class TimeSeriesForecaster(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for time series forecasting that handles
    autoregressive feature creation during prediction.

    Args:
        model: A scikit-learn compatible regressor (e.g., LGBMRegressor).
        lags (list): A list of integer lags to create for the target variable.
        rolling_windows (list): A list of integer window sizes for rolling stats.
        grouping_cols (list): List of columns to group by for feature creation
                              (e.g., ['Store', 'Dept']).
    """
    def __init__(self, model, lags: list, rolling_windows: list, grouping_cols: list):
        self.model = model
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.grouping_cols = grouping_cols
        
    def _create_historical_features(self, df, target_col='Weekly_Sales'):
        """Creates lag and rolling window features on a given dataframe."""
        df_out = df.copy()
        
        # Ensure data is sorted for correct time-based calculations
        df_out.sort_values(by=self.grouping_cols + ['Date'], inplace=True)

        # Create lag features
        for lag in self.lags:
            df_out[f'sales_lag_{lag}'] = df_out.groupby(self.grouping_cols)[target_col].shift(lag)

        # Create rolling window features
        # .shift(1) is crucial to prevent data leakage from the current week
        shifted_sales = df_out.groupby(self.grouping_cols)[target_col].shift(1)
        for w in self.rolling_windows:
            df_out[f'sales_rolling_mean_{w}'] = shifted_sales.rolling(window=w, min_periods=1).mean()
            df_out[f'sales_rolling_std_{w}'] = shifted_sales.rolling(window=w, min_periods=1).std()
            
        return df_out

    def fit(self, X, y):
        """
        Trains the model.

        Args:
            X (pd.DataFrame): Exogenous features for the training period.
            y (pd.Series): The target variable for the training period.
        """
        # 1. Combine features and target into a single DataFrame
        df_train = pd.concat([X, y], axis=1)
        
        # 2. Store the full training history for use in prediction later
        self.history_ = df_train.copy()
        
        # 3. Create historical features on the training data
        df_train_featured = self._create_historical_features(df_train, target_col=y.name)
        
        # 4. Prepare final training set by dropping rows with NaNs created by lags
        df_train_featured.dropna(inplace=True)
        
        X_train = df_train_featured.drop(columns=[y.name])
        y_train = df_train_featured[y.name]
      

        # 5. Train the underlying model
        self.model.fit(X_train, y_train)
        
        # 6. Store the feature names seen during training
        self.feature_names_in_ = X_train.columns.tolist()
        
        return self

    def predict(self, X_test):
        """
        Makes step-by-step autoregressive predictions.

        Args:
            X_test (pd.DataFrame): Exogenous features for the future prediction period.
        """
        # Check if the model has been fitted
        check_is_fitted(self)

        # Initialize history with the full training data
        history = self.history_.copy()
        
        # List to store the predictions
        predictions = []
        
        print(f"Starting autoregressive prediction for {len(X_test)} steps...")
        
        # Loop through each row of the test set (each time step)
        for i in range(len(X_test)):
            
            # 1. Get the exogenous features for the current step
            current_X_row = X_test.iloc[[i]]
            
            # 2. Combine history and the current step's features to create lags
            # This simulates having all data up to the point of prediction
            current_full_record = pd.concat([history, current_X_row], ignore_index=True)
            
            # 3. Create historical features on this combined data
            features_for_pred = self._create_historical_features(current_full_record, target_col=history.columns[-1])
            
            # 4. Isolate the very last row, which has the features for our current step
            last_row_features = features_for_pred.iloc[[-1]]
            
            # 5. Ensure the feature columns are in the same order as during training
            X_pred = last_row_features[self.feature_names_in_]
            
            # 6. Make a prediction
            pred = self.model.predict(X_pred)[0]
            
            # 7. Store the prediction
            predictions.append(pred)
            
            # 8. THE AUTOREGRESSIVE STEP: Update history with the prediction we just made
            # This prediction will be used to generate the lag for the *next* step.
            new_history_row = current_X_row.copy()
            new_history_row[history.columns[-1]] = pred # Add predicted value to the target column
            history = pd.concat([history, new_history_row], ignore_index=True)

        print("Prediction complete.")
        return np.array(predictions)