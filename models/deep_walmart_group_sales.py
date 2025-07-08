from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.pipeline import Pipeline
import torch
import logging

from neuralforecast.models import NBEATS, DLinear, PatchTST, TFT
from models.neural_forecast_models import NeuralForecastModels
from models.walmart_group_salesv2 import GeneralWalmartGroupSalesModel
from feature_engineering import time_features
from feature_engineering import feature_transformers

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("neuralforecast").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

class DeepWalmartGroupSales(BaseEstimator, RegressorMixin):
    def __init__(self):
        nbeats = NBEATS(
            max_steps=25 * 104,
            h=53,
            random_seed=42,
            input_size=52,
            batch_size=256,
            learning_rate=1e-3,
            shared_weights=True,
            optimizer=torch.optim.AdamW,
            activation='ReLU',
            enable_progress_bar=False
        )
        self.nbeats_model = NeuralForecastModels(
            models=[nbeats], model_names=['NBEATS'], freq='W-FRI', one_model=True
        )

        dlinear = DLinear(
            max_steps=25 * 104,
            h=53,
            random_seed=42,
            input_size=60,
            batch_size=512,
            learning_rate=1e-2,
            optimizer=torch.optim.Adagrad,
            scaler_type='robust',
            enable_progress_bar=False,
            enable_model_summary=False
        )
        self.dlinear_model = NeuralForecastModels(
            models=[dlinear], model_names=['DLinear'], freq='W-FRI', one_model=True
        )

        patchtst = PatchTST(
            input_size=52,
            dropout=0.2,
            h=53,
            max_steps=60 * 104,
            batch_size=64,
            random_seed=42,
            activation='relu',
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        self.patchtst_model = NeuralForecastModels(
            models=[patchtst], model_names=['PatchTST'], freq='W-FRI', one_model=True
        )

        tft = TFT(
            input_size=60,
            dropout=0.1,
            h=53,
            max_steps=20 * 104,
            random_seed=42,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        self.tft_model = NeuralForecastModels(
            models=[tft], model_names=['TFT'], freq='W-FRI', one_model=True
        )

        preprocess = Pipeline([
            ('feature_adder', time_features.FeatureAdder(
                add_week_num=True,
                add_holiday_flags=True,
                add_holiday_proximity=True,
                add_holiday_windows=True,
                add_fourier_features=True,
                add_month_and_year=True,
                replace_time_index=True
            )),
            ('object_to_cat', feature_transformers.ObjectToCategory()),
            ('week_of_year_avg', time_features.WeeklyStoreDept()),
            ('make_cat', feature_transformers.MakeCategorical(['Dept', 'Store'])),
            ('drop_markdowns', feature_transformers.ChangeColumns(columns_to_drop=[])),  # fill columns_to_drop as needed
        ])

        self.group_stat_pipeline = Pipeline([
            ('preprocess', preprocess),
            ('model', GeneralWalmartGroupSalesModel())
        ])

        self.models = [
            self.nbeats_model,
            self.dlinear_model,
            self.patchtst_model,
            self.tft_model,
            self.group_stat_pipeline
        ]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        # First four are deep learning models
        deep_preds = [model.predict(X) for model in self.models[:-1]]
        deep_avg = np.mean(deep_preds, axis=0)

        # Last model is the group statistical model
        group_stat_pred = self.models[-1].predict(X)

        # Final prediction: average of deep_avg and group_stat_pred
        final_pred = (deep_avg + group_stat_pred) / 2

        return final_pred
