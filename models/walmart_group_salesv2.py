from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from feature_engineering import feature_transformers, time_features
from models.group_stat import GroupStatModel  

class GeneralWalmartGroupSalesModel(GroupStatModel):
    def __init__(self):
        columns_to_drop=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

        store_pipeline = Pipeline([
            ('group_stat', feature_transformers.GroupStatFeatureAdder(groupby_cols='Store')),
            ('model', XGBRegressor(
                objective='reg:squarederror',
                enable_categorical=True,
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                subsample=0.6,
                colsample_bytree=1.0,
                min_child_weight=5
            ))
        ])

        dept_pipeline = Pipeline([
            ('group_stat', feature_transformers.GroupStatFeatureAdder(groupby_cols='Dept')),
            ('model', XGBRegressor(
                objective='reg:squarederror',
                enable_categorical=True,
                random_state=42,
                n_estimators=300,
                learning_rate=0.1,
                max_depth=7,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight=1
            ))
        ])

        global_pipeline = Pipeline([
            ('model', LGBMRegressor(
                objective='regression',
                random_state=42,
                verbose=-1,
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=10
            ))
        ])

        super().__init__(
            store_pipeline=store_pipeline,
            dept_pipeline=dept_pipeline,
            global_pipeline=global_pipeline
        )
