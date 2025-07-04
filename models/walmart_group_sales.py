from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from feature_engineering import feature_transformers, time_features
from models.group_stat import GroupStatModel  

class WalmartGroupSalesModel(GroupStatModel):
    def __init__(self):
        columns_to_drop=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

        store_pipeline = Pipeline([
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
            ('group_stat', feature_transformers.GroupStatFeatureAdder(groupby_cols='Store')),
            ('make_cat', feature_transformers.MakeCategorical(['Store'])),
            ('drop_markdowns', feature_transformers.ChangeColumns(columns_to_drop=columns_to_drop)),
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
            ('feature_adder', time_features.FeatureAdder(
                add_week_num=True,
                add_holiday_flags=True,
                add_holiday_proximity=True,
                add_holiday_windows=True,
                add_fourier_features=True,
                add_month_and_year=True,
                replace_time_index=True
            )),
            ('group_stat', feature_transformers.GroupStatFeatureAdder(groupby_cols='Dept')),
            ('object_to_cat', feature_transformers.ObjectToCategory()),
            ('make_cat', feature_transformers.MakeCategorical(['Dept'])),
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
            ('feature_adder', time_features.FeatureAdder(
                add_week_num=True,
                add_holiday_flags=True,
                add_holiday_proximity=True,
                add_holiday_windows=True,
                add_fourier_features=True,
                add_month_and_year=True,
                replace_time_index=True
            )),
            ('drop_markdowns', feature_transformers.ChangeColumns(columns_to_drop=columns_to_drop)),
            ('object_to_cat', feature_transformers.ObjectToCategory()),
            ('make_cat', feature_transformers.MakeCategorical(['Dept', 'Store'])),
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
