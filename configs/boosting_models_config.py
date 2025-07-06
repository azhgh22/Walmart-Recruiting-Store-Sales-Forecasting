store_avg_xgboost_config = {
    'objective': 'reg:squarederror',
    'enable_categorical': True,
    'random_state': 42,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 7,
    'subsample': 0.6,
    'colsample_bytree': 1.0,
    'min_child_weight': 5
}

dept_avg_xgboost_config = {
    'objective': 'reg:squarederror',
    'enable_categorical': True,
    'random_state': 42,
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 7,
    'subsample': 1.0,
    'colsample_bytree': 0.5,
    'min_child_weight': 1
}

lgbm_config = {
    'objective': 'regression',
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 10
}
