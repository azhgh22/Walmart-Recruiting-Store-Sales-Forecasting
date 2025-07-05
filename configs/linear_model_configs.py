arima_config = {
    'order': (0, 1, 0),
    'use_all_exog': False,
    'filterwarnings': True,
    'verbose': 0
}

sarima_config = {
    "model_name": "StoreDeptSARIMAX",
    "order": (1, 1, 1),
    "seasonal": True,
    "seasonal_order": (1, 1, 1, 52), 
    "use_all_exog": False,
    "filterwarnings": True,
    "verbose": 0,
    "random_sampling": True, 
    "number_of_dept_store": 50,
}

sarimax_config = {
    "model_name": "StoreDeptSARIMAX",
    "order": (1, 1, 1),
    "seasonal": True,
    "seasonal_order": (1, 1, 1, 52), 
    "use_all_exog": False,
    "filterwarnings": True,
    "verbose": 0,
    "random_sampling": True,
    "number_of_dept_store": 30
}
