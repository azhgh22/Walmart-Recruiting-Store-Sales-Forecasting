TIME_FEATURE_PRESETS = {
    "default": [
        'Month', 'Year', 'WeekOfYear', 'HolidayFlags',
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
        'Days_until_next_holiday', 'Days_since_next_holiday',
        "Is_7_Days_Before_some_holiday", "Is_7_Days_After_Some_Holiday",
    ],
    "minimal": ['Month', 'Year'],
}



config = {
    'merge1': 'train, store, how=left, on=Store',
    'merge2': 'train, features, how=left, on=Store, Date, IsHoliday',
    'merged_tables': ['train', 'stores', 'features'],
    'time_features': TIME_FEATURE_PRESETS['default'],
    'add_dummy_date': False,
    'replace_time_index': True,
    'start_date': '2010-02-05',
    'score_metric': 'WMAE',
    'score_policy': {
        'weight on holidays': 5,
        'weight on non_holidays': 1
    },
}
