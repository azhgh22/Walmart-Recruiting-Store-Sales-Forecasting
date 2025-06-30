from .filters.compose_filter import *
from .filters.store_filter import *
from .filters.dept_filter import *
from .data_filter import DataFilter
from .time_series_split import TimeSeriesSplit

from functools import reduce
class DeptKeeper:
    def __init__(self, data:pd.DataFrame,split_date:pd.Timestamp,dept_id:int) -> None:
        self.train_mapping:dict[int,pd.DataFrame] = {}
        self.val_mapping:dict[int,pd.DataFrame] = {}
        for store in data['Store'].unique():
            compose_filter = ComposeFilter([StoreFilter(store),DeptFilter(dept_id)])
            filtered = DataFilter().filter(data, compose_filter)
            train_part, val_part = TimeSeriesSplit(split_date).split(filtered)
            self.train_mapping[store] = train_part
            self.val_mapping[store] = val_part

    def get_train(self,store_id:int) -> pd.DataFrame:
        return self.train_mapping[store_id].copy()

    def get_val(self,store_id:int) -> pd.DataFrame:
        return self.val_mapping[store_id].copy()

    def get_train_keys(self) -> list[int]:
        return list(self.train_mapping.keys())

    def get_val_keys(self) -> list[int]:
        return list(self.val_mapping.keys())

    def get_train_avarage(self) -> pd.DataFrame:
        dfs = [self.get_train(s)[['Date','Weekly_Sales']] for s in self.get_train_keys()]
        return self.__calc_avarage(dfs)

    def get_val_avarage(self) -> pd.DataFrame:
        dfs = [self.get_val(s)[['Date','Weekly_Sales']] for s in self.get_val_keys()]
        return self.__calc_avarage(dfs)

    def __calc_avarage(self,data:pd.DataFrame) -> pd.DataFrame:
        for i, df in enumerate(data):
            df.rename(columns={'Weekly_Sales': f'Sales{i+1}'}, inplace=True)
            df.set_index('Date',inplace=True)

        merged = reduce(lambda left, right: left.join(right, how='outer'), data)

        merged['Avg_Sales'] = merged[[col for col in merged.columns if col.startswith('Sales')]].mean(axis=1,skipna=True)

        merged.reset_index(inplace=True)

        return merged