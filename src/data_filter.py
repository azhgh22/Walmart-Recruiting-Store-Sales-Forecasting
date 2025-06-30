import pandas as pd
from .filters.filter import Filter

class DataFilter:
    def filter(self, data:pd.DataFrame, filter:Filter) -> pd.DataFrame:
        return data.query(filter.get_query())