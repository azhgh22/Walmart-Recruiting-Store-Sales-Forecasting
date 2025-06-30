import pandas as pd
from .filter import *

class ComposeFilter(Filter):
    def __init__(self, filter_list:list[Filter]) -> None:
        self.filter_list = filter_list

    def get_query(self) -> str:
        return ' and '.join([filter.get_query() for filter in self.filter_list])