import pandas as pd
from .filter import *

class StoreFilter(Filter):
    def __init__(self, store_id:int) -> None:
        self.store_id = store_id

    def get_query(self) -> str:
        return f'Store=={self.store_id}'