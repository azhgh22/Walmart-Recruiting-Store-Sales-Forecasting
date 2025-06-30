import pandas as pd
from .filter import *

class DeptFilter(Filter):
    def __init__(self, dept_id:int) -> None:
        self.dept_id = dept_id

    def get_query(self) -> str:
        return f'Dept=={self.dept_id}'