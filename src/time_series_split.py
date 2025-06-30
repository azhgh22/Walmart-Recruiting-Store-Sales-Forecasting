import pandas as pd
from datetime import datetime

class TimeSeriesSplit:
    def __init__(self, split_date:pd.Timestamp,start_date:pd.Timestamp=pd.Timestamp('2010-02-05')) -> None:
        self.split_date = split_date
        self.start_date = start_date

    def split(self, data:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
        # data['Date'] = pd.to_datetime(data['Date'])
        data = data.copy().sort_values(by='Date')
        data['DateDummy'] = ((data['Date'] - self.start_date).dt.days // 7).astype(int)
        train_data = data[data['Date']<=self.split_date]
        val_data = data[data['Date']>self.split_date]
        return train_data, val_data