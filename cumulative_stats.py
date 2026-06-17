import pandas as pd
from typing import List, Union, Optional
import numpy as np


class CumulativeStats:
    def __init__(self, data: Optional[Union[List, pd.Series, pd.DataFrame]] = None):
        self._data = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data: Union[List, pd.Series, pd.DataFrame]) -> None:
        if isinstance(data, pd.DataFrame):
            self._data = data.copy()
        elif isinstance(data, pd.Series):
            self._data = data.to_frame()
        else:
            self._data = pd.DataFrame({'value': data})

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        return self._data.copy()

    def cumsum(self, axis: int = 0, skipna: bool = True) -> pd.DataFrame:
        data = self.get_data()
        return data.cumsum(axis=axis, skipna=skipna)

    def cumprod(self, axis: int = 0, skipna: bool = True) -> pd.DataFrame:
        data = self.get_data()
        return data.cumprod(axis=axis, skipna=skipna)

    def cummax(self, axis: int = 0, skipna: bool = True) -> pd.DataFrame:
        data = self.get_data()
        return data.cummax(axis=axis, skipna=skipna)

    def cummin(self, axis: int = 0, skipna: bool = True) -> pd.DataFrame:
        data = self.get_data()
        return data.cummin(axis=axis, skipna=skipna)

    def all_stats(self, axis: int = 0, skipna: bool = True) -> dict:
        return {
            'cumsum': self.cumsum(axis=axis, skipna=skipna),
            'cumprod': self.cumprod(axis=axis, skipna=skipna),
            'cummax': self.cummax(axis=axis, skipna=skipna),
            'cummin': self.cummin(axis=axis, skipna=skipna)
        }


def compute_cumsum(data: Union[List, pd.Series, pd.DataFrame],
              axis: int = 0,
              skipna: bool = True) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cumsum(axis=axis, skipna=skipna)


def compute_cumprod(data: Union[List, pd.Series, pd.DataFrame],
                   axis: int = 0,
                   skipna: bool = True) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cumprod(axis=axis, skipna=skipna)


def compute_cummax(data: Union[List, pd.Series, pd.DataFrame],
               axis: int = 0,
               skipna: bool = True) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cummax(axis=axis, skipna=skipna)


def compute_all_stats(data: Union[List, pd.Series, pd.DataFrame],
                     axis: int = 0,
                     skipna: bool = True) -> dict:
    stats = CumulativeStats(data)
    return stats.all_stats(axis=axis, skipna=skipna)
