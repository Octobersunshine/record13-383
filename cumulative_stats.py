import pandas as pd
from typing import List, Union, Optional, Any
import numpy as np
from enum import Enum


class NaNPolicy(str, Enum):
    PRESERVE = 'preserve'
    FILL_FORWARD = 'fill_forward'


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

    @staticmethod
    def _apply_nan_policy(result: pd.DataFrame,
                          original: pd.DataFrame,
                          nan_policy: NaNPolicy,
                          skipna: bool) -> pd.DataFrame:
        if nan_policy == NaNPolicy.PRESERVE:
            return result.where(~original.isna() | result.notna(), np.nan)
        elif nan_policy == NaNPolicy.FILL_FORWARD:
            return result.ffill()
        return result

    def _compute_grouped(self,
                         data: pd.DataFrame,
                         groupby: Union[str, List[str]],
                         method: str,
                         nan_policy: NaNPolicy,
                         skipna: bool,
                         value_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = data.copy()
        if value_cols is None:
            value_cols = [c for c in df.columns if c not in (groupby if isinstance(groupby, list) else [groupby])]

        result = df.copy()
        for col in value_cols:
            if nan_policy == NaNPolicy.FILL_FORWARD and skipna:
                grouped = df.groupby(groupby)[col].transform(
                    lambda x: x.ffill().cumsum() if method == 'cumsum' else
                              x.ffill().cumprod() if method == 'cumprod' else
                              x.ffill().cummax() if method == 'cummax' else
                              x.ffill().cummin()
                )
                result[col] = grouped
            elif nan_policy == NaNPolicy.PRESERVE and skipna:
                grouped = df.groupby(groupby)[col].transform(
                    lambda x: getattr(x, method)(skipna=True)
                )
                result[col] = grouped
            else:
                grouped = df.groupby(groupby)[col].transform(
                    lambda x: getattr(x, method)(skipna=False)
                )
                result[col] = grouped

        return result

    def _compute_single(self,
                        data: pd.DataFrame,
                        method: str,
                        axis: int,
                        nan_policy: NaNPolicy,
                        skipna: bool) -> pd.DataFrame:
        if nan_policy == NaNPolicy.FILL_FORWARD and skipna:
            data_filled = data.ffill(axis=axis)
            result = getattr(data_filled, method)(axis=axis, skipna=skipna)
        else:
            result = getattr(data, method)(axis=axis, skipna=skipna)
            if nan_policy == NaNPolicy.PRESERVE and skipna:
                result = result.where(~data.isna() | result.notna(), np.nan)
        return result

    def cumsum(self,
               axis: int = 0,
               skipna: bool = True,
               groupby: Optional[Union[str, List[str]]] = None,
               nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
        data = self.get_data()
        if isinstance(nan_policy, str):
            nan_policy = NaNPolicy(nan_policy)

        if groupby is not None:
            return self._compute_grouped(data, groupby, 'cumsum', nan_policy, skipna)
        return self._compute_single(data, 'cumsum', axis, nan_policy, skipna)

    def cumprod(self,
                axis: int = 0,
                skipna: bool = True,
                groupby: Optional[Union[str, List[str]]] = None,
                nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
        data = self.get_data()
        if isinstance(nan_policy, str):
            nan_policy = NaNPolicy(nan_policy)

        if groupby is not None:
            return self._compute_grouped(data, groupby, 'cumprod', nan_policy, skipna)
        return self._compute_single(data, 'cumprod', axis, nan_policy, skipna)

    def cummax(self,
               axis: int = 0,
               skipna: bool = True,
               groupby: Optional[Union[str, List[str]]] = None,
               nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
        data = self.get_data()
        if isinstance(nan_policy, str):
            nan_policy = NaNPolicy(nan_policy)

        if groupby is not None:
            return self._compute_grouped(data, groupby, 'cummax', nan_policy, skipna)
        return self._compute_single(data, 'cummax', axis, nan_policy, skipna)

    def cummin(self,
               axis: int = 0,
               skipna: bool = True,
               groupby: Optional[Union[str, List[str]]] = None,
               nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
        data = self.get_data()
        if isinstance(nan_policy, str):
            nan_policy = NaNPolicy(nan_policy)

        if groupby is not None:
            return self._compute_grouped(data, groupby, 'cummin', nan_policy, skipna)
        return self._compute_single(data, 'cummin', axis, nan_policy, skipna)

    def all_stats(self,
                  axis: int = 0,
                  skipna: bool = True,
                  groupby: Optional[Union[str, List[str]]] = None,
                  nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> dict:
        return {
            'cumsum': self.cumsum(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy),
            'cumprod': self.cumprod(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy),
            'cummax': self.cummax(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy),
            'cummin': self.cummin(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy)
        }


def compute_cumsum(data: Union[List, pd.Series, pd.DataFrame],
                   axis: int = 0,
                   skipna: bool = True,
                   groupby: Optional[Union[str, List[str]]] = None,
                   nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cumsum(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy)


def compute_cumprod(data: Union[List, pd.Series, pd.DataFrame],
                    axis: int = 0,
                    skipna: bool = True,
                    groupby: Optional[Union[str, List[str]]] = None,
                    nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cumprod(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy)


def compute_cummax(data: Union[List, pd.Series, pd.DataFrame],
                   axis: int = 0,
                   skipna: bool = True,
                   groupby: Optional[Union[str, List[str]]] = None,
                   nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> pd.DataFrame:
    stats = CumulativeStats(data)
    return stats.cummax(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy)


def compute_all_stats(data: Union[List, pd.Series, pd.DataFrame],
                      axis: int = 0,
                      skipna: bool = True,
                      groupby: Optional[Union[str, List[str]]] = None,
                      nan_policy: Union[str, NaNPolicy] = NaNPolicy.PRESERVE) -> dict:
    stats = CumulativeStats(data)
    return stats.all_stats(axis=axis, skipna=skipna, groupby=groupby, nan_policy=nan_policy)
