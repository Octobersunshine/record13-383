import pytest
import pandas as pd
import numpy as np
from cumulative_stats import (
    CumulativeStats,
    compute_cumsum,
    compute_cumprod,
    compute_cummax,
    compute_all_stats
)


class TestCumulativeStats:
    def test_init_with_list(self):
        data = [1, 2, 3, 4, 5]
        stats = CumulativeStats(data)
        result = stats.get_data()
        assert list(result['value']) == data

    def test_init_with_series(self):
        data = pd.Series([1, 2, 3, 4, 5], name='test')
        stats = CumulativeStats(data)
        result = stats.get_data()
        assert 'test' in result.columns
        assert list(result['test']) == [1, 2, 3, 4, 5]

    def test_init_with_dataframe(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        stats = CumulativeStats(data)
        result = stats.get_data()
        assert list(result.columns) == ['A', 'B']
        assert list(result['A']) == [1, 2, 3]

    def test_set_data(self):
        stats = CumulativeStats()
        stats.set_data([1, 2, 3])
        result = stats.get_data()
        assert list(result['value']) == [1, 2, 3]

    def test_get_data_raises_without_data(self):
        stats = CumulativeStats()
        with pytest.raises(ValueError, match="No data has been set"):
            stats.get_data()

    def test_cumsum_with_list(self):
        data = [1, 2, 3, 4, 5]
        stats = CumulativeStats(data)
        result = stats.cumsum()
        assert list(result['value']) == [1, 3, 6, 10, 15]

    def test_cumsum_with_dataframe(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        stats = CumulativeStats(data)
        result = stats.cumsum()
        assert list(result['A']) == [1, 3, 6]
        assert list(result['B']) == [4, 9, 15]

    def test_cumprod_with_list(self):
        data = [1, 2, 3, 4, 5]
        stats = CumulativeStats(data)
        result = stats.cumprod()
        assert list(result['value']) == [1, 2, 6, 24, 120]

    def test_cumprod_with_dataframe(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
        stats = CumulativeStats(data)
        result = stats.cumprod()
        assert list(result['A']) == [1, 2, 6]
        assert list(result['B']) == [2, 6, 24]

    def test_cummax_with_list(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        stats = CumulativeStats(data)
        result = stats.cummax()
        assert list(result['value']) == [3, 3, 4, 4, 5, 9, 9, 9]

    def test_cummax_with_dataframe(self):
        data = pd.DataFrame({'A': [5, 2, 7, 3], 'B': [3, 8, 1, 6]})
        stats = CumulativeStats(data)
        result = stats.cummax()
        assert list(result['A']) == [5, 5, 7, 7]
        assert list(result['B']) == [3, 8, 8, 8]

    def test_cummin(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        stats = CumulativeStats(data)
        result = stats.cummin()
        assert list(result['value']) == [3, 1, 1, 1, 1, 1, 1, 1]

    def test_all_stats(self):
        data = [1, 2, 3, 4, 5]
        stats = CumulativeStats(data)
        result = stats.all_stats()
        assert 'cumsum' in result
        assert 'cumprod' in result
        assert 'cummax' in result
        assert 'cummin' in result
        assert list(result['cumsum']['value']) == [1, 3, 6, 10, 15]
        assert list(result['cumprod']['value']) == [1, 2, 6, 24, 120]
        assert list(result['cummax']['value']) == [1, 2, 3, 4, 5]
        assert list(result['cummin']['value']) == [1, 1, 1, 1, 1]

    def test_with_nan_skipna_true(self):
        data = [1, np.nan, 3, np.nan, 5]
        stats = CumulativeStats(data)
        result = stats.cumsum(skipna=True)
        assert list(result['value'].fillna(0)) == [1, 0, 4, 0, 9]

    def test_with_nan_skipna_false(self):
        data = [1, np.nan, 3, np.nan, 5]
        stats = CumulativeStats(data)
        result = stats.cumsum(skipna=False)
        assert pd.isna(result['value'].iloc[1])
        assert pd.isna(result['value'].iloc[3])

    def test_axis_parameter_dataframe(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        stats = CumulativeStats(data)
        result = stats.cumsum(axis=1)
        assert list(result.iloc[0]) == [1, 5]
        assert list(result.iloc[1]) == [2, 7]
        assert list(result.iloc[2]) == [3, 9]


class TestConvenienceFunctions:
    def test_compute_cumsum(self):
        data = [1, 2, 3, 4, 5]
        result = compute_cumsum(data)
        assert list(result['value']) == [1, 3, 6, 10, 15]

    def test_compute_cumprod(self):
        data = [1, 2, 3, 4, 5]
        result = compute_cumprod(data)
        assert list(result['value']) == [1, 2, 6, 24, 120]

    def test_compute_cummax(self):
        data = [3, 1, 4, 1, 5]
        result = compute_cummax(data)
        assert list(result['value']) == [3, 3, 4, 4, 5]

    def test_compute_all_stats(self):
        data = [1, 2, 3, 4, 5]
        result = compute_all_stats(data)
        assert 'cumsum' in result
        assert 'cumprod' in result
        assert 'cummax' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
