import pytest
import pandas as pd
import numpy as np
from cumulative_stats import (
    CumulativeStats,
    NaNPolicy,
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


class TestNaNPolicy:
    def test_nan_policy_enum_values(self):
        assert NaNPolicy.PRESERVE == 'preserve'
        assert NaNPolicy.FILL_FORWARD == 'fill_forward'

    def test_nan_policy_cumsum_preserve(self):
        data = [1, np.nan, 3, np.nan, 5]
        stats = CumulativeStats(data)
        result = stats.cumsum(nan_policy=NaNPolicy.PRESERVE)
        assert list(result['value'].fillna(-999)) == [1, -999, 4, -999, 9]

    def test_nan_policy_cumsum_fill_forward(self):
        data = [1, np.nan, 3, np.nan, 5]
        stats = CumulativeStats(data)
        result = stats.cumsum(nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [1, 2, 5, 8, 13]

    def test_nan_policy_cummax_fill_forward(self):
        data = [3, np.nan, 5, np.nan, 4]
        stats = CumulativeStats(data)
        result = stats.cummax(nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [3, 3, 5, 5, 5]

    def test_nan_policy_cumprod_fill_forward(self):
        data = [2, np.nan, 3, np.nan, 4]
        stats = CumulativeStats(data)
        result = stats.cumprod(nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [2, 4, 12, 36, 144]

    def test_nan_policy_string_input(self):
        data = [1, np.nan, 3]
        stats = CumulativeStats(data)
        result = stats.cumsum(nan_policy='preserve')
        assert pd.isna(result['value'].iloc[1])


class TestGroupedCumulative:
    def test_grouped_cumsum_basic(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby='group')
        assert list(result['value']) == [1, 3, 6, 4, 9, 15]

    def test_grouped_cumprod_basic(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        stats = CumulativeStats(df)
        result = stats.cumprod(groupby='group')
        assert list(result['value']) == [1, 2, 3, 12]

    def test_grouped_cummax_basic(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 4, 5, 2, 6]
        })
        stats = CumulativeStats(df)
        result = stats.cummax(groupby='group')
        assert list(result['value']) == [3, 3, 4, 5, 5, 6]

    def test_grouped_cummin_basic(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 4, 5, 2, 6]
        })
        stats = CumulativeStats(df)
        result = stats.cummin(groupby='group')
        assert list(result['value']) == [3, 1, 1, 5, 2, 2]

    def test_grouped_cumsum_with_nan_preserve(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'value': [1, np.nan, 3, 4, 5, np.nan, np.nan, 8]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby='group', nan_policy=NaNPolicy.PRESERVE)
        result_vals = list(result['value'].fillna(-999))
        assert result_vals == [1, -999, 4, 8, 5, -999, -999, 13]

    def test_grouped_cumsum_with_nan_fill_forward(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'value': [1, np.nan, 3, 4, 5, np.nan, np.nan, 8]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby='group', nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [1, 2, 5, 9, 5, 10, 15, 23]

    def test_grouped_cummax_with_nan_fill_forward(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, np.nan, 5, 2, np.nan, 4]
        })
        stats = CumulativeStats(df)
        result = stats.cummax(groupby='group', nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [3, 3, 5, 2, 2, 4]

    def test_grouped_cumprod_with_nan_fill_forward(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [2, np.nan, 3, 1, np.nan, 4]
        })
        stats = CumulativeStats(df)
        result = stats.cumprod(groupby='group', nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [2, 4, 12, 1, 1, 4]

    def test_grouped_skipna_false(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, np.nan, 3, 4, 5, 6]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby='group', skipna=False)
        result_vals = list(result['value'])
        assert result_vals[0] == 1
        assert pd.isna(result_vals[1])
        assert pd.isna(result_vals[2])
        assert result_vals[3] == 4
        assert result_vals[4] == 9
        assert result_vals[5] == 15

    def test_grouped_all_stats(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        stats = CumulativeStats(df)
        result = stats.all_stats(groupby='group')
        assert 'cumsum' in result
        assert 'cumprod' in result
        assert 'cummax' in result
        assert 'cummin' in result
        assert list(result['cumsum']['value']) == [1, 3, 3, 7]
        assert list(result['cumprod']['value']) == [1, 2, 3, 12]

    def test_grouped_multiple_columns(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'sales': [100, 200, 150, 250],
            'orders': [5, 8, 6, 10]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby='group')
        assert list(result['sales']) == [100, 300, 150, 400]
        assert list(result['orders']) == [5, 13, 6, 16]

    def test_grouped_multi_group_columns(self):
        df = pd.DataFrame({
            'region': ['N', 'N', 'N', 'S', 'S', 'S'],
            'product': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        stats = CumulativeStats(df)
        result = stats.cumsum(groupby=['region', 'product'])
        assert list(result['value']) == [1, 3, 3, 4, 5, 11]


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

    def test_compute_cumsum_grouped(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        result = compute_cumsum(df, groupby='group')
        assert list(result['value']) == [1, 3, 3, 7]

    def test_compute_cumsum_grouped_nan_fill_forward(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, np.nan, 3, 4, np.nan, 6]
        })
        result = compute_cumsum(df, groupby='group', nan_policy=NaNPolicy.FILL_FORWARD)
        assert list(result['value']) == [1, 2, 5, 4, 8, 14]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
