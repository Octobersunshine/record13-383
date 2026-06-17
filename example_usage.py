import pandas as pd
import numpy as np
from cumulative_stats import (
    CumulativeStats,
    NaNPolicy,
    compute_cumsum,
    compute_cumprod,
    compute_cummax,
    groupby_compute_cumsum,
    groupby_compute_cumprod,
    groupby_compute_cummax,
    groupby_compute_all_stats
)

print("=" * 75)
print(" 分组累计功能使用示例")
print("=" * 75)

print("\n【示例 1】基础分组累计和 - 每个分组内独立累计")
print("-" * 60)
df1 = pd.DataFrame({
    '部门': ['销售部', '销售部', '销售部', '技术部', '技术部', '技术部'],
    '月度业绩': [100, 200, 150, 80, 120, 90]
})
print("原始数据:")
print(df1.to_string(index=False))

stats1 = CumulativeStats(df1)
result1 = stats1.groupby_cumsum(groupby='部门')
print("\n按部门分组累计和 (每个部门独立累计):")
print(result1.to_string(index=False))

print("\n【示例 2】分组累计 + 空值处理 (两种策略)")
print("-" * 60)
df2 = pd.DataFrame({
    '部门': ['销售部', '销售部', '销售部', '销售部',
             '技术部', '技术部', '技术部', '技术部'],
    '月度业绩': [100, np.nan, 150, 200, 80, np.nan, np.nan, 120]
})
print("原始数据 (含空值):")
print(df2.to_string(index=False))

stats2 = CumulativeStats(df2)

print("\n  2.1 PRESERVE 策略 (保留空值位置，累计继续):")
result2a = stats2.groupby_cumsum(groupby='部门', nan_policy=NaNPolicy.PRESERVE)
print(result2a.to_string(index=False))

print("\n  2.2 FILL_FORWARD 策略 (前值填充，累计不中断):")
result2b = stats2.groupby_cumsum(groupby='部门', nan_policy=NaNPolicy.FILL_FORWARD)
print(result2b.to_string(index=False))

print("\n【示例 3】多种分组累计统计 (累计乘积/最大值/最小值)")
print("-" * 60)
df3 = pd.DataFrame({
    '股票': ['AAPL', 'AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'GOOGL'],
    '日收益率': [0.05, -0.02, 0.03, 0.04, 0.01, -0.01]
})
df3['日收益倍数'] = 1 + df3['日收益率']
print("原始数据 (股票日收益):")
print(df3.to_string(index=False))

stats3 = CumulativeStats(df3)

print("\n  3.1 分组累计乘积 (累计收益倍数):")
result3a = stats3.groupby_cumprod(groupby='股票', value_cols='日收益倍数')
print(result3a.to_string(index=False))

print("\n  3.2 分组累计最大值 (历史最高收益倍数):")
result3b = stats3.groupby_cummax(groupby='股票', value_cols='日收益倍数')
print(result3b.to_string(index=False))

print("\n  3.3 分组累计最小值 (历史最低收益倍数):")
result3c = stats3.groupby_cummin(groupby='股票', value_cols='日收益倍数')
print(result3c.to_string(index=False))

print("\n【示例 4】多列分组 (按区域+产品维度分组)")
print("-" * 60)
df4 = pd.DataFrame({
    '月份': ['1月', '1月', '1月', '1月', '2月', '2月', '2月', '2月'],
    '区域': ['华东', '华东', '华南', '华南', '华东', '华东', '华南', '华南'],
    '产品': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    '销量': [100, 200, 150, 180, 120, 220, 160, 200]
})
print("原始数据 (多维度销售数据):")
print(df4.to_string(index=False))

stats4 = CumulativeStats(df4)
result4 = stats4.groupby_cumsum(groupby=['区域', '产品'])
print("\n按【区域+产品】分组累计销量:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(result4.to_string(index=False))
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

print("\n【示例 5】选择特定列进行累计 (value_cols)")
print("-" * 60)
df5 = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=6, freq='D'),
    '区域': ['华东', '华东', '华东', '华南', '华南', '华南'],
    '销售额': [1000, 1500, 1200, 800, 900, 1100],
    '订单数': [10, 15, 12, 8, 9, 11],
    '利润': [200, 300, 240, 160, 180, 220]
})
print("原始数据 (多列指标):")
print(df5.to_string(index=False))

stats5 = CumulativeStats(df5)

print("\n  5.1 只对【销售额】进行累计，其他列保持不变:")
result5a = stats5.groupby_cumsum(groupby='区域', value_cols='销售额')
print(result5a.to_string(index=False))

print("\n  5.2 对【销售额, 订单数】进行累计，【利润】不变:")
result5b = stats5.groupby_cumsum(groupby='区域', value_cols=['销售额', '订单数'])
print(result5b.to_string(index=False))

print("\n【示例 6】一次性获取所有分组统计结果 (groupby_all_stats)")
print("-" * 60)
df6 = pd.DataFrame({
    '小组': ['甲', '甲', '甲', '乙', '乙', '乙'],
    '得分': [85, 90, 88, 78, 82, 80]
})
print("原始数据 (小组得分):")
print(df6.to_string(index=False))

stats6 = CumulativeStats(df6)
result6 = stats6.groupby_all_stats(groupby='小组')
for name, res in result6.items():
    print(f"\n  分组{name}:")
    print(res.to_string(index=False))

print("\n【示例 7】便捷函数方式调用 (groupby_compute_*)")
print("-" * 60)
df7 = pd.DataFrame({
    '类别': ['X', 'X', 'Y', 'Y'],
    '价值': [10, 20, 30, 40]
})
print("原始数据:")
print(df7.to_string(index=False))

print("\n  7.1 groupby_compute_cumsum (分组累计和):")
res7a = groupby_compute_cumsum(df7, groupby='类别')
print(res7a.to_string(index=False))

print("\n  7.2 groupby_compute_cumprod (分组累计乘积):")
res7b = groupby_compute_cumprod(df7, groupby='类别')
print(res7b.to_string(index=False))

print("\n  7.3 groupby_compute_cummax (分组累计最大值):")
res7c = groupby_compute_cummax(df7, groupby='类别')
print(res7c.to_string(index=False))

print("\n【示例 8】混合使用 groupby 参数 (通用方法方式)")
print("-" * 60)
df8 = pd.DataFrame({
    '组': ['G1', 'G1', 'G2', 'G2'],
    '值': [1, 2, 3, 4]
})
print("原始数据:")
print(df8.to_string(index=False))

stats8 = CumulativeStats(df8)
print("\n使用 cumsum(groupby='组') 通用方式:")
result8 = stats8.cumsum(groupby='组')
print(result8.to_string(index=False))

print("\n【示例 9】真实业务场景 - 区域月度销售报表")
print("-" * 60)
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=12, freq='ME')
regions = ['华北', '华北', '华北', '华北', '华东', '华东', '华东', '华东', '华南', '华南', '华南', '华南']
sales_values = np.random.randint(50, 200, size=12).astype(float)
sales_values[[2, 6, 9]] = np.nan
df9 = pd.DataFrame({
    '月份': dates.strftime('%Y-%m'),
    '区域': regions,
    '销售额': sales_values,
    '订单数': np.random.randint(5, 30, size=12)
})
print("原始销售数据 (含缺失月份):")
print(df9.to_string(index=False))

stats9 = CumulativeStats(df9)
result9 = stats9.groupby_cumsum(
    groupby='区域',
    nan_policy=NaNPolicy.FILL_FORWARD,
    value_cols=['销售额', '订单数']
)
print("\n按区域分组累计 (空值填充前值):")
print(result9.to_string(index=False))

print("\n" + "=" * 75)
print(" 示例运行完成！")
print("=" * 75)
