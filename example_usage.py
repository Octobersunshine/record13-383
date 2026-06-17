import pandas as pd
import numpy as np
from cumulative_stats import CumulativeStats, NaNPolicy, compute_cumsum, compute_cumprod, compute_cummax

print("=" * 70)
print("累计统计服务使用示例 (含分组累计和空值处理修复)")
print("=" * 70)

print("\n1. 使用列表数据 - 面向对象方式")
print("-" * 50)
data = [1, 2, 3, 4, 5]
print(f"原始数据: {data}")

stats = CumulativeStats(data)
print(f"\n累计和 (cumsum): {list(stats.cumsum()['value'])}")
print(f"累计乘积 (cumprod): {list(stats.cumprod()['value'])}")
print(f"累计最大值 (cummax): {list(stats.cummax()['value'])}")
print(f"累计最小值 (cummin): {list(stats.cummin()['value'])}")

print("\n2. 使用 DataFrame 数据")
print("-" * 50)
df = pd.DataFrame({
    '销售额': [100, 200, 150, 300, 250],
    '订单数': [5, 8, 6, 12, 10]
}, index=['周一', '周二', '周三', '周四', '周五'])
print("原始数据:")
print(df)

stats2 = CumulativeStats(df)
print("\n累计和 (cumsum):")
print(stats2.cumsum())
print("\n累计最大值 (cummax):")
print(stats2.cummax())

print("\n3. 一次性获取所有统计结果")
print("-" * 50)
all_results = stats2.all_stats()
for name, result in all_results.items():
    print(f"\n{name}:")
    print(result)

print("\n4. 使用便捷函数 - 函数式方式")
print("-" * 50)
data2 = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"原始数据: {data2}")
print(f"累计和: {list(compute_cumsum(data2)['value'])}")
print(f"累计乘积: {list(compute_cumprod(data2)['value'])}")
print(f"累计最大值: {list(compute_cummax(data2)['value'])}")

print("\n5. 空值处理策略 (NaNPolicy)")
print("-" * 50)
data3 = [1, np.nan, 3, np.nan, 5]
stats3 = CumulativeStats(data3)
print(f"原始数据: {data3}")

print("\n  5.1 PRESERVE 策略 (保留空值位置):")
result_preserve = stats3.cumsum(nan_policy=NaNPolicy.PRESERVE)
print(f"      累计和: {list(result_preserve['value'].fillna(-1))}  (-1 表示 NaN)")

print("\n  5.2 FILL_FORWARD 策略 (前值填充，累计不中断):")
result_fill = stats3.cumsum(nan_policy=NaNPolicy.FILL_FORWARD)
print(f"      累计和: {list(result_fill['value'])}")

print("\n  5.3 skipna=False (遇到 NaN 后全部为 NaN):")
result_no_skip = stats3.cumsum(skipna=False)
print(f"      累计和: {list(result_no_skip['value'])}")

print("\n6. 分组累计 (groupby) - Bug 修复展示")
print("-" * 50)
df_grouped = pd.DataFrame({
    '部门': ['销售部', '销售部', '销售部', '销售部',
             '技术部', '技术部', '技术部', '技术部'],
    '月度业绩': [100, np.nan, 150, 200, 80, np.nan, np.nan, 120]
})
print("原始数据 (含空值):")
print(df_grouped.to_string(index=False))

stats_grouped = CumulativeStats(df_grouped)

print("\n  6.1 分组累计和 - PRESERVE (保留空值):")
result1 = stats_grouped.cumsum(groupby='部门', nan_policy=NaNPolicy.PRESERVE)
print(result1.to_string(index=False))

print("\n  6.2 分组累计和 - FILL_FORWARD (前值填充，累计不中断):")
result2 = stats_grouped.cumsum(groupby='部门', nan_policy=NaNPolicy.FILL_FORWARD)
print(result2.to_string(index=False))

print("\n  6.3 分组累计最大值 - FILL_FORWARD:")
result3 = stats_grouped.cummax(groupby='部门', nan_policy=NaNPolicy.FILL_FORWARD)
print(result3.to_string(index=False))

print("\n7. 多列分组累计")
print("-" * 50)
df_multi = pd.DataFrame({
    '区域': ['华东', '华东', '华东', '华东', '华南', '华南', '华南', '华南'],
    '产品': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    '销量': [10, 20, 15, 25, 30, np.nan, 40, 50]
})
print("原始数据 (按区域+产品分组):")
print(df_multi.to_string(index=False))

stats_multi = CumulativeStats(df_multi)
result_multi = stats_multi.cumsum(groupby=['区域', '产品'], nan_policy=NaNPolicy.FILL_FORWARD)
print("\n分组累计和 (前值填充):")
print(result_multi.to_string(index=False))

print("\n8. 按行累计 (axis=1)")
print("-" * 50)
df2 = pd.DataFrame({
    'Q1': [100, 200, 300],
    'Q2': [150, 180, 250],
    'Q3': [200, 220, 280],
    'Q4': [250, 300, 350]
}, index=['产品A', '产品B', '产品C'])
print("原始数据 (季度销售额):")
print(df2)
stats4 = CumulativeStats(df2)
print("\n按行累计和 (年度累计):")
print(stats4.cumsum(axis=1))

print("\n" + "=" * 70)
print("示例运行完成！")
print("=" * 70)
