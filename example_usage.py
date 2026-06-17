import pandas as pd
import numpy as np
from cumulative_stats import CumulativeStats, compute_cumsum, compute_cumprod, compute_cummax

print("=" * 60)
print("累计统计服务使用示例")
print("=" * 60)

print("\n1. 使用列表数据 - 面向对象方式")
print("-" * 40)
data = [1, 2, 3, 4, 5]
print(f"原始数据: {data}")

stats = CumulativeStats(data)
print(f"\n累计和 (cumsum): {list(stats.cumsum()['value'])}")
print(f"累计乘积 (cumprod): {list(stats.cumprod()['value'])}")
print(f"累计最大值 (cummax): {list(stats.cummax()['value'])}")
print(f"累计最小值 (cummin): {list(stats.cummin()['value'])}")

print("\n2. 使用 DataFrame 数据")
print("-" * 40)
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
print("-" * 40)
all_results = stats2.all_stats()
for name, result in all_results.items():
    print(f"\n{name}:")
    print(result)

print("\n4. 使用便捷函数 - 函数式方式")
print("-" * 40)
data2 = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"原始数据: {data2}")
print(f"累计和: {list(compute_cumsum(data2)['value'])}")
print(f"累计乘积: {list(compute_cumprod(data2)['value'])}")
print(f"累计最大值: {list(compute_cummax(data2)['value'])}")

print("\n5. 处理包含 NaN 的数据")
print("-" * 40)
data3 = [1, np.nan, 3, np.nan, 5]
stats3 = CumulativeStats(data3)
print(f"原始数据: {data3}")
print(f"skipna=True 累计和: {list(stats3.cumsum(skipna=True)['value'].fillna(-1))}")
print(f"skipna=False 累计和: {list(stats3.cumsum(skipna=False)['value'])}")

print("\n6. 按行累计 (axis=1)")
print("-" * 40)
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

print("\n" + "=" * 60)
print("示例运行完成！")
print("=" * 60)
