from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


project_dir = Path(__file__).parent
data_path = project_dir / "data" / "processed" / "steel_strength_clean.csv"
figure_dir = project_dir / "results" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

feature_columns = [
    "c",
    "mn",
    "si",
    "cr",
    "ni",
    "mo",
    "v",
    "nb",
    "co",
    "al",
    "w",
    "ti",
    "n",
]

target_column = "yield strength"

print("屈服强度统计信息：")
print(df[target_column].describe())

# 设置绘图风格
sns.set_theme(style="whitegrid")

# 图1：屈服强度分布
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.histplot(
    data=df,
    x=target_column,
    bins=20,
    kde=True,
    color="steelblue",
    ax=axes[0],
)

axes[0].set_title("Distribution of Yield Strength")
axes[0].set_xlabel("Yield Strength (MPa)")
axes[0].set_ylabel("Number of Samples")

sns.boxplot(
    data=df,
    x=target_column,
    color="lightblue",
    ax=axes[1],
)

axes[1].set_title("Boxplot of Yield Strength")
axes[1].set_xlabel("Yield Strength (MPa)")

plt.tight_layout()

distribution_path = figure_dir / "yield_strength_distribution.png"
plt.savefig(distribution_path, dpi=300)
print(f"\n强度分布图已保存至：{distribution_path}")

# 计算元素含量与屈服强度的皮尔逊相关系数
correlations = (
    df[feature_columns + [target_column]]
    .corr(numeric_only=True)[target_column]
    .drop(target_column)
    .sort_values()
)

print("\n元素含量与屈服强度的相关系数：")
print(correlations)

# 图2：相关系数条形图
plt.figure(figsize=(8, 6))

colors = [
    "tomato" if value < 0 else "steelblue"
    for value in correlations.values
]

plt.barh(
    correlations.index,
    correlations.values,
    color=colors,
)

plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Element")
plt.title("Composition Correlation with Yield Strength")
plt.tight_layout()

correlation_path = figure_dir / "element_yield_correlation.png"
plt.savefig(correlation_path, dpi=300)

print(f"相关性图已保存至：{correlation_path}")

plt.show()