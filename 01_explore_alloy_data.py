from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 获取项目文件夹及数据文件路径
project_dir = Path(__file__).parent
data_path = project_dir / "data" / "alloys.csv"

# 从CSV读取数据
df = pd.read_csv(data_path)

print("材料数据表：")
print(df)

print(f"\n数据行数：{df.shape[0]}")
print(f"数据列数：{df.shape[1]}")
print("\n每列缺失值数量：")
print(df.isna().sum())

average_strength = df["tensile_strength_MPa"].mean()
print(f"\n平均抗拉强度：{average_strength:.1f} MPa")

# 绘制散点图
plt.figure(figsize=(7, 5))

plt.scatter(
    df["Al_wt_pct"],
    df["tensile_strength_MPa"],
    s=80,
    color="steelblue",
)

# 在每个数据点旁标注合金名称
for _, row in df.iterrows():
    plt.annotate(
        row["alloy"],
        (row["Al_wt_pct"], row["tensile_strength_MPa"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel("Al Content (wt%)")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Al Content vs Tensile Strength")
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = project_dir / "alloy_strength.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\n图片已保存至：{output_path}")