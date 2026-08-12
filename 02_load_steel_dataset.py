from pathlib import Path

import pandas as pd
from matminer.datasets import load_dataset


project_dir = Path(__file__).parent
raw_data_dir = project_dir / "data" / "raw"
raw_data_dir.mkdir(parents=True, exist_ok=True)

print("正在加载 steel_strength 数据集……")

df = load_dataset(
    "steel_strength",
    data_home=str(raw_data_dir),
)

pd.set_option("display.max_columns", None)

print("\n前5行数据：")
print(df.head())

print(f"\n数据行数：{df.shape[0]}")
print(f"数据列数：{df.shape[1]}")

print("\n全部字段：")
for column in df.columns:
    print(f"- {column}")

print("\n存在缺失值的字段：")
missing_values = df.isna().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

print("\n强度统计：")
print(df[["yield strength", "tensile strength"]].describe())