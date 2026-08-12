from pathlib import Path

import pandas as pd
from matminer.datasets import load_dataset


project_dir = Path(__file__).parent
raw_data_dir = project_dir / "data" / "raw"
processed_data_dir = project_dir / "data" / "processed"

raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# 加载原始数据
df = load_dataset(
    "steel_strength",
    data_home=str(raw_data_dir),
)

# 选择化学成分作为模型输入
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

# formula保留下来用于识别样本，但以后不直接送入模型
selected_columns = ["formula"] + feature_columns + [target_column]
clean_df = df[selected_columns].copy()

print(f"清洗前数据量：{len(clean_df)}")

# 删除完全重复的数据
duplicate_count = clean_df.duplicated().sum()
print(f"完全重复的数据：{duplicate_count}")
clean_df = clean_df.drop_duplicates()

# 删除输入特征或目标值缺失的记录
missing_row_count = clean_df[feature_columns + [target_column]].isna().any(axis=1).sum()
print(f"关键字段存在缺失的数据：{missing_row_count}")
clean_df = clean_df.dropna(subset=feature_columns + [target_column])

print(f"清洗后数据量：{len(clean_df)}")

print("\n清洗后各字段缺失值：")
print(clean_df.isna().sum())

# 保存处理后的数据
output_path = processed_data_dir / "steel_strength_clean.csv"
clean_df.to_csv(output_path, index=False)

print(f"\n清洗后的数据已保存至：{output_path}")