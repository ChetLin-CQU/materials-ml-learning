from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# =========================
# 1. 路径设置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

data_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "steel_strength_clean.csv"
)

model_dir = PROJECT_ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = (
    model_dir / "random_forest_yield_strength.joblib"
)

metadata_path = (
    model_dir / "random_forest_yield_strength_metadata.json"
)


# =========================
# 2. 读取完整数据
# =========================

data = pd.read_csv(data_path)

target_column = "yield strength"

feature_columns = [
    column
    for column in data.columns
    if column not in ["formula", target_column]
]

X = data[feature_columns]
y = data[target_column]

print(f"训练样本数量：{len(data)}")
print(f"特征数量：{len(feature_columns)}")
print(f"特征顺序：{feature_columns}")


# =========================
# 3. 训练最终部署模型
# =========================

# 前面的独立测试已经完成。
# 部署模型使用全部312条数据重新训练，
# 使模型充分利用现有数据。
model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

print("\n最终模型训练完成")


# =========================
# 4. 保存模型及必要信息
# =========================

model_package = {
    "model": model,
    "feature_names": feature_columns,
    "target_column": target_column,
    "target_unit": "MPa",
    "feature_unit": "wt%",
    "feature_minimums": X.min().to_dict(),
    "feature_maximums": X.max().to_dict(),
}

joblib.dump(
    model_package,
    model_path,
)


# =========================
# 5. 保存可阅读的模型说明
# =========================

metadata = {
    "model_name": "Random Forest Yield Strength Predictor",
    "algorithm": "RandomForestRegressor",
    "training_samples": len(data),
    "target_column": target_column,
    "target_unit": "MPa",
    "feature_unit": "wt%",
    "feature_names": feature_columns,
    "hyperparameters": {
        "n_estimators": 500,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "max_depth": None,
        "random_state": 42,
    },
    "feature_ranges": {
        feature: {
            "minimum": float(X[feature].min()),
            "maximum": float(X[feature].max()),
        }
        for feature in feature_columns
    },
    "evaluation_note": (
        "The deployment model was retrained on all 312 samples. "
        "The previously locked test set produced MAE 78.83 MPa, "
        "RMSE 110.73 MPa, and R2 0.822."
    ),
    "limitation": (
        "The model uses composition features only and does not include "
        "heat treatment, processing, microstructure, or test conditions."
    ),
}

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=2,
    )


# =========================
# 6. 验证模型可以重新加载
# =========================

loaded_package = joblib.load(model_path)
loaded_model = loaded_package["model"]

example_prediction = loaded_model.predict(
    X.iloc[[0]]
)[0]

print("\n模型重新加载验证成功")
print(
    "第一个样本的预测屈服强度："
    f"{example_prediction:.2f} MPa"
)

print(f"\n模型已保存至：{model_path}")
print(f"模型说明已保存至：{metadata_path}")