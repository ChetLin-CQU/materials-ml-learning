from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


project_dir = Path(__file__).parent
data_path = project_dir / "data" / "processed" / "steel_strength_clean.csv"

figure_dir = project_dir / "results" / "figures"
metrics_dir = project_dir / "results" / "metrics"

figure_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

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

X = df[feature_columns]
y = df[target_column]

# 使用与线性回归完全相同的数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 平均值基线
baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=y_train.mean(),
)

# 随机森林基线模型
model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2


baseline_mae, baseline_rmse, baseline_r2 = calculate_metrics(
    y_test,
    baseline_predictions,
)

train_mae, train_rmse, train_r2 = calculate_metrics(
    y_train,
    train_predictions,
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    test_predictions,
)

print("平均值基线（测试集）：")
print(f"MAE：{baseline_mae:.2f} MPa")
print(f"RMSE：{baseline_rmse:.2f} MPa")
print(f"R²：{baseline_r2:.3f}")

print("\n随机森林（训练集）：")
print(f"MAE：{train_mae:.2f} MPa")
print(f"RMSE：{train_rmse:.2f} MPa")
print(f"R²：{train_r2:.3f}")

print("\n随机森林（测试集）：")
print(f"MAE：{test_mae:.2f} MPa")
print(f"RMSE：{test_rmse:.2f} MPa")
print(f"R²：{test_r2:.3f}")

# 保存评价指标
metrics_df = pd.DataFrame(
    {
        "model": [
            "mean_baseline_test",
            "random_forest_train",
            "random_forest_test",
        ],
        "MAE_MPa": [
            baseline_mae,
            train_mae,
            test_mae,
        ],
        "RMSE_MPa": [
            baseline_rmse,
            train_rmse,
            test_rmse,
        ],
        "R2": [
            baseline_r2,
            train_r2,
            test_r2,
        ],
    }
)

metrics_path = metrics_dir / "random_forest_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# 计算特征重要性
importance_df = pd.DataFrame(
    {
        "element": feature_columns,
        "importance": model.feature_importances_,
    }
).sort_values("importance")

print("\n随机森林特征重要性：")
print(importance_df.sort_values("importance", ascending=False))

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    y_test,
    test_predictions,
    alpha=0.75,
    color="seagreen",
    edgecolor="white",
)

lower_limit = min(y_test.min(), test_predictions.min())
upper_limit = max(y_test.max(), test_predictions.max())

axes[0].plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    linestyle="--",
    color="tomato",
    label="Ideal prediction",
)

axes[0].set_xlabel("Actual Yield Strength (MPa)")
axes[0].set_ylabel("Predicted Yield Strength (MPa)")
axes[0].set_title("Random Forest: Actual vs Predicted")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].barh(
    importance_df["element"],
    importance_df["importance"],
    color="steelblue",
)

axes[1].set_xlabel("Feature Importance")
axes[1].set_ylabel("Element")
axes[1].set_title("Random Forest Feature Importance")

plt.tight_layout()

figure_path = figure_dir / "random_forest_results.png"
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"\n评价指标已保存至：{metrics_path}")
print(f"结果图已保存至：{figure_path}")