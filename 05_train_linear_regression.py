from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

# 80%作为训练集，20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")

# 平均值基线
baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=y_train.mean(),
)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

# 标准化 + 线性回归
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

model_mae = mean_absolute_error(y_test, predictions)
model_rmse = mean_squared_error(y_test, predictions) ** 0.5
model_r2 = r2_score(y_test, predictions)

print("\n平均值基线：")
print(f"MAE：{baseline_mae:.2f} MPa")
print(f"RMSE：{baseline_rmse:.2f} MPa")
print(f"R²：{baseline_r2:.3f}")

print("\n线性回归：")
print(f"MAE：{model_mae:.2f} MPa")
print(f"RMSE：{model_rmse:.2f} MPa")
print(f"R²：{model_r2:.3f}")

# 保存评价指标
metrics_df = pd.DataFrame(
    {
        "model": ["mean_baseline", "linear_regression"],
        "MAE_MPa": [baseline_mae, model_mae],
        "RMSE_MPa": [baseline_rmse, model_rmse],
        "R2": [baseline_r2, model_r2],
    }
)

metrics_path = metrics_dir / "linear_regression_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# 绘制真实值与预测值对比图
plt.figure(figsize=(6.5, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.75,
    color="steelblue",
    edgecolor="white",
)

lower_limit = min(y_test.min(), predictions.min())
upper_limit = max(y_test.max(), predictions.max())

plt.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    linestyle="--",
    color="tomato",
    label="Ideal prediction",
)

plt.xlabel("Actual Yield Strength (MPa)")
plt.ylabel("Predicted Yield Strength (MPa)")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

figure_path = figure_dir / "linear_regression_predictions.png"
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"\n评价指标已保存至：{metrics_path}")
print(f"预测图已保存至：{figure_path}")