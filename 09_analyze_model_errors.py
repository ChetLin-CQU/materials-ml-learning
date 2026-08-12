from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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

metrics_dir = PROJECT_ROOT / "results" / "metrics"
figure_dir = PROJECT_ROOT / "results" / "figures"

metrics_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取数据
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


# =========================
# 3. 使用与前面相同的数据划分
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# =========================
# 4. 训练当前最优随机森林
# =========================

model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# =========================
# 5. 建立误差分析表
# =========================

error_report = data.loc[X_test.index].copy()

error_report["actual_yield_strength_MPa"] = y_test
error_report["predicted_yield_strength_MPa"] = predictions

# 残差为正：模型低估；残差为负：模型高估
error_report["residual_MPa"] = (
    error_report["actual_yield_strength_MPa"]
    - error_report["predicted_yield_strength_MPa"]
)

error_report["absolute_error_MPa"] = (
    error_report["residual_MPa"].abs()
)

error_report["absolute_percentage_error_pct"] = (
    error_report["absolute_error_MPa"]
    / error_report["actual_yield_strength_MPa"]
    * 100
)

error_report["prediction_direction"] = np.where(
    error_report["residual_MPa"] > 0,
    "underestimated",
    "overestimated",
)

error_report = error_report.sort_values(
    "absolute_error_MPa",
    ascending=False,
).reset_index(drop=True)

# 为误差样本添加简短编号，方便绘图和查表
error_report.insert(
    0,
    "error_rank",
    range(1, len(error_report) + 1),
)

output_path = (
    metrics_dir / "random_forest_error_analysis.csv"
)

error_report.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 6. 统计高强度样本误差
# =========================

high_strength_threshold = 1800

high_strength_report = error_report[
    error_report["actual_yield_strength_MPa"]
    >= high_strength_threshold
]

overall_mean_residual = error_report["residual_MPa"].mean()

high_strength_mean_residual = (
    high_strength_report["residual_MPa"].mean()
    if len(high_strength_report) > 0
    else np.nan
)


# =========================
# 7. 绘制误差分析图
# =========================

figure, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)

# 左图：真实强度与残差
colors = np.where(
    error_report["residual_MPa"] > 0,
    "tomato",
    "steelblue",
)

axes[0].scatter(
    error_report["actual_yield_strength_MPa"],
    error_report["residual_MPa"],
    c=colors,
    alpha=0.75,
    edgecolor="white",
)

axes[0].axhline(
    0,
    color="black",
    linestyle="--",
)

axes[0].axvline(
    high_strength_threshold,
    color="gray",
    linestyle=":",
    label="High-strength threshold",
)

axes[0].set_xlabel("Actual Yield Strength (MPa)")
axes[0].set_ylabel("Residual: Actual - Predicted (MPa)")
axes[0].set_title("Residual Analysis")
axes[0].legend()
axes[0].grid(alpha=0.3)


# 右图：误差最大的10个样本
top_errors = error_report.head(10).copy()

labels = [
    f"Error Rank {rank}"
    for rank in top_errors["error_rank"]
]

axes[1].barh(
    labels[::-1],
    top_errors["absolute_error_MPa"][::-1],
    color="darkorange",
)

axes[1].set_xlabel("Absolute Error (MPa)")
axes[1].set_title("Top 10 Prediction Errors")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()

figure_path = (
    figure_dir / "random_forest_error_analysis.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# =========================
# 8. 输出结果
# =========================

print(f"测试集样本数：{len(error_report)}")

print(
    "模型低估的样本数："
    f"{(error_report['residual_MPa'] > 0).sum()}"
)

print(
    "模型高估的样本数："
    f"{(error_report['residual_MPa'] < 0).sum()}"
)

print(
    "\n全部测试样本平均残差："
    f"{overall_mean_residual:.2f} MPa"
)

print(
    f"高强度样本数量（≥{high_strength_threshold} MPa）："
    f"{len(high_strength_report)}"
)

print(
    "高强度样本平均残差："
    f"{high_strength_mean_residual:.2f} MPa"
)

print("\n误差最大的5个样本：")

display_columns = [
    column
    for column in [
        "formula",
        "actual_yield_strength_MPa",
        "predicted_yield_strength_MPa",
        "residual_MPa",
        "absolute_error_MPa",
        "prediction_direction",
    ]
    if column in error_report.columns
]

print(
    error_report[display_columns]
    .head(5)
    .to_string(index=False)
)

print(f"\n误差分析表已保存至：{output_path}")
print(f"误差分析图已保存至：{figure_path}")