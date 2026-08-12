from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
)


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
# 2. 读取清洗后的数据
# =========================

data = pd.read_csv(data_path)

target_column = "yield strength"

excluded_columns = [
    "formula",
    target_column,
]

feature_columns = [
    column
    for column in data.columns
    if column not in excluded_columns
]

X = data[feature_columns]
y = data[target_column]

print(f"样本数量：{len(data)}")
print(f"特征数量：{len(feature_columns)}")
print(f"特征列表：{feature_columns}")


# =========================
# 3. 固定训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print(f"\n训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")


# =========================
# 4. 设置交叉验证
# =========================

cross_validation = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# =========================
# 5. 设置待搜索参数
# =========================

parameter_distributions = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", 0.7],
}


base_model = RandomForestRegressor(
    random_state=42,
)


# =========================
# 6. 随机参数搜索
# =========================

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=parameter_distributions,
    n_iter=30,
    scoring="neg_mean_absolute_error",
    cv=cross_validation,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print("\n开始参数搜索，请耐心等待……")

search.fit(X_train, y_train)

best_model = search.best_estimator_


# =========================
# 7. 保存全部搜索结果
# =========================

search_results = pd.DataFrame(search.cv_results_)

search_results["validation_MAE_MPa"] = (
    -search_results["mean_test_score"]
)

search_results["train_MAE_MPa"] = (
    -search_results["mean_train_score"]
)

result_columns = [
    "rank_test_score",
    "params",
    "validation_MAE_MPa",
    "std_test_score",
    "train_MAE_MPa",
    "mean_fit_time",
]

search_results = search_results[
    result_columns
].sort_values("rank_test_score")

search_results_path = (
    metrics_dir / "random_forest_parameter_search.csv"
)

search_results.to_csv(
    search_results_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 8. 最优模型最终评估
# =========================

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


train_mae, train_rmse, train_r2 = calculate_metrics(
    y_train,
    train_predictions,
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    test_predictions,
)


metrics = pd.DataFrame(
    [
        {
            "dataset": "training",
            "MAE_MPa": train_mae,
            "RMSE_MPa": train_rmse,
            "R2": train_r2,
        },
        {
            "dataset": "testing",
            "MAE_MPa": test_mae,
            "RMSE_MPa": test_rmse,
            "R2": test_r2,
        },
    ]
)

metrics_path = (
    metrics_dir / "tuned_random_forest_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 9. 绘制测试集预测结果
# =========================

plt.figure(figsize=(8, 7))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.75,
    color="seagreen",
    edgecolor="white",
)

minimum_value = min(
    y_test.min(),
    test_predictions.min(),
)

maximum_value = max(
    y_test.max(),
    test_predictions.max(),
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    "--",
    color="tomato",
    label="Ideal prediction",
)

plt.xlabel("Actual Yield Strength (MPa)")
plt.ylabel("Predicted Yield Strength (MPa)")
plt.title("Tuned Random Forest: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

figure_path = (
    figure_dir / "tuned_random_forest_results.png"
)

plt.savefig(
    figure_path,
    dpi=300,
)

plt.show()


# =========================
# 10. 输出结果
# =========================

print("\n最优参数：")

for parameter, value in search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\n交叉验证最佳MAE："
    f"{-search.best_score_:.2f} MPa"
)

print("\n最优随机森林——训练集：")
print(f"MAE：{train_mae:.2f} MPa")
print(f"RMSE：{train_rmse:.2f} MPa")
print(f"R²：{train_r2:.3f}")

print("\n最优随机森林——测试集：")
print(f"MAE：{test_mae:.2f} MPa")
print(f"RMSE：{test_rmse:.2f} MPa")
print(f"R²：{test_r2:.3f}")

print(f"\n参数搜索结果已保存至：{search_results_path}")
print(f"评价指标已保存至：{metrics_path}")
print(f"预测结果图已保存至：{figure_path}")