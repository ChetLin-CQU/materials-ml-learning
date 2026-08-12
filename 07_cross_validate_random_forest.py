from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate


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

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
)

# 5折交叉验证；shuffle后固定随机种子，保证结果可复现
cross_validator = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cross_validator,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,
)

results_df = pd.DataFrame(
    {
        "fold": range(1, 6),
        "train_MAE_MPa": -cv_results["train_MAE"],
        "validation_MAE_MPa": -cv_results["test_MAE"],
        "train_RMSE_MPa": -cv_results["train_RMSE"],
        "validation_RMSE_MPa": -cv_results["test_RMSE"],
        "train_R2": cv_results["train_R2"],
        "validation_R2": cv_results["test_R2"],
    }
)

print("5折交叉验证结果：")
print(results_df.round(3).to_string(index=False))

print("\n验证集平均表现：")
print(
    f"MAE：{results_df['validation_MAE_MPa'].mean():.2f} "
    f"± {results_df['validation_MAE_MPa'].std():.2f} MPa"
)
print(
    f"RMSE：{results_df['validation_RMSE_MPa'].mean():.2f} "
    f"± {results_df['validation_RMSE_MPa'].std():.2f} MPa"
)
print(
    f"R²：{results_df['validation_R2'].mean():.3f} "
    f"± {results_df['validation_R2'].std():.3f}"
)

print("\n训练集平均R²：")
print(f"{results_df['train_R2'].mean():.3f}")

metrics_path = metrics_dir / "random_forest_cross_validation.csv"
results_df.to_csv(metrics_path, index=False)

# 绘制每折的训练集与验证集R²
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["fold"],
    results_df["train_R2"],
    marker="o",
    linewidth=2,
    label="Training R²",
)

plt.plot(
    results_df["fold"],
    results_df["validation_R2"],
    marker="o",
    linewidth=2,
    label="Validation R²",
)

plt.xticks(results_df["fold"])
plt.xlabel("Fold")
plt.ylabel("R²")
plt.title("Random Forest 5-Fold Cross-Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

figure_path = figure_dir / "random_forest_cross_validation.png"
plt.savefig(figure_path, dpi=300)
plt.show()

print(f"\n交叉验证结果已保存至：{metrics_path}")
print(f"交叉验证图已保存至：{figure_path}")