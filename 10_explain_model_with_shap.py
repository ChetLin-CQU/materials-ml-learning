from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
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

print(f"样本数量：{len(data)}")
print(f"特征数量：{len(feature_columns)}")
print(f"特征列表：{feature_columns}")


# =========================
# 3. 使用相同的数据划分
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
# 4. 训练随机森林
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

print("\n随机森林训练完成")


# =========================
# 5. 计算SHAP值
# =========================

print("正在计算SHAP值……")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print("SHAP值计算完成")


# =========================
# 6. 计算全局特征重要性
# =========================

mean_absolute_shap = (
    pd.DataFrame(
        {
            "element": feature_columns,
            "mean_absolute_SHAP_MPa": (
                abs(shap_values.values).mean(axis=0)
            ),
        }
    )
    .sort_values(
        "mean_absolute_SHAP_MPa",
        ascending=False,
    )
    .reset_index(drop=True)
)

mean_absolute_shap.insert(
    0,
    "importance_rank",
    range(1, len(mean_absolute_shap) + 1),
)

importance_path = (
    metrics_dir / "shap_feature_importance.csv"
)

mean_absolute_shap.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 7. 绘制SHAP重要性柱状图
# =========================

shap.plots.bar(
    shap_values,
    max_display=len(feature_columns),
    show=False,
)

plt.gcf().set_size_inches(9, 7)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()

bar_figure_path = (
    figure_dir / "shap_feature_importance.png"
)

plt.savefig(
    bar_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================
# 8. 绘制SHAP蜂群图
# =========================

shap.plots.beeswarm(
    shap_values,
    max_display=len(feature_columns),
    show=False,
)

plt.gcf().set_size_inches(10, 7)
plt.title("SHAP Feature Effects on Yield Strength")
plt.tight_layout()

beeswarm_figure_path = (
    figure_dir / "shap_beeswarm.png"
)

plt.savefig(
    beeswarm_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# =========================
# 9. 输出结果
# =========================

print("\nSHAP特征重要性排名：")

print(
    mean_absolute_shap.to_string(
        index=False,
        formatters={
            "mean_absolute_SHAP_MPa": "{:.2f}".format,
        },
    )
)

print(
    "\n平均绝对SHAP值表示该元素平均能够使模型预测值"
    "改变多少MPa，不表示因果关系。"
)

print(f"\nSHAP重要性表已保存至：{importance_path}")
print(f"SHAP柱状图已保存至：{bar_figure_path}")
print(f"SHAP蜂群图已保存至：{beeswarm_figure_path}")