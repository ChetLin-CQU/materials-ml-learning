from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_yield_strength.joblib"
)


# =========================
# 1. 加载模型
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "没有找到模型文件，请先运行 "
        "11_train_and_save_model.py"
    )

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
feature_names = model_package["feature_names"]
feature_minimums = model_package["feature_minimums"]
feature_maximums = model_package["feature_maximums"]


# =========================
# 2. 输入成分
# =========================

print("=" * 55)
print("钢材屈服强度预测程序")
print("=" * 55)

print(
    "\n请输入各元素的质量百分数（wt%）。"
    "\n例如1%输入1，0.5%输入0.5。"
    "\n直接按回车表示该元素含量为0。"
)

composition = {}
warnings = []

for element in feature_names:
    minimum = feature_minimums[element]
    maximum = feature_maximums[element]

    while True:
        user_input = input(
            f"{element.upper()} "
            f"（训练范围 {minimum:.6f}～{maximum:.6f}）："
        ).strip()

        if user_input == "":
            value = 0.0
            break

        try:
            value = float(user_input)
        except ValueError:
            print("输入无效，请输入数字。")
            continue

        if value < 0:
            print("元素含量不能为负数，请重新输入。")
            continue

        break

    composition[element] = value

    if value < minimum or value > maximum:
        warnings.append(
            f"{element.upper()}={value:.6f} "
            f"超出训练范围 "
            f"{minimum:.6f}～{maximum:.6f}"
        )


# =========================
# 3. 基本合理性检查
# =========================

alloying_sum = sum(composition.values())
estimated_fe_wt_pct = 100.0 - alloying_sum

print("\n" + "=" * 55)
print("输入成分汇总")
print("=" * 55)

for element in feature_names:
    print(
        f"{element.upper():>3s}："
        f"{composition[element]:.6f}"
    )

print(f"\n已输入合金元素总和：{alloying_sum:.6f} wt%")
print(f"估算Fe余量：{estimated_fe_wt_pct:.6f} wt%")

if alloying_sum >= 100.0:
    print(
        "\n错误：合金元素质量百分数总和大于或等于100，"
        "无法计算Fe余量。"
    )
    raise SystemExit(1)

# =========================
# 4. 进行预测
# =========================

input_data = pd.DataFrame(
    [[composition[name] for name in feature_names]],
    columns=feature_names,
)

prediction = model.predict(input_data)[0]


# =========================
# 5. 输出预测及警告
# =========================

print("\n" + "=" * 55)
print("预测结果")
print("=" * 55)

print(
    "预测屈服强度："
    f"{prediction:.2f} MPa"
)

if warnings:
    print("\n警告：存在超出模型训练范围的输入：")

    for warning in warnings:
        print(f"  - {warning}")

    print(
        "\n该结果属于外推预测，可靠性较低，"
        "不建议直接用于工程设计。"
    )
else:
    print(
        "\n所有输入均处于单个特征的训练范围内。"
    )

print(
    "\n模型限制：该预测仅使用化学成分，"
    "没有考虑热处理、加工工艺、显微组织"
    "和测试条件。"
)