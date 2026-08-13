from pathlib import Path
import os
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE_SCRIPTS = [
    "02_load_steel_dataset.py",
    "03_clean_steel_data.py",
    "04_explore_steel_data.py",
    "05_train_linear_regression.py",
    "06_train_random_forest.py",
    "07_cross_validate_random_forest.py",
    "08_tune_random_forest.py",
    "09_analyze_model_errors.py",
    "10_explain_model_with_shap.py",
    "11_train_and_save_model.py",
]


def check_scripts():
    """检查流水线需要的程序是否完整。"""
    missing_scripts = []

    for script_name in PIPELINE_SCRIPTS:
        script_path = PROJECT_ROOT / script_name

        if not script_path.exists():
            missing_scripts.append(script_name)

    if missing_scripts:
        print("无法运行，以下程序不存在：")

        for script_name in missing_scripts:
            print(f"  - {script_name}")

        raise SystemExit(1)


def run_pipeline():
    """按照规定顺序运行完整项目。"""
    check_scripts()

    environment = os.environ.copy()

    # 使用非交互绘图模式，防止图片窗口阻塞流水线
    environment["MPLBACKEND"] = "Agg"

    print("=" * 60)
    print("Materials ML Learning：完整流水线")
    print("=" * 60)

    print(f"项目目录：{PROJECT_ROOT}")
    print(f"Python解释器：{sys.executable}")
    print(f"任务数量：{len(PIPELINE_SCRIPTS)}")

    total_start_time = time.perf_counter()

    for step_number, script_name in enumerate(
        PIPELINE_SCRIPTS,
        start=1,
    ):
        script_path = PROJECT_ROOT / script_name

        print("\n" + "=" * 60)
        print(
            f"正在运行 {step_number}/{len(PIPELINE_SCRIPTS)}："
            f"{script_name}"
        )
        print("=" * 60)

        step_start_time = time.perf_counter()

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print("\n流水线运行失败")
            print(f"失败程序：{script_name}")
            print(f"退出代码：{error.returncode}")
            raise SystemExit(error.returncode)

        step_elapsed_time = (
            time.perf_counter() - step_start_time
        )

        print(
            f"\n{script_name} 运行完成，"
            f"耗时 {step_elapsed_time:.1f} 秒"
        )

    total_elapsed_time = (
        time.perf_counter() - total_start_time
    )

    print("\n" + "=" * 60)
    print("完整流水线运行成功")
    print(
        f"总耗时：{total_elapsed_time / 60:.1f} 分钟"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()