# -*- coding: utf-8 -*-
"""
华泰证券债券预测 —— 工程特征构建（2026_ori_data_new 版）

================================================================================
复用 03_XGB训练_二分类.py 的 build_features（STL 分解 / 多周期趋势 / 技术面
动量 / 跨资产背离 / 季节性等全部衍生特征），基于 01_数据预处理_2026new.py
产出的新数据构建完整特征表，供 XGB 与 NN（BiGRU）训练使用。

遗传算法特征（gp_feature_*）依赖标签共同演化，需先跑
02_训练数据构建_二分类.py --data-file/--rate-file/--output-dir 指向 data_new
重新生成，本脚本不包含。

输出：
  model_new/data_new/xgb_train_features_new.csv — 完整工程特征表
================================================================================
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # 项目根目录


def load_build_features():
    """从 03_XGB训练_二分类.py 导入 build_features 与参数缓存"""
    spec = importlib.util.spec_from_file_location(
        "xgb_train_bin", BASE / "model_new" / "03_XGB训练_二分类.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["xgb_train_bin"] = module
    spec.loader.exec_module(module)  # 模块级 os.chdir(项目根) 无副作用
    return module


def main():
    parser = argparse.ArgumentParser(description="工程特征构建（2026new）")
    parser.add_argument("--data-file", default="model_new/data_new/data.csv")
    parser.add_argument("--rate-file", default="model_new/data_new/rate.csv")
    parser.add_argument("--output-file", default="model_new/data_new/xgb_train_features_new.csv")
    parser.add_argument("--no-stl", action="store_true", help="禁用 STL 分解特征")
    parser.add_argument("--stl-period", type=int, default=63)
    args = parser.parse_args()

    module = load_build_features()

    # 与 03.main 的加载逻辑一致
    df_data = pd.read_csv(BASE / args.data_file)
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate = pd.read_csv(BASE / args.rate_file)
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(
        df_data, df_rate, left_on="Unnamed: 0", right_on="time", how="inner"
    ).sort_values("Unnamed: 0")
    print(f"底表: {len(merged)} 行, {merged.shape[1]} 列")

    # 复用 03 的完整特征工程（含 STL / 多周期趋势 / 技术面 / 跨资产背离 / 季节性）
    module._ARGS_CACHE = args
    features = module.build_features(merged)
    print(f"工程特征: {features.shape[0]} 行, {features.shape[1]} 列")

    out_path = BASE / args.output_file
    features.to_csv(out_path)
    print(f"输出: {out_path}")

    # 与旧版 Top-30 重要特征的对齐校验
    old_imp_path = BASE / "model_new" / "data" / "xgb_feature_importance_bin.csv"
    if old_imp_path.exists():
        old_imp = pd.read_csv(old_imp_path)
        top30 = old_imp[old_imp["importance"] > 0].head(30)["feature"].tolist()
        missing = [f for f in top30 if f not in features.columns]
        print("\n=== 旧版 Top-30 重要特征覆盖校验 ===")
        print(f"覆盖: {30 - len(missing)}/30")
        if missing:
            print(f"缺失（对应输入列在新数据中不存在）: {missing}")
        covered = [f for f in top30 if f in features.columns]
        print("覆盖清单:", covered)


if __name__ == "__main__":
    main()
