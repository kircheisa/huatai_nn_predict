# -*- coding: utf-8 -*-
"""
新旧特征拼接：新库281特征(2010-01~2026-11) + 旧库280特征(2018-01~2025-01)

旧特征文件(model_new/data/xgb_train_features_bin.csv)无日期索引，
按 01_BiGRU训练_二分类.py 同款逻辑重建日期（data.csv∩rate.csv内连接，
shift(-5)标签有效行，>=2018-01-01），并与行数断言对齐。

重要口径：旧库原始数据(data/43_data.csv)只到2025-01-10，因此拼接特征中
旧列在2025-01之后为NaN——25/26推理时旧特征缺失，XGB按缺失方向路由。
评估时需分"旧特征可用段(<=2024)"与"缺失段(2025+)"分别看。

输出: data_new/43v2cat_features.csv（约561列，index=新库交易日）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OLD_DIR = BASE / "model_new" / "data"
NEW_DIR = BASE / "data_new"

# 前视特征剔除清单：旧管线 STL 为全样本双边平滑(03脚本L161)，其派生族
# 含未来信息（dev_from_trend 对63日前向方向AUC=0.83，实锤）；gp_* 为
# 全样本遗传规划拟合（gp_feature_2 前向AUC 0.58-0.65无衰减）。全部剔除。
STL_FAMILY = [
    "stl_trend", "stl_seasonal", "stl_residual",
    "dev_from_trend", "dev_zscore_120", "dev_chg_20d",
    "reversal_signal", "reversal_flag",
    "trend_slope_20", "trend_accel", "season_position",
    "stl_resid_lag_1", "stl_resid_lag_5", "stl_resid_ma_5",
] + [f"gp_feature_{i}" for i in range(1, 11)]


def rebuild_old_dates() -> pd.DatetimeIndex:
    """复刻 01_BiGRU脚本 的日期重建逻辑（horizon=5 版本）"""
    df_data = pd.read_csv(OLD_DIR / "data.csv", usecols=["Unnamed: 0"])
    df_rate = pd.read_csv(OLD_DIR / "rate.csv")[["time", "rate"]]
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0",
                      right_on="time", how="inner").sort_values("Unnamed: 0")
    merged = merged.loc[merged["rate"].shift(-5).notna()].reset_index(drop=True)
    mask = merged["Unnamed: 0"].values >= np.datetime64("2018-01-01")
    return pd.DatetimeIndex(merged.loc[mask, "Unnamed: 0"])


def main() -> None:
    old = pd.read_csv(OLD_DIR / "xgb_train_features_bin.csv", index_col=0)
    dates = rebuild_old_dates()
    assert len(dates) == len(old), \
        f"旧特征{len(old)}行 与重建日期{len(dates)}行不一致"
    old.index = dates
    old.columns = [f"OLD__{c}" for c in old.columns]

    new = pd.read_csv(NEW_DIR / "43v2_features.csv", index_col=0,
                      parse_dates=True)

    cat = new.join(old, how="left")
    # 去泄漏版：剔除前视 STL/GP 族后另存
    leak_cols = [f"OLD__{c}" for c in STL_FAMILY if f"OLD__{c}" in cat.columns]
    cat_clean = cat.drop(columns=leak_cols)
    print(f"剔除前视特征 {len(leak_cols)} 列 -> 干净版 {cat_clean.shape[1]} 列")
    cat_clean.to_csv(NEW_DIR / "43v2cat_clean_features.csv",
                     encoding="utf-8-sig")

    n_overlap = int(old.index.isin(new.index).sum())
    old_cols = [c for c in cat.columns if c.startswith("OLD__")]
    cov = cat[old_cols].notna().mean(axis=1)
    seg = {"2010~2017(旧缺失)": ((cov == 0) & (cat.index < "2018-01-01")).sum(),
           "2018~2025-01(旧可用)": ((cov > 0.99)).sum(),
           "2025-01后(旧缺失)": ((cat.index >= "2025-01-10")).sum()}
    print(f"拼接: 新{new.shape[1]}列 + 旧{len(old_cols)}列 = {cat.shape[1]}列, "
          f"{len(cat)}行")
    print(f"旧特征日期落在新网格: {n_overlap}/{len(old)}")
    for k, v in seg.items():
        print(f"  {k}: {v}行")
    cat.to_csv(NEW_DIR / "43v2cat_features.csv", encoding="utf-8-sig")
    print("输出:", NEW_DIR / "43v2cat_features.csv")


if __name__ == "__main__":
    main()
