# -*- coding: utf-8 -*-
"""汇总基础特征实验过程，导出错误日期、方向概率误差和真实收益率变化。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent


def export():
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    rate = rate.dropna().sort_values("date").drop_duplicates("date")
    rate["maturity"] = rate["date"].shift(-10)
    rate["future_rate"] = rate["rate"].shift(-10)
    all_errors, progress, summary = [], [], []
    for path in sorted(OUT.glob("RAW_*_predictions.csv")):
        tag = path.name.removesuffix("_predictions.csv")
        if "SMOKE" in tag:
            continue
        pred = pd.read_csv(path)
        pred["date"] = pd.to_datetime(pred["date"])
        merged = pred.merge(rate, on="date", how="left", validate="one_to_one")
        assert merged["rate"].notna().all()
        assert np.array_equal((merged["future_rate"] < merged["rate"]).astype(int), merged["y_true"])
        detail = pd.DataFrame({
            "实验": tag, "日期": merged["date"].dt.strftime("%Y-%m-%d"),
            "折号": merged["fold"], "预测值_下行概率": merged["proba_down"],
            "预测方向": merged["pred"].map({0: "上行或不变", 1: "下行"}),
            "真实值_方向标签": merged["y_true"],
            "真实方向": merged["y_true"].map({0: "上行或不变", 1: "下行"}),
            "差值_概率减真实标签": merged["proba_down"] - merged["y_true"],
            "差值_预测标签减真实标签": merged["pred"] - merged["y_true"],
            "绝对概率误差": np.abs(merged["proba_down"] - merged["y_true"]),
            "是否预测错误": merged["pred"] != merged["y_true"],
            "当日真实收益率_百分数": merged["rate"],
            "标签成熟日期": merged["maturity"].dt.strftime("%Y-%m-%d"),
            "未来10交易日真实收益率_百分数": merged["future_rate"],
            "真实变化_bp": (merged["future_rate"] - merged["rate"]) * 100,
        })
        detail.to_csv(OUT / f"{tag}_预测明细.csv", index=False, encoding="utf-8-sig")
        bad = detail[detail["是否预测错误"]]
        bad.to_csv(OUT / f"{tag}_预测错误日期.csv", index=False, encoding="utf-8-sig")
        all_errors.append(bad)
        folds_path = OUT / f"{tag}_folds.csv"
        if folds_path.exists():
            folds = pd.read_csv(folds_path)
            folds.insert(0, "experiment", tag)
            progress.append(folds)
        report_path = OUT / f"{tag}_metrics.json"
        audit_path = OUT / f"{tag}_audit.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            params = json.loads(audit_path.read_text(encoding="utf-8")).get("experiment_args", {}) if audit_path.exists() else {}
            summary.append({"实验": tag, "状态": report["status"],
                            "完成折数": report["folds_completed"], "耗时秒": report["seconds"],
                            **{f"全局_{k}": v for k, v in report["global"].items()},
                            **{f"2025_{k}": v for k, v in report["years"].get("2025", {}).items()},
                            "错误日期数": len(bad), "参数": json.dumps(params, ensure_ascii=False)})
    if all_errors:
        pd.concat(all_errors).to_csv(OUT / "RAW_全部实验_预测错误日期.csv", index=False, encoding="utf-8-sig")
    if progress:
        pd.concat(progress).to_csv(OUT / "RAW_训练过程_逐折指标.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT / "RAW_训练过程_实验汇总.csv", index=False, encoding="utf-8-sig")
    print(f"已导出 {len(summary)}组实验记录；错误日期为硬方向预测错误，概率差值不是收益率误差。")


if __name__ == "__main__":
    export()
