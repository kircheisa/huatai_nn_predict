# -*- coding: utf-8 -*-
"""
文本标签特征构建（方案A：政策事件日历 -> 日度衰减特征）

逻辑：
  1. 读取 LLM 辅助标注的政策事件日历（data/out_data/文本标签_政策事件日历.csv）
     - direction: +1 利多债市（收益率下行） / -1 利空债市 / 0 中性
     - intensity: 1 一般 / 2 重要 / 3 重大
  2. 在完整交易日历上（复用 09 脚本的原始利率日期）计算日度特征：
     - txt_pos / txt_neg: 近60交易日利多/利空事件的强度衰减和（指数半衰10日）
     - txt_net: 净情绪 = pos - neg
     - txt_cat_*: 七类事件的衰减强度；txt_intensity为总衰减强度
     - txt_cnt20: 近20交易日事件数（事件密度）
     - txt_gap: 距最近事件的交易日数（封顶60，事件稀疏度）
  3. 客观特征不使用direction；方向特征追加txt_pos/txt_neg/txt_net。
     输出 model_nn/data/text_features_daily.csv，09训练直接复用构建函数。

防泄漏：无发布时间的事件，从事件日期之后的第一个交易日起生效。
注意：回溯标注仍可能存在事后选择/解释偏差，时间对齐不能消除该风险。
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EVENT_CSV = BASE / "data" / "out_data" / "文本标签_政策事件日历.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"

HALF_LIFE = 10.0   # 衰减半衰期（交易日）
MAX_GAP = 60       # gap 封顶
CATEGORIES = {
    "货币宽松": "loose", "货币收紧": "tight", "重大会议": "meeting",
    "财政发力": "fiscal", "外部冲击": "external",
    "风险事件": "risk", "宽信用地产": "credit",
}
OBJECTIVE_COLUMNS = (
    "txt_intensity", "txt_cnt20", "txt_gap",
    *(f"txt_cat_{name}" for name in CATEGORIES.values()),
)
DIRECTION_COLUMNS = ("txt_pos", "txt_neg", "txt_net")


def read_events(path=EVENT_CSV):
    """兼容逗号/制表符，拒绝静默丢失或非法标注。"""
    ev = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {"date", "category", "direction", "intensity"}
    if not required.issubset(ev.columns):
        raise ValueError(f"事件日历缺少列: {sorted(required - set(ev.columns))}")
    ev["date"] = pd.to_datetime(ev["date"], errors="raise").dt.normalize()
    ev["direction"] = pd.to_numeric(ev["direction"], errors="raise")
    ev["intensity"] = pd.to_numeric(ev["intensity"], errors="raise")
    if ev[list(required)].isna().any().any():
        raise ValueError("事件日历存在空值")
    if not ev["direction"].isin([-1, 0, 1]).all():
        raise ValueError("direction必须为-1/0/1")
    if not ev["intensity"].isin([1, 2, 3]).all():
        raise ValueError("intensity必须为1/2/3")
    if not ev["category"].isin(CATEGORIES).all():
        raise ValueError("事件日历包含未知类别")
    return ev.sort_values("date", kind="stable").reset_index(drop=True)


def build_event_features(days, events, mode="direction"):
    """在完整交易日历上计算特征；同日及未来事件不可用于当天预测。"""
    if mode not in ("objective", "direction"):
        raise ValueError("mode必须为objective或direction")
    days = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
    if days.hasnans or not days.is_monotonic_increasing or days.has_duplicates:
        raise ValueError("交易日历必须非空值、严格递增且不重复")
    ev = events.sort_values("date", kind="stable").reset_index(drop=True)
    ev_idx = days.searchsorted(pd.to_datetime(ev["date"]), side="right")
    # 日历必须覆盖历史起点，避免旧事件被错误当作首日新事件。
    if len(days) and len(ev) and ev["date"].min() < days[0]:
        raise ValueError("请传入覆盖最早事件的完整交易日历，再按预测日期筛选")
    n = len(days)
    out = pd.DataFrame(0.0, index=days,
                       columns=list(OBJECTIVE_COLUMNS) + list(DIRECTION_COLUMNS))
    out["txt_gap"] = MAX_GAP
    for d in range(n):
        lo = np.searchsorted(ev_idx, d - MAX_GAP + 1, side="left")
        hi = np.searchsorted(ev_idx, d, side="right")
        recent = ev.iloc[lo:hi]
        dist = d - ev_idx[lo:hi]
        decay = np.exp(-np.log(2.0) * dist / HALF_LIFE)
        strength = recent["intensity"].to_numpy(dtype=float) * decay
        out.iloc[d, out.columns.get_loc("txt_intensity")] = strength.sum()
        out.iloc[d, out.columns.get_loc("txt_cnt20")] = (
            hi - np.searchsorted(ev_idx, d - 19, side="left"))
        if len(dist):
            out.iloc[d, out.columns.get_loc("txt_gap")] = dist.min()
        for category, name in CATEGORIES.items():
            out.iloc[d, out.columns.get_loc(f"txt_cat_{name}")] = (
                strength[recent["category"].to_numpy() == category].sum())
        if mode == "direction":
            direction = recent["direction"].to_numpy(dtype=float)
            out.iloc[d, out.columns.get_loc("txt_pos")] = strength[direction > 0].sum()
            out.iloc[d, out.columns.get_loc("txt_neg")] = strength[direction < 0].sum()
    out["txt_net"] = out["txt_pos"] - out["txt_neg"]
    columns = OBJECTIVE_COLUMNS + (DIRECTION_COLUMNS if mode == "direction" else ())
    return out.loc[:, list(columns)].round(4)


def build_trading_days():
    """复用09原始利率的完整交易日历，避免截断事件衰减历史。"""
    rate = pd.read_csv(BASE / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    return rate.dropna().sort_values("date").drop_duplicates("date")["date"].reset_index(drop=True)


def main():
    days = build_trading_days()
    ev = read_events()
    out = build_event_features(days, ev).rename_axis("date").reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "text_features_daily.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"交易日: {len(days)}, 事件数: {len(ev)}, 半衰期: {HALF_LIFE}日，下一交易日起生效")
    print(f"输出: {out_path}")
    print(out[out["txt_cnt20"] > 0].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
