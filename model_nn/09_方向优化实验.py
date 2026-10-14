# -*- coding: utf-8 -*-
"""
09_方向优化实验.py
======================================================================
在原始工作簿因果特征框架上，统一验证四个方向能否真正提高准确率：

  T1 稳健性验证   : 所有实验都在开发期(<=2024,标签已成熟)选口径，2025只作查看；
                    多数类增益、逐年/逐seed和分块区间仅作证据，不等于已证实改善。
  T2 两阶段(死区) : Stage1 预测"未来10日是否大幅变动"(|Δ|>=2bp)，
                    Stage2 预测方向；组合后死区弃权，只输出高把握方向。
  T3 重定义标签   : down = (Δ < -2bp)，把 |Δ|<=2bp 并入上行/持平。
  T4 趋势动量特征 : 变体E = 变体D + 动量/利率水平/回撤/长周期波动(全部因果)。

统一协议：
  - 变体 D/E，历史起点 2010-01-01，28折沿用历史XGB折边界。
  - 训练标签必须在测试首日前成熟；purge gap = max(seq_len, horizon)。
  - 标准化/填补/筛列只用梯度训练段；决策阈值固定0.5。
  - 两阶段的 gate 阈值只在开发期选择。

运行：
  快速自检 : python 09_方向优化实验.py --task base_D --max-folds 2 --n-seeds 1 --epochs 3
  全量     : python 09_方向优化实验.py --suite --n-seeds 3 --epochs 25
  仅汇总   : python 09_方向优化实验.py --summarize-only
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.append(str(HERE / "_libs"))

DEAD_BP = 2.0          # 死区阈值(bp)
CUTOFF = pd.Timestamp("2024-12-31")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("nn_core9", HERE / "01_BiGRU训练_二分类_10天.py")
b5 = load_module("base5", HERE / "05_基础特征对照训练.py")
text2 = load_module("text_features2", HERE / "02_文本特征构建.py")

TASKS = {
    "base_D":      {"variant": "D", "label": "down"},
    "relabel_D":   {"variant": "D", "label": "down_ex2"},
    "twostage_D":  {"variant": "D", "label": "down", "twostage": True},
    "features_E":  {"variant": "E", "label": "down"},
    "featrelab_E": {"variant": "E", "label": "down_ex2"},
    # 第二轮：基于开发期选出的 featrelab_E，验证训练侧三类适配
    "featrelab_E_W1260": {"variant": "E", "label": "down_ex2", "max_train_days": 1260},
    "featrelab_E_W756":  {"variant": "E", "label": "down_ex2", "max_train_days": 756},
    "featrelab_E_W504":  {"variant": "E", "label": "down_ex2", "max_train_days": 504},
    "featrelab_E_R252":  {"variant": "E", "label": "down_ex2", "recency_halflife": 252},
    "featrelab_E_S20":   {"variant": "E", "label": "down_ex2", "state_similarity": 20},
    # 第三轮：在开发期最优的 featrelab_E 上补充资金面/政策/机构行为特征
    "featF_relab": {"variant": "F", "label": "down_ex2"},
    "featF_down":  {"variant": "F", "label": "down"},
    # 事件特征消融：固定E基线与down_ex2标签，不根据2025选参。
    "event_objective_E": {"variant": "E", "label": "down_ex2", "event_mode": "objective"},
    "event_direction_E": {"variant": "E", "label": "down_ex2", "event_mode": "direction"},
}

# 任务级训练参数覆盖键（spec 中的值优先于命令行默认值）
SPEC_TRAIN_KEYS = ("max_train_days", "recency_halflife", "state_similarity")


# =====================================================================
# 数据构建（沿用05的因果对齐，变体E追加动量/水平/回撤/长周期波动）
# =====================================================================
def build_dataset(variant, horizon, history_start, event_mode="none"):
    book = pd.ExcelFile(ROOT / "data_ori" / "2026_origin_data.xlsx")
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    rate = rate.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    daily = b5.sheet(book, 5)
    dates = rate["date"]

    features = {}
    for index, prefix in [(5, "daily"), (4, "weekly"), (6, "policy")]:
        data = b5.sheet(book, index)
        for col in data:
            series = pd.to_numeric(data[col], errors="coerce")
            features[f"{prefix}:{col}"] = b5.align_published(series, data.index, dates)
    for index, pub_index, mapping, prefix in [
        (0, 1, {"GDP:不变价:当季同比": "GDP同比",
                "中国:工业产能利用率:当季值": "产能利用率"}, "quarter"),
        (2, 3, b5.pre.MONTHLY_PUBLISH_MAP, "month"),
    ]:
        data, pub = b5.sheet(book, index), b5.sheet(book, pub_index)
        for col, release in mapping.items():
            if col not in data or release is None or release not in pub:
                continue
            series = pd.to_numeric(data[col], errors="coerce")
            published = pub[release].reindex(data.index)
            features[f"{prefix}:{col}"] = b5.align_published(series, published, dates)
    frame = pd.DataFrame(features, index=dates)

    # C/D 共有衍生（与05一致）
    selected = ["DR007", "R007", "1Y存单发行利率",
                "国债期货收盘价(活跃合约):10年", "沪深300", "10年美债", "美元指数"]
    for col in selected:
        key = f"daily:{col}"
        if key in frame:
            for lag in (5, 20):
                frame[f"diff{lag}:{col}"] = frame[key].diff(lag)
            frame[f"vol20:{col}"] = frame[key].diff().rolling(20, min_periods=10).std()
    frame["spread:资金"] = frame["daily:R007"] - frame["daily:DR007"]
    past_rate = pd.Series(rate["rate"].to_numpy(), index=dates).shift(1)
    for lag in (1, 5, 20):
        frame[f"rate:diff{lag}"] = past_rate.diff(lag)
    frame["rate:vol20"] = past_rate.diff().rolling(20, min_periods=10).std()

    # 变体E：趋势/动量/利率水平/回撤/长周期波动（全部只用当前及过去）
    if variant in ("E", "F"):
        for lag in (60, 120):
            frame[f"E:rate_diff{lag}"] = past_rate.diff(lag)
        m60 = past_rate.rolling(60, min_periods=30).mean()
        s60 = past_rate.rolling(60, min_periods=30).std()
        frame["E:rate_z60"] = (past_rate - m60) / s60.where(s60.abs() > 1e-9)
        frame["E:rate_dd252"] = past_rate / past_rate.rolling(252, min_periods=60).max() - 1
        frame["E:rate_vol60"] = past_rate.diff().rolling(60, min_periods=30).std()
        for col in ["DR007", "R007", "1Y存单发行利率",
                    "国债期货收盘价(活跃合约):10年", "沪深300", "10年美债", "美元指数"]:
            key = f"daily:{col}"
            if key not in frame:
                continue
            frame[f"E:mom60:{col}"] = frame[key].diff(60)
            mm = frame[key].rolling(60, min_periods=30).mean()
            ss = frame[key].rolling(60, min_periods=30).std()
            frame[f"E:z60:{col}"] = (frame[key] - mm) / ss.where(ss.abs() > 1e-9)
        frame["E:curve10y2y"] = frame.get("daily:10年美债", np.nan)

    # 变体F：在E之上补充资金面/政策锚/机构行为/供给特征（全部只用当前及过去）
    if variant == "F":
        def _col(name):
            return frame[f"daily:{name}"] if f"daily:{name}" in frame \
                else pd.Series(np.nan, index=frame.index)
        omo = _col("逆回购利率:7天")
        frame["F:rate_vs_omo"] = past_rate - omo          # 利率相对政策锚的利差
        frame["F:omo_chg"] = omo.diff()                    # 政策利率调整事件
        fr007 = _col("FR007:1年")
        frame["F:swap_gap"] = fr007 - _col("DR007")        # 资金利率预期
        frame["F:swap_slope20"] = fr007.diff(20)
        for col in ["1年信用利差", "3年信用利差", "5年信用利差", "股债相对"]:
            key = f"daily:{col}"
            if key in frame:
                frame[f"F:{col}:d20"] = frame[key].diff(20)
        for col in ["基金", "理财", "保险", "货币基金", "券商自营"]:
            key = f"daily:净买入：{col}"
            if key in frame:
                frame[f"F:flow:{col}:ma5"] = frame[key].rolling(5, min_periods=3).mean()
                frame[f"F:flow:{col}:ma20"] = frame[key].rolling(20, min_periods=10).mean()
        if "daily:国债期货持仓量:10年" in frame:
            frame["F:oi:d20"] = frame["daily:国债期货持仓量:10年"].diff(20)
            frame["F:oi:ma5"] = frame["daily:国债期货持仓量:10年"].rolling(5, min_periods=3).mean()
        if "daily:债券净融资" in frame:
            frame["F:issue:ma20"] = frame["daily:债券净融资"].rolling(20, min_periods=10).mean()
            frame["F:issue:d20"] = frame["daily:债券净融资"].diff(20)
        v = _col("十年国债活跃券成交量")
        mu, sd = v.rolling(60, min_periods=30).mean(), v.rolling(60, min_periods=30).std()
        frame["F:turn_z60"] = (v - mu) / sd.where(sd.abs() > 1e-9)
        frame["F:rr_chg"] = _col("存款准备金率:中小金融机构(变动公告日期)").diff()  # 降准事件

    frame = b5.causal_rolling_features(frame)
    if event_mode != "none":
        event_features = text2.build_event_features(dates, text2.read_events(), event_mode)
        # 已按可用交易日对齐，不再shift；折内填补/标准化沿用同一训练流程。
        frame = pd.concat([frame, event_features], axis=1)

    target = rate["rate"].shift(-horizon) - rate["rate"]
    maturity = rate["date"].shift(-horizon)
    start = history_start if variant in ("D", "E", "F") else "2018-01-01"
    mask = (dates >= start) & target.notna() & (dates >= daily.index.min())
    positions = np.flatnonzero(mask)
    frame = frame.iloc[positions].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    delta = target.iloc[positions].to_numpy()
    return (frame, delta,
            dates.iloc[positions].reset_index(drop=True),
            maturity.iloc[positions].reset_index(drop=True))


def make_targets(delta):
    return {
        "down": (delta < 0).astype(int),
        "down_ex2": (delta < -DEAD_BP / 100.0).astype(int),   # Δ<-2bp 才算下行
        "big": (np.abs(delta) >= DEAD_BP / 100.0).astype(int),
    }


# =====================================================================
# 指标
# =====================================================================
def seg_metrics(y, pred_label, proba, keep=None):
    y = np.asarray(y); pred_label = np.asarray(pred_label); proba = np.asarray(proba)
    if keep is None:
        keep = np.ones(len(y), dtype=bool)
    yk, pk, bk = y[keep], pred_label[keep], proba[keep]
    n = int(keep.sum())
    if n == 0:
        return {"n": 0}
    out = {"n": n, "coverage": float(n / len(y)),
           "acc": float(accuracy_score(yk, pk)),
           "macro_f1": float(f1_score(yk, pk, average="macro", zero_division=0)),
           "pred_down": float(pk.mean()), "true_down": float(yk.mean()),
           "majority_baseline": float(max(yk.mean(), 1 - yk.mean()))}
    out["auc"] = float(roc_auc_score(yk, bk)) if len(np.unique(yk)) > 1 else float("nan")
    out["vs_majority"] = out["acc"] - out["majority_baseline"]
    return out


# =====================================================================
# 单任务训练
# =====================================================================
def run_task(task, args):
    spec = TASKS[task]
    variant, label_mode = spec["variant"], spec["label"]
    twostage = spec.get("twostage", False)

    # core 配置
    for key in SPEC_TRAIN_KEYS:            # 任务级覆盖优先
        if key in spec:
            setattr(args, key, spec[key])
    argv = sys.argv
    sys.argv = [argv[0]]
    cfg = core.parse_args()
    sys.argv = argv
    cfg.horizon, cfg.seq_len, cfg.hidden_size = 10, args.seq_len, args.hidden_size
    cfg.lr, cfg.epochs, cfg.patience = args.lr, args.epochs, 8
    cfg.dropout, cfg.max_train_days = args.dropout, args.max_train_days
    cfg.recency_halflife = getattr(args, "recency_halflife", 0)
    cfg.state_similarity = getattr(args, "state_similarity", 0)
    cfg.label_smoothing, cfg.class_weight = 0.0, False
    cfg.n_seeds = args.n_seeds
    cfg.threshold_mode, cfg.decision_threshold = "fixed", 0.5
    cfg.val_rate = 0.2
    core.BiGRUInceptionAttentionModel = b5.CompactGRU

    event_mode = spec.get("event_mode", "none")
    frame, delta, dates, maturity = build_dataset(
        variant, cfg.horizon, args.history_start, event_mode)
    tgt = make_targets(delta)
    y = tgt[label_mode] if not twostage else tgt["down"]
    w = np.ones(len(y), dtype=float)

    tag = f"OPT9_{task}"
    print(f"[{tag}] 变体{variant} 标签{label_mode} 两阶段={twostage} "
          f"{len(y)}行 {frame.shape[1]}列 {dates.min().date()}~{dates.max().date()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    raw = frame.to_numpy(dtype=float)
    folds = core.load_folds(cfg)
    if args.max_folds:
        folds = folds.iloc[:args.max_folds]

    predictions = []
    for _, fd in folds.iterrows():
        s = dates.searchsorted(fd["date_start"])
        e = dates.searchsorted(fd["date_end"], side="right")
        if s >= len(y) or e <= s:
            continue
        hi = int(maturity.searchsorted(dates.iloc[s], side="left")) - 1
        lo = 0 if cfg.max_train_days <= 0 else max(0, s - cfg.max_train_days)
        idx = core.seq_indices(len(y), lo, hi, cfg.seq_len)
        n_val = max(8, int(len(idx) * cfg.val_rate))
        gap = max(cfg.seq_len, cfg.horizon)
        cfg.purge_gap = gap
        train_idx = idx[:-(n_val + gap)]
        if len(train_idx) < 20:
            print(f"[{tag}] 跳过 fold{fd['fold']}: 训练不足", flush=True)
            continue
        va_idx = idx[-n_val:]
        assert maturity.iloc[train_idx[-1]] < dates.iloc[va_idx[0]]
        assert maturity.iloc[hi] < dates.iloc[s]
        scale_rows = np.arange(max(0, train_idx[0] - cfg.seq_len + 1), train_idx[-1] + 1)
        xs, keep, _scaler = b5.standardize(raw, scale_rows)
        xt = torch.from_numpy(xs)
        end_idx = np.arange(max(s, cfg.seq_len - 1), e)
        seed_dir, seed_big = [], []
        for seed in range(cfg.n_seeds):
            model, info = core.train_one_fold(cfg, xt, y, w, lo, hi, device,
                                             xs.shape[1], cfg.seed + 1000 * seed)
            seed_dir.append(core.predict_proba(model, xt, end_idx, cfg, device))
            if twostage:
                mb, _ = core.train_one_fold(cfg, xt, tgt["big"], w, lo, hi, device,
                                            xs.shape[1], cfg.seed + 7000 * seed)
                seed_big.append(core.predict_proba(mb, xt, end_idx, cfg, device))
                del mb
            print(f"[{tag}] fold={fd['fold']} seed={seed} ep={info.get('best_ep')}", flush=True)
            del model
        p_dir = np.mean(seed_dir, axis=0)
        for j, row in enumerate(end_idx):
            item = {"date": str(dates.iloc[row].date()), "y_true": int(y[row]),
                    "proba_down": float(p_dir[j]), "pred": int(p_dir[j] >= 0.5),
                    "threshold": 0.5, "fold": int(fd["fold"])}
            item.update({f"seed{k}": float(pb[j]) for k, pb in enumerate(seed_dir)})
            if twostage:
                item["proba_big"] = float(np.mean([pb[j] for pb in seed_big]))
            predictions.append(item)
        pd.DataFrame(predictions).to_csv(OUT / f"{tag}_predictions.csv",
                                         index=False, encoding="utf-8-sig")
    if not predictions:
        raise RuntimeError("无有效测试预测")

    pr = pd.DataFrame(predictions)
    pr["date"] = pd.to_datetime(pr["date"])
    pr = pr.sort_values("date").reset_index(drop=True)
    mat_map = dict(zip(dates, maturity))
    pr["maturity"] = pr["date"].map(mat_map)
    pr["dev"] = (pr["date"] <= CUTOFF) & (pr["maturity"] <= CUTOFF)
    yv = pr["y_true"].to_numpy(); pv = pr["proba_down"].to_numpy()
    predv = (pv >= 0.5).astype(int)
    y2025 = (pr["date"].dt.year == 2025).to_numpy()
    dev_m = pr["dev"].to_numpy()

    report = {
        "task": task, "variant": variant, "label": label_mode, "twostage": twostage,
        "rows": len(pr), "features": int(frame.shape[1]),
        "max_train_days": int(cfg.max_train_days),
        "recency_halflife": int(cfg.recency_halflife),
        "state_similarity": int(cfg.state_similarity),
        "training": {"n_seeds": cfg.n_seeds, "epochs": cfg.epochs,
                     "seq_len": cfg.seq_len, "hidden_size": cfg.hidden_size,
                     "lr": cfg.lr, "dropout": cfg.dropout, "seed": cfg.seed,
                     "history_start": args.history_start},
        "event_mode": event_mode,
        "dev": seg_metrics(yv[dev_m], predv[dev_m], pv[dev_m]),
        "full": seg_metrics(yv, predv, pv),
        "2025": seg_metrics(yv[y2025], predv[y2025], pv[y2025]),
    }
    if event_mode != "none":
        events = text2.read_events()
        report["events"] = {
            "path": str(text2.EVENT_CSV),
            "sha256": hashlib.sha256(text2.EVENT_CSV.read_bytes()).hexdigest(),
            "count": len(events), "first_date": str(events["date"].min().date()),
            "last_date": str(events["date"].max().date()),
            "half_life": text2.HALF_LIFE, "max_gap": text2.MAX_GAP,
            "availability": "next_trading_day",
            "features": [c for c in frame if c.startswith("txt_")],
            "limitation": "回溯精选日历，未验证标注的当时可得性；2018年前无事件覆盖",
        }

    # 两阶段：用开发期选 gate 阈值，再套用
    if twostage:
        pb = pr["proba_big"].to_numpy()
        dev_keep = dev_m.copy()
        best = None
        for tau in np.round(np.arange(0.30, 0.705, 0.025), 3):
            keep = dev_keep & (pb >= tau)
            m = seg_metrics(yv[dev_keep], predv[dev_keep], pv[dev_keep], (pb[dev_keep] >= tau))
            if m["n"] < 100:
                continue
            if best is None or m["acc"] > best[1]["acc"]:
                best = (float(tau), m, keep)
        if best is not None:
            tau = best[0]
            report["gate_tau"] = tau
            report["gate_dev"] = best[1]
            report["gate_full"] = seg_metrics(yv, predv, pv, pb >= tau)
            report["gate_2025"] = seg_metrics(yv[y2025], predv[y2025], pv[y2025], pb[y2025] >= tau)
        big_map = dict(zip(dates, tgt["big"]))
        big_true = pr["date"].map(big_map).to_numpy()
        report["big_auc_dev"] = float(roc_auc_score(big_true[dev_m],
                                                    pb[dev_m])) if len(np.unique(big_true[dev_m])) > 1 else None

    pr.to_csv(OUT / f"{tag}_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(m):
        if m.get("n", 0) == 0:
            return "n=0（该区间无样本）"
        return (f"Acc={m['acc']:.4f}(vs多数类{m.get('vs_majority', 0):+.4f}) "
                f"AUC={m['auc']:.4f} n={m['n']} 覆盖={m.get('coverage', 1):.2f}")
    print(f"[{tag}] 开发期 {fmt(report['dev'])}", flush=True)
    print(f"[{tag}] 全期   {fmt(report['full'])}", flush=True)
    print(f"[{tag}] 2025   {fmt(report['2025'])}", flush=True)
    if twostage and "gate_tau" in report:
        print(f"[{tag}] 两阶段 tau={report['gate_tau']} 开发期 {fmt(report['gate_dev'])}", flush=True)
        print(f"[{tag}] 两阶段 2025 {fmt(report['gate_2025'])}", flush=True)
        print(f"[{tag}] Stage1 big_auc(dev)={report['big_auc_dev']}", flush=True)
    return report


def write_comparison(summary):
    rows = []
    for tag, rep in summary.items():
        if "full" not in rep:
            rows.append({"实验": tag, "状态": rep.get("status")})
            continue
        row = {"实验": tag, "标签": rep["label"], "特征数": rep["features"],
               "事件模式": rep.get("event_mode", "none"),
               "窗口": rep.get("max_train_days"), "半衰期": rep.get("recency_halflife"),
               "状态相似": rep.get("state_similarity")}
        for seg in ("dev", "full", "2025", "original_dev", "original_2025",
                    "gate_dev", "gate_2025"):
            for metric in ("acc", "vs_majority", "auc", "n", "coverage"):
                row[f"{seg}_{metric}"] = rep.get(seg, {}).get(metric)
        row["gate_tau"] = rep.get("gate_tau")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "OPT9_对比汇总.csv", index=False, encoding="utf-8-sig")


def block_interval(values, block=20, repeats=2000):
    """按连续日期分块重采样，保留10日重叠标签的局部相关性。"""
    values = np.asarray(values, dtype=float)
    if len(values) < block:
        return None
    rng = np.random.default_rng(42)
    offsets = np.arange(block)
    means = []
    for _ in range(repeats):
        starts = rng.integers(0, len(values) - block + 1,
                              size=int(np.ceil(len(values) / block)))
        sampled = values[(starts[:, None] + offsets).ravel()[:len(values)]]
        means.append(sampled.mean())
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize_saved():
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"])
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    rate = rate.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    rate["delta"] = rate["rate"].shift(-10) - rate["rate"]
    rate["maturity"] = rate["date"].shift(-10)
    rate = rate.set_index("date")
    summary, audit = {}, {}
    baseline = pd.read_csv(OUT / "OPT9_base_D_predictions.csv", parse_dates=["date"])
    baseline = baseline.set_index("date")["pred"]
    event_reference = pd.read_csv(
        OUT / "OPT9_featrelab_E_predictions.csv", parse_dates=["date"]).set_index("date")
    event_days = text2.build_trading_days()
    events = text2.read_events()
    event_density = text2.build_event_features(
        event_days, events, "objective")["txt_cnt20"]
    for task in TASKS:
        tag = f"OPT9_{task}"
        path = OUT / f"{tag}_metrics.json"
        if not path.exists():
            summary[tag] = {"status": "pending"}
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        pr = pd.read_csv(OUT / f"{tag}_predictions.csv", parse_dates=["date"])
        if len(pr) != rep["rows"] or pr["date"].duplicated().any():
            raise ValueError(f"{tag}: 预测产物不完整或日期重复")
        delta = pr["date"].map(rate["delta"]).to_numpy()
        maturity = pr["date"].map(rate["maturity"])
        if not np.isfinite(delta).all() or maturity.isna().any():
            raise ValueError(f"{tag}: 标签日期无法对齐")
        targets = make_targets(delta)
        np.testing.assert_array_equal(pr["y_true"], targets[rep["label"]])
        dev = ((pr["date"] <= CUTOFF) & (maturity <= CUTOFF)).to_numpy()
        masks = {"dev": dev, "2025": (pr["date"].dt.year == 2025).to_numpy()}
        detail = {"by_segment": {}, "by_year": {}, "by_seed": {}, "months_2025": {},
                  "original_by_year": {}}
        p = pr["proba_down"].to_numpy()
        pred = (p >= 0.5).astype(int)
        for name, mask in masks.items():
            y = targets[rep["label"]][mask]
            correct = (pred[mask] == y).astype(float)
            majority = max(y.mean(), 1 - y.mean()) if len(y) else np.nan
            detail["by_segment"][name] = {
                "native": seg_metrics(y, pred[mask], p[mask]),
                "original_direction": seg_metrics(targets["down"][mask], pred[mask], p[mask]),
                "accuracy_ci_block20": block_interval(correct),
                "vs_majority_ci_block20": block_interval(
                    correct - (y == int(y.mean() >= 0.5)).astype(float)),
            }
            original_y = targets["down"][mask]
            original_correct = (pred[mask] == original_y).astype(float)
            baseline_pred = pr["date"].map(baseline).to_numpy()[mask]
            if np.isnan(baseline_pred).any():
                raise ValueError(f"{tag}: 基线预测日期缺失")
            detail["by_segment"][name]["original_vs_baseline_ci_block20"] = block_interval(
                original_correct - (baseline_pred == original_y).astype(float))
            detail["by_segment"][name]["original_vs_majority_ci_block20"] = block_interval(
                original_correct - (original_y == int(original_y.mean() >= 0.5)).astype(float))
            reference = event_reference.reindex(pr["date"])
            if reference["pred"].isna().any():
                raise ValueError(f"{tag}: E基线预测日期缺失")
            reference_correct = (
                reference["pred"].to_numpy()[mask] == original_y).astype(float)
            detail["by_segment"][name]["original_vs_E"] = float(
                np.mean(original_correct - reference_correct))
            detail["by_segment"][name]["original_vs_E_ci_block20"] = block_interval(
                original_correct - reference_correct)
            print(f"{task} {name}: native={correct.mean():.4f} "
                  f"original={detail['by_segment'][name]['original_direction']['acc']:.4f} "
                  f"native_majority={majority:.4f} "
                  f"original_majority={detail['by_segment'][name]['original_direction']['majority_baseline']:.4f}",
                  flush=True)
        for year in sorted(pr["date"].dt.year.unique()):
            mask = (pr["date"].dt.year == year).to_numpy()
            detail["by_year"][str(year)] = seg_metrics(
                targets[rep["label"]][mask], pred[mask], p[mask])
            detail["original_by_year"][str(year)] = seg_metrics(
                targets["down"][mask], pred[mask], p[mask])
        for col in [c for c in pr if c.startswith("seed")]:
            sp = pr[col].to_numpy()
            detail["by_seed"][col] = {
                name: seg_metrics(targets[rep["label"]][mask], (sp[mask] >= 0.5).astype(int), sp[mask])
                for name, mask in masks.items()
            }
            detail["by_seed"][col]["original_2025"] = seg_metrics(
                targets["down"][masks["2025"]],
                (sp[masks["2025"]] >= 0.5).astype(int), sp[masks["2025"]])
            detail["by_seed"][col]["original_dev"] = seg_metrics(
                targets["down"][dev], (sp[dev] >= 0.5).astype(int), sp[dev])
            reference_seed = event_reference.reindex(pr["date"])[col].to_numpy()
            detail["by_seed"][col]["original_dev_vs_E"] = float(
                np.mean(((sp[dev] >= 0.5) == targets["down"][dev]).astype(float)
                        - ((reference_seed[dev] >= 0.5) == targets["down"][dev]).astype(float)))
        for month in sorted(pr.loc[masks["2025"], "date"].dt.month.unique()):
            mask = masks["2025"] & (pr["date"].dt.month == month).to_numpy()
            detail["months_2025"][str(month)] = seg_metrics(
                targets["down"][mask], pred[mask], p[mask])
        density = pr["date"].map(event_density).to_numpy()
        if not np.isfinite(density).all():
            raise ValueError(f"{tag}: 事件窗口日期无法对齐")
        detail["event_windows"] = {}
        for name, segment in masks.items():
            covered = segment & (pr["date"] > events["date"].min()).to_numpy()
            for label, window in [("covered", covered),
                                  ("after_event20", covered & (density > 0)),
                                  ("no_event20", covered & (density == 0))]:
                detail["event_windows"][f"{name}_{label}"] = seg_metrics(
                    targets["down"][segment], pred[segment], p[segment],
                    keep=window[segment])
        if "gate_tau" in rep:
            detail["gate_original_2025"] = seg_metrics(
                targets["down"][masks["2025"]], pred[masks["2025"]], p[masks["2025"]],
                pr["proba_big"].to_numpy()[masks["2025"]] >= rep["gate_tau"])
        rep["original_dev"] = detail["by_segment"]["dev"]["original_direction"]
        rep["original_2025"] = detail["by_segment"]["2025"]["original_direction"]
        summary[tag], audit[tag] = rep, detail
    (OUT / "OPT9_suite_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "OPT9_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_comparison(summary)
    # 采用规则预先固定：开发期同口径超多数类，配对增益区间为正且多数seed改善。
    decisions = {}
    for task in ("event_objective_E", "event_direction_E"):
        tag = f"OPT9_{task}"
        if tag not in audit:
            decisions[tag] = {"status": "pending"}
            continue
        detail = audit[tag]
        dev_result = detail["by_segment"]["dev"]
        interval = dev_result["original_vs_E_ci_block20"]
        seed_improvements = [
            r["original_dev_vs_E"] > 0 for r in detail["by_seed"].values()]
        checks = {
            "original_dev_above_majority": dev_result["original_direction"]["vs_majority"] > 0,
            "paired_dev_interval_positive": interval is not None and interval[0] > 0,
            "majority_of_seeds_improve": sum(seed_improvements) > len(seed_improvements) / 2,
        }
        decisions[tag] = {"adopt": all(checks.values()), "checks": checks,
                          "original_dev_vs_E": dev_result["original_vs_E"],
                          "paired_dev_interval": interval}
    (OUT / "OPT9_event_decision.json").write_text(json.dumps({
        "selection": "仅开发期原始方向；2025只作诊断，不用于采用判断",
        "limitation": "精选回溯事件标注不能视为已验证的实时新闻数据",
        "candidates": decisions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def run_suite(args):
    OUT.mkdir(exist_ok=True)
    summary = {}
    only = {t.strip() for t in args.only.split(",") if t.strip()}
    for task in TASKS:
        if only and task not in only:
            continue
        tag = f"OPT9_{task}"
        spec = TASKS[task]
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--task", task, "--n-seeds", str(args.n_seeds),
                   "--epochs", str(args.epochs), "--history-start", args.history_start,
                   "--seq-len", str(args.seq_len), "--hidden-size", str(args.hidden_size),
                   "--lr", str(args.lr), "--dropout", str(args.dropout),
                   "--max-folds", str(args.max_folds),
                   "--max-train-days", str(spec.get("max_train_days", args.max_train_days)),
                   "--recency-halflife", str(spec.get("recency_halflife", args.recency_halflife)),
                   "--state-similarity", str(spec.get("state_similarity", args.state_similarity))]
        print(f"启动 {tag}", flush=True)
        with (OUT / f"{tag}_run.log").open("w", encoding="utf-8") as log:
            r = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode:
            summary[tag] = {"status": "failed", "exit_code": r.returncode}
        else:
            summary[tag] = json.loads((OUT / f"{tag}_metrics.json").read_text(encoding="utf-8"))
        print(f"结束 {tag}: {summary[tag].get('full', summary[tag])}", flush=True)
        (OUT / "OPT9_suite_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 汇总对比表
    rows = []
    for tag, rep in summary.items():
        if "full" not in rep:
            rows.append({"实验": tag, "状态": rep.get("status")})
            continue
        row = {"实验": tag, "变体": rep["variant"], "标签": rep["label"],
               "两阶段": rep["twostage"], "特征数": rep["features"],
               "窗口": rep.get("max_train_days"), "半衰期": rep.get("recency_halflife"),
               "状态相似": rep.get("state_similarity")}
        for seg in ["dev", "full", "2025"]:
            m = rep[seg]
            row[f"{seg}_Acc"] = m.get("acc"); row[f"{seg}_vs多数类"] = m.get("vs_majority")
            row[f"{seg}_AUC"] = m.get("auc"); row[f"{seg}_n"] = m.get("n")
        if rep.get("twostage") and "gate_2025" in rep:
            row["gate_tau"] = rep.get("gate_tau")
            row["gate_dev_Acc"] = rep["gate_dev"].get("acc")
            row["gate_dev_覆盖"] = rep["gate_dev"].get("coverage")
            row["gate_2025_Acc"] = rep["gate_2025"].get("acc")
            row["gate_2025_覆盖"] = rep["gate_2025"].get("coverage")
            row["stage1_bigAUC_dev"] = rep.get("big_auc_dev")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "OPT9_对比汇总.csv", index=False, encoding="utf-8-sig")
    print("已生成 OPT9_对比汇总.csv", flush=True)
    if any(rep.get("status") == "failed" for rep in summary.values()):
        raise RuntimeError("部分实验失败，请查看各任务日志")
    if not args.max_folds:
        summarize_saved()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=list(TASKS), default="base_D")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-train-days", type=int, default=0)
    ap.add_argument("--recency-halflife", type=int, default=0)
    ap.add_argument("--state-similarity", type=int, default=0)
    ap.add_argument("--only", default="", help="suite模式下仅运行指定任务(逗号分隔)")
    ap.add_argument("--history-start", default="2010-01-01")
    ap.add_argument("--max-folds", type=int, default=0)
    args = ap.parse_args()
    if args.summarize_only:
        return summarize_saved()
    if args.suite:
        return run_suite(args)
    run_task(args.task, args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT.mkdir(exist_ok=True)
        (OUT / "OPT9_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
