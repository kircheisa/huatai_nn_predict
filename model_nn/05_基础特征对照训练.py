# -*- coding: utf-8 -*-
"""原始工作簿因果特征对照：B基础、C因果衍生、D扩展历史；不覆盖旧实验。"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.append(str(HERE / "_libs"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("nn_core", HERE / "01_BiGRU训练_二分类_10天.py")
pre = load_module("pre_core", ROOT / "model_new" / "01_数据预处理.py")
ORIGINAL_MODEL = core.BiGRUInceptionAttentionModel


class CompactGRU(torch.nn.Module):
    """小样本对照：特征投影、单层双向GRU、均值/末端联合池化。"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3, n_blocks=1):
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                           torch.nn.LayerNorm(hidden_size), torch.nn.GELU())
        self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                        torch.nn.Linear(hidden_size * 4, 2))

    def forward(self, x):
        out, _ = self.gru(self.project(x))
        return self.head(torch.cat([out.mean(1), out[:, -1]], dim=1))


class ScaledAttention(core.SelfAttention):
    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.shape[-1])
        return torch.bmm(self.softmax(scores), v)


def causal_rolling_features(frame, window=252):
    # 每日统计只使用前一日及更早的可用特征；不访问未来数据。
    past = frame.shift(1)
    med = past.rolling(window, min_periods=30).median()
    scale = (past.rolling(window, min_periods=30).quantile(0.75)
             - past.rolling(window, min_periods=30).quantile(0.25))
    return ((frame - med) / scale.where(scale.abs() > 1e-6)).clip(-8, 8)


def sheet(book, index):
    frame = pd.read_excel(book, sheet_name=index)
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").set_index("date")


def align_published(series, published, dates):
    tmp = pd.DataFrame({"available": pd.to_datetime(published), "value": series.to_numpy()})
    # 未知日内发布时间：从下一日开始使用，避免同日收盘/海外时区泄漏。
    tmp["available"] += pd.Timedelta(days=1)
    tmp = tmp.dropna().sort_values("available").drop_duplicates("available", keep="last")
    if tmp.empty:
        return np.full(len(dates), np.nan)
    joined = pd.merge_asof(pd.DataFrame({"date": dates}), tmp,
                           left_on="date", right_on="available", direction="backward")
    available = joined["available"].notna()
    assert (joined.loc[available, "available"] <= joined.loc[available, "date"]).all()
    return joined["value"].to_numpy()


def build_data(args):
    if args.variant == "A":
        config = argparse.Namespace(horizon=args.horizon, start_date="2018-01-01",
                                    deadzone_bp=2, full_features=False, top_k=30)
        features, y, weights, dates = core.load_aligned_data(config)
        # 仅作旧缓存参考；其特征构建/筛选不满足新的时间因果性要求。
        book_rate = pd.read_csv(ROOT / "model_new" / "data" / "rate.csv")
        book_rate["time"] = pd.to_datetime(book_rate["time"])
        data_dates = pd.read_csv(ROOT / "model_new" / "data" / "data.csv", usecols=["Unnamed: 0"])
        data_dates["Unnamed: 0"] = pd.to_datetime(data_dates["Unnamed: 0"])
        calendar = pd.merge(data_dates, book_rate, left_on="Unnamed: 0", right_on="time").sort_values("time")
        mature = pd.Series(calendar.time.shift(-args.horizon).to_numpy(),
                           index=calendar.time).reindex(pd.to_datetime(dates)).reset_index(drop=True)
        return features, y, weights, dates, mature
    book = pd.ExcelFile(ROOT / "data_ori" / "2026_origin_data.xlsx")
    rate = pd.read_csv(ROOT / "data_ori" / "2026_origin_rate.csv").iloc[:, :2]
    rate.columns = ["date", "rate"]
    rate["date"] = pd.to_datetime(rate["date"], errors="coerce")
    rate["rate"] = pd.to_numeric(rate["rate"], errors="coerce")
    rate = rate.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    daily = sheet(book, 5)
    dates = rate["date"]
    features = {}
    for index, prefix in [(5, "daily"), (4, "weekly"), (6, "policy")]:
        data = sheet(book, index)
        for col in data:
            series = pd.to_numeric(data[col], errors="coerce")
            features[f"{prefix}:{col}"] = align_published(series, data.index, dates)
    for index, pub_index, mapping, prefix in [
        (0, 1, {"GDP:不变价:当季同比": "GDP同比",
                "中国:工业产能利用率:当季值": "产能利用率"}, "quarter"),
        (2, 3, pre.MONTHLY_PUBLISH_MAP, "month"),
    ]:
        data, pub = sheet(book, index), sheet(book, pub_index)
        for col, release in mapping.items():
            if col not in data or release is None or release not in pub:
                continue
            series = pd.to_numeric(data[col], errors="coerce")
            published = pub[release].reindex(data.index)
            features[f"{prefix}:{col}"] = align_published(series, published, dates)
    frame = pd.DataFrame(features, index=dates)
    if args.variant in ("C", "D"):
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
    if getattr(args, "normalization", "fold") == "rolling":
        frame = causal_rolling_features(frame)
    target = rate["rate"].shift(-args.horizon) - rate["rate"]
    maturity = rate["date"].shift(-args.horizon)
    start = args.history_start if args.variant == "D" else "2018-01-01"
    mask = (dates >= start) & target.notna() & (dates >= daily.index.min())
    positions = np.flatnonzero(mask)
    frame = frame.iloc[positions].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    delta = target.iloc[positions].to_numpy()
    return (frame, (delta < 0).astype(np.int64),
            np.where(np.abs(delta) * 100 < 2, 0.3, 1.0),
            dates.iloc[positions].reset_index(drop=True),
            maturity.iloc[positions].reset_index(drop=True))


def standardize(raw, train_rows):
    train = raw[train_rows]
    # 只在梯度训练段拟合，验证/测试不参与填补、筛列或标准化。
    keep = np.isfinite(train).sum(axis=0) >= 20
    x = raw[:, keep]
    train = train[:, keep]
    med = np.nanmedian(train, axis=0)
    missing = ~np.isfinite(x)
    filled = np.where(missing, med, x)
    train_filled = np.where(np.isfinite(train), train, med)
    mu, sd = train_filled.mean(axis=0), train_filled.std(axis=0)
    scaled = np.clip((filled - mu) / np.maximum(sd, 1e-6), -8, 8)
    return (np.concatenate([scaled, missing.astype(float)], axis=1).astype(np.float32),
            keep, {"median": med, "mean": mu, "std": sd})


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    assert np.isclose(acc, np.mean(pred == y))
    return {"n": len(y), "acc": acc,
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else None,
            "brier": brier_score_loss(y, p), "pred_down": float(pred.mean()),
            "true_down": float(np.mean(y)),
            "majority_baseline": float(max(np.mean(y), 1 - np.mean(y)))}


def run_suite(args):
    OUT.mkdir(exist_ok=True)
    summary = {}
    # 串行使用GPU，每个子进程直接写日志，无无人读取的stdout管道。
    experiments = [("B", "nn"), ("C", "nn"), ("D", "nn"),
                   ("A", "nn"), ("B", "xgb")]
    for variant, model in experiments:
        tag = f"RAW_{variant}_{model}_S{args.n_seeds}_E{args.epochs}"
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--variant", variant, "--model", model, "--tag", tag,
                   "--n-seeds", str(args.n_seeds), "--epochs", str(args.epochs),
                   "--history-start", args.history_start]
        if args.max_folds:
            command += ["--max-folds", str(args.max_folds)]
        print(f"启动 {tag}", flush=True)
        with (OUT / f"{tag}_run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            summary[tag] = {"status": "failed", "exit_code": result.returncode}
        else:
            summary[tag] = json.loads((OUT / f"{tag}_metrics.json").read_text(encoding="utf-8"))
        (OUT / "RAW_suite_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结束 {tag}: {summary[tag].get('global', summary[tag])}", flush=True)
    if any(item.get("status") == "failed" for item in summary.values()):
        raise RuntimeError("部分实验失败，请查看各组日志")


def optimize(args):
    """预先列出有限候选，用截至2024年的滚动样本外评分选择，最后全期确认。"""
    candidates = [
        {"variant": "B", "architecture": "compact", "seq_len": 20, "hidden_size": 32, "normalization": "fold"},
        {"variant": "B", "architecture": "compact", "seq_len": 20, "hidden_size": 32},
        {"variant": "B", "architecture": "compact", "seq_len": 40, "hidden_size": 32},
        {"variant": "C", "architecture": "compact", "seq_len": 20, "hidden_size": 32},
        {"variant": "C", "architecture": "compact", "seq_len": 40, "hidden_size": 64},
        {"variant": "D", "architecture": "compact", "seq_len": 20, "hidden_size": 32, "max_train_days": 0},
        {"variant": "B", "architecture": "compact", "seq_len": 20, "hidden_size": 32, "max_train_days": 504},
        {"variant": "B", "architecture": "scaled", "seq_len": 40, "hidden_size": 32},
    ]
    defaults = {"normalization": "rolling", "lr": 5e-4, "max_train_days": 1260,
                "dropout": 0.3, "label_smoothing": 0.0, "no_class_weight": True,
                "uniform_weights": True}
    manifest = {"selection": "截至2024年开发期，score=(AUC+MacroF1)/2；2025未用于选型",
                "candidates": [{**defaults, **item} for item in candidates]}
    (OUT / "RAW_optimization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ranking = []

    def execute(tag, params, seeds, development):
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--tag", tag, "--n-seeds", str(seeds), "--epochs", str(args.epochs)]
        for key, value in params.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command += [flag, str(value)]
        if development:
            command += ["--test-end", "2024-12-31"]
        print(f"开始 {tag}", flush=True)
        with (OUT / f"{tag}_run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            raise RuntimeError(f"{tag}失败，见日志")
        return json.loads((OUT / f"{tag}_metrics.json").read_text(encoding="utf-8"))

    for i, params in enumerate(manifest["candidates"]):
        tag = f"RAW_OPT_DEV{i:02d}_E{args.epochs}"
        report = execute(tag, params, 1, True)
        g = report["global"]
        ranking.append({"tag": tag, "params": params, "metrics": g,
                        "score": (g["auc"] + g["macro_f1"]) / 2})
        ranking.sort(key=lambda item: item["score"], reverse=True)
        (OUT / "RAW_optimization_ranking.json").write_text(
            json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完成 {tag}: {g}", flush=True)
    winner = ranking[0]
    tag = f"RAW_OPT_FINAL_S{args.n_seeds}_E{args.epochs}"
    final = execute(tag, winner["params"], args.n_seeds, False)
    result = {"winner": winner, "final_tag": tag, "final": final,
              "warning": "2025已被历史迭代观察，属于本轮未参与选型的回测，非全新盲测"}
    (OUT / "RAW_optimization_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--model", choices=["nn", "xgb"], default="nn")
    parser.add_argument("--history-start", default="2010-01-01")
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--suite", action="store_true", help="串行运行B/C/D/A参考与10天XGB")
    parser.add_argument("--tag", default="")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--architecture", choices=["original", "compact", "scaled"], default="original")
    parser.add_argument("--normalization", choices=["fold", "rolling"], default="fold")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-train-days", type=int, default=1260)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--uniform-weights", action="store_true")
    parser.add_argument("--test-end", default="")
    args = parser.parse_args()
    if args.n_seeds < 1 or args.epochs < 1:
        parser.error("n-seeds和epochs必须为正整数")
    if args.suite:
        return run_suite(args)
    if args.optimize:
        return optimize(args)
    # 沿用结构/损失，首轮只改变输入与可用历史，不同时改注意力实现。
    argv = sys.argv
    sys.argv = [argv[0]]
    cfg = core.parse_args()
    sys.argv = argv
    cfg.horizon, cfg.seq_len, cfg.hidden_size = 10, args.seq_len, args.hidden_size
    cfg.lr, cfg.epochs, cfg.patience = args.lr, args.epochs, 8
    cfg.dropout, cfg.max_train_days = args.dropout, args.max_train_days
    cfg.label_smoothing, cfg.class_weight = args.label_smoothing, not args.no_class_weight
    if args.architecture == "compact":
        core.BiGRUInceptionAttentionModel = CompactGRU
    elif args.architecture == "scaled":
        core.SelfAttention = ScaledAttention
    cfg.n_seeds = args.n_seeds
    cfg.threshold_mode, cfg.decision_threshold = "fixed", 0.5
    args.horizon = cfg.horizon
    tag = args.tag or f"RAW_{args.variant}_{args.model}_S{args.n_seeds}"
    started = time.time()
    feats, y, w, dates, maturity = build_data(args)
    if args.uniform_weights:
        w = np.ones_like(w)
    print(f"数据 {args.variant}: {len(y)}行 {feats.shape[1]}列 "
          f"{dates.min().date()}~{dates.max().date()} 缺失={feats.isna().mean().mean():.3f}", flush=True)
    audit = {"variant": args.variant, "model": args.model, "rows": len(y), "columns": list(feats.columns),
             "policy": "backward asof; availability +1 calendar day; no genetic/importance",
             "excluded": ["未知发布日期月度指标", "时代/事后主题变量"],
             "limitations": ["原始工作簿未提供历史修订版本；周频按记录日期+1日，发布日期待核实"],
             "config": vars(cfg), "experiment_args": vars(args)}
    if args.variant == "A":
        audit["limitations"].append("旧Top30特征含未来匹配/全期筛选风险，仅作历史参考")
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_only:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    raw = feats.to_numpy(dtype=float)
    folds = core.load_folds(cfg)
    if args.max_folds:
        folds = folds.iloc[:args.max_folds]
    predictions, fold_rows = [], []
    eval_limit = len(y)
    if args.test_end:
        cutoff = pd.Timestamp(args.test_end)
        eval_limit = min(dates.searchsorted(cutoff, side="right"),
                         maturity.searchsorted(cutoff, side="right"))
    for _, fd in folds.iterrows():
        s = dates.searchsorted(fd["date_start"])
        e = dates.searchsorted(fd["date_end"], side="right")
        e = min(e, eval_limit)
        if s >= len(y) or e <= s:
            continue
        # 训练标签必须在测试第一日之前成熟，而非直接训练到s-1。
        hi = int(maturity.searchsorted(dates.iloc[s], side="left")) - 1
        lo = 0 if cfg.max_train_days <= 0 else max(0, s - cfg.max_train_days)
        idx = core.seq_indices(len(y), lo, hi, cfg.seq_len)
        n_val = max(8, int(len(idx) * cfg.val_rate))
        gap = max(cfg.seq_len, cfg.horizon)
        cfg.purge_gap = gap
        train_idx = idx[:-(n_val + gap)]
        if len(train_idx) < 20:
            print(f"跳过 fold {fd['fold']}:训练不足", flush=True)
            continue
        va_idx = idx[-n_val:]
        assert maturity.iloc[train_idx[-1]] < dates.iloc[va_idx[0]]
        assert maturity.iloc[hi] < dates.iloc[s]
        scale_rows = np.arange(max(0, train_idx[0] - cfg.seq_len + 1), train_idx[-1] + 1)
        xs, keep, scaler = standardize(raw, scale_rows)
        xt = torch.from_numpy(xs)
        end_idx = np.arange(max(s, cfg.seq_len - 1), e)
        seed_probs = []
        for seed in range(cfg.n_seeds):
            if args.model == "nn":
                model, info = core.train_one_fold(cfg, xt, y, w, lo, hi, device,
                                                  xs.shape[1], cfg.seed + 1000 * seed)
                seed_probs.append(core.predict_proba(model, xt, end_idx, cfg, device))
            else:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                                      random_state=cfg.seed + 1000 * seed, n_jobs=4,
                                      eval_metric="logloss", early_stopping_rounds=30)
                # XGB用当日可用特征；不宣称与NN的40日序列完全同构。
                model.fit(xs[train_idx], y[train_idx], sample_weight=w[train_idx],
                          eval_set=[(xs[va_idx], y[va_idx])], verbose=False)
                seed_probs.append(model.predict_proba(xs[end_idx])[:, 1])
                model.save_model(OUT / f"{tag}_fold{int(fd['fold']):02d}_seed{seed}.ubj")
                info = {"best_ep": model.best_iteration + 1}
            # 保存每折每seed权重及训练统计量，保证训练结果可追溯。
            torch.save({"state_dict": model.state_dict() if args.model == "nn" else None,
                        "feature_columns": list(feats.columns[keep]), "scaler": scaler,
                        "scale_rows": [int(scale_rows[0]), int(scale_rows[-1])],
                        "config": vars(cfg), "experiment_args": vars(args), "best_ep": info["best_ep"]},
                       OUT / f"{tag}_fold{int(fd['fold']):02d}_seed{seed}.pt")
            print(f"{tag} fold={fd['fold']} seed={seed} ep={info['best_ep']}", flush=True)
            del model
        p = np.mean(seed_probs, axis=0)
        result = metrics(y[end_idx], p)
        result.update({"fold": int(fd["fold"]), "features": int(keep.sum()),
                       "date_start": str(dates.iloc[end_idx[0]].date()),
                       "date_end": str(dates.iloc[end_idx[-1]].date())})
        fold_rows.append(result)
        for j, row in enumerate(end_idx):
            item = {"date": str(dates.iloc[row].date()), "y_true": int(y[row]),
                    "proba_down": float(p[j]), "pred": int(p[j] >= 0.5),
                    "threshold": 0.5, "fold": int(fd["fold"])}
            item.update({f"seed{k}": float(pb[j]) for k, pb in enumerate(seed_probs)})
            predictions.append(item)
        pr = pd.DataFrame(predictions)
        pr.to_csv(OUT / f"{tag}_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(OUT / f"{tag}_folds.csv", index=False, encoding="utf-8-sig")
        report = {"global": metrics(pr.y_true.to_numpy(), pr.proba_down.to_numpy()),
                  "years": {str(year): metrics(g.y_true.to_numpy(), g.proba_down.to_numpy())
                            for year, g in pr.groupby(pd.to_datetime(pr.date).dt.year)},
                  "seed_metrics": {str(k): metrics(pr.y_true.to_numpy(), pr[f"seed{k}"].to_numpy())
                                   for k in range(cfg.n_seeds)},
                  "folds_completed": len(fold_rows), "folds_requested": len(folds),
                  "seconds": time.time() - started,
                  "status": "complete" if len(fold_rows) == len(folds) else "partial"}
        (OUT / f"{tag}_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{tag} fold={fd['fold']} {result}", flush=True)
    if not predictions:
        raise RuntimeError("没有有效测试预测")
    # 测试截止日期可能裁掉末尾fold，完整性按实际请求的日期区间计算。
    expected = int(sum(dates.searchsorted(fd.date_start) < eval_limit for fd in folds.itertuples()))
    report["status"] = "complete" if len(fold_rows) == expected else "partial"
    report["folds_requested"] = expected
    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
