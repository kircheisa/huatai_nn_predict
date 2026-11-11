# -*- coding: utf-8 -*-
"""
BiGRU + Inception + Self-Attention（1/sqrt(d) 缩放）—— 新基础数据版
预测10年期国债收益率未来5天方向（二分类：下行=1）

与 01_BiGRU训练_二分类.py 的关系：模型结构、训练协议、评估口径完全一致，
仅数据入口换成新基础数据管线（model_new/20/21 脚本产物）：
  1. 特征: data_new/43v2_features.csv（281列因果特征，2010-01~2026-11）
  2. 标签: data_new/43v2_labels_bin.csv（label/sample_weight，死区0.3已含）
  3. fold: data_new/out/xgbv2_fold_metrics.csv（35 folds，2018-01~2026-10，
           与新数据XGB同fold同测试段，保证两模型可直接对比）
  4. 特征筛选: 默认按新XGB重要性(xgbv2_feature_importance.csv)取Top-K(30)

基线对照（均28/35-fold滚动样本外）：
  新数据XGB:      Acc=0.5236  MacroF1=0.5209  AUC=0.5277
  旧数据NN(BASE_ATTN): Acc=0.5385  MacroF1=0.5361  AUC=0.5745
  旧数据XGB:      Acc=0.5635  MacroF1=0.5627  AUC=0.5822
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
DATA_NEW = BASE / "data_new"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 参数
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--top-k", type=int, default=30,
                   help="按新XGB重要性取Top-K特征（--full-features则忽略）")
    p.add_argument("--full-features", action="store_true",
                   help="使用全部281列特征")
    # 模型结构
    p.add_argument("--seq-len", type=int, default=60, help="滑动窗口长度")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--dropout", type=float, default=0.5)
    # 训练
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--val-rate", type=float, default=0.2)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--threshold-mode", default="fixed",
                   choices=["fixed", "val_macro_f1", "val_acc"])
    # 泛化
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weight", action="store_true", default=True)
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--purge-gap", type=int, default=-1)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-train-days", type=int, default=1260,
                   help="训练窗口上限（交易日），0=expanding全量")
    p.add_argument("--seed", type=int, default=42)
    # 运行
    p.add_argument("--features-file", default="43v2_features.csv",
                   help="特征文件（data_new/下，新版原始数据用43v3_features.csv）")
    p.add_argument("--labels-file", default="43v2_labels_bin.csv",
                   help="标签文件（data_new/下，h10版用43v2_labels_bin_h10.csv）")
    p.add_argument("--folds-file", default="xgbv2_fold_metrics.csv",
                   help="新数据XGB的fold边界（保证两模型测试区间一致）")
    p.add_argument("--importance-file", default="xgbv2_feature_importance.csv",
                   help="特征重要性文件（与folds同前缀）")
    p.add_argument("--fold-importance-file", default="",
                   help="逐fold训练窗重要性CSV(34脚本产物,fold/feature/gain)；"
                        "提供时每fold用自身训练窗的Top-K特征，"
                        "消除全期重要性选特征的选择泄漏")
    p.add_argument("--tag", default="", help="输出文件名前缀")
    return p.parse_args()


# =====================================================================
# 模型结构（与 01_BiGRU训练_二分类.py 完全一致，含 1/sqrt(d) 缩放）
# =====================================================================
class InceptionModule(nn.Module):
    """4分支多尺度1D卷积 + 1x1残差 + BN + MaxPool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 9, padding=4),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * 4, 1),
            nn.BatchNorm1d(out_channels * 4))
        self.pool = nn.MaxPool1d(2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)
        out = out + self.residual(x)
        out = self.relu(out)
        return self.pool(out)


class SelfAttention(nn.Module):
    """单头自注意力 Q/K/V，scores 除以 sqrt(d) 缩放"""

    def __init__(self, embed_size):
        super().__init__()
        self.scale = embed_size ** -0.5
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (batch, seq, embed)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale)
        return torch.bmm(scores, V)


class BiGRUInceptionAttentionBlock(nn.Module):
    """BiGRU -> Inception -> SelfAttention -> Dropout"""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.inception = InceptionModule(hidden_size * 2, hidden_size // 2)
        self.attention = SelfAttention((hidden_size // 2) * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)              # (B, L, hidden*2)
        out = out.permute(0, 2, 1)            # (B, hidden*2, L)
        out = self.inception(out)             # (B, hidden*2, L//2)
        out = out.permute(0, 2, 1)            # (B, L//2, hidden*2)
        out = self.attention(out)             # (B, L//2, hidden*2)
        return self.dropout(out)


class BiGRUInceptionAttentionModel(nn.Module):
    """N块堆叠 + 全局平均池化 + 全连接二分类"""

    def __init__(self, input_size, hidden_size, num_layers, output_size=2,
                 dropout=0.5, n_blocks=1):
        super().__init__()
        blocks = []
        ch_in = input_size
        for _ in range(n_blocks):
            blocks.append(BiGRUInceptionAttentionBlock(
                ch_in, hidden_size, num_layers, dropout))
            ch_in = (hidden_size // 2) * 4
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(ch_in, output_size)

    def forward(self, x):
        out = self.blocks(x)
        out = torch.mean(out, dim=1)
        return self.fc(out)


# =====================================================================
# 数据加载（新基础数据）
# =====================================================================
def load_aligned_data(args):
    feats = pd.read_csv(DATA_NEW / args.features_file,
                        index_col=0, parse_dates=True)
    lab = pd.read_csv(DATA_NEW / args.labels_file,
                      index_col=0, parse_dates=True)
    df = feats.join(lab, how="inner")
    df = df.loc[df["label"].notna() & df["sample_weight"].notna()]
    feats, y, w = (df.drop(columns=["label", "rate_delta_5d",
                                    "sample_weight"]),
                   df["label"].astype(int).values,
                   df["sample_weight"].astype(float).values)
    dates = feats.index

    if not args.full_features and not args.fold_importance_file:
        imp = pd.read_csv(DATA_NEW / "out" / args.importance_file)
        top_cols = imp.sort_values("gain_mean", ascending=False)[
            "feature"].tolist()
        top_cols = [c for c in top_cols if c in feats.columns][: args.top_k]
        feats = feats[top_cols]
        print(f"  特征筛选: 新XGB重要性Top{len(top_cols)}")

    n_dead = int((w < 1).sum())
    print(f"  样本: {len(feats)}, 特征: {feats.shape[1]}, "
          f"死区降权: {n_dead}/{len(feats)}")
    return feats, y, w, dates


def load_folds(args):
    fm = pd.read_csv(DATA_NEW / "out" / args.folds_file)
    fm["date_start"] = pd.to_datetime(fm["date_start"])
    fm["date_end"] = pd.to_datetime(fm["date_end"])
    return fm


# =====================================================================
# 滑动窗口序列与训练（与 01 脚本一致）
# =====================================================================
def gather_windows(X_t, indices, seq_len):
    rows = [list(range(i - seq_len + 1, i + 1)) for i in indices]
    idx_t = torch.tensor(rows, dtype=torch.long)
    return X_t[idx_t]


def seq_indices(y_len, end_lo, end_hi, seq_len):
    return [i for i in range(max(seq_len - 1, end_lo), end_hi + 1)
            if i < y_len]


def select_threshold(y_true, prob, mode, default_threshold):
    if mode == "fixed" or len(y_true) == 0 or np.unique(y_true).size < 2:
        return float(default_threshold)
    best_th = float(default_threshold)
    best_pair = (-np.inf, -np.inf)
    for th in np.round(np.arange(0.30, 0.701, 0.005), 3):
        pred = (prob >= th).astype(int)
        acc = accuracy_score(y_true, pred)
        f1m = f1_score(y_true, pred, average="macro", zero_division=0)
        pair = (f1m, acc) if mode == "val_macro_f1" else (acc, f1m)
        if pair > best_pair:
            best_pair = pair
            best_th = float(th)
    return best_th


def train_one_fold(args, X_t, y, w, tr_lo, tr_hi, device, input_size, seed):
    idx = seq_indices(len(y), tr_lo, tr_hi, args.seq_len)
    if len(idx) < 40:
        return None, {}
    gap = args.purge_gap if args.purge_gap >= 0 else args.seq_len
    n_val = max(8, int(len(idx) * args.val_rate))
    tr_idx, va_idx = idx[: -(n_val + gap)], idx[-n_val:]
    if len(tr_idx) < 20:
        return None, {}
    Xtr = gather_windows(X_t, tr_idx, args.seq_len)
    Xv = gather_windows(X_t, va_idx, args.seq_len)
    ytr_np = np.array([int(y[i]) for i in tr_idx])
    yva = torch.tensor([int(y[i]) for i in va_idx])
    wtr = torch.tensor([float(w[i]) for i in tr_idx], dtype=torch.float32)

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    model = BiGRUInceptionAttentionModel(
        input_size=input_size, hidden_size=args.hidden_size,
        num_layers=args.gru_layers, dropout=args.dropout,
        n_blocks=args.n_blocks).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr)

    if args.class_weight:
        cnt = np.bincount(ytr_np, minlength=2).astype(np.float64)
        cw = len(ytr_np) / (2.0 * np.maximum(cnt, 1.0))
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        cw_t = None
    ce = nn.CrossEntropyLoss(reduction="none", weight=cw_t,
                             label_smoothing=args.label_smoothing)
    ce_val = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    tr_loader = DataLoader(
        TensorDataset(Xtr, torch.tensor(ytr_np, dtype=torch.long), wtr),
        batch_size=args.batch_size, shuffle=True)

    Xv_d, yv_d = Xv.to(device), yva.to(device)
    yva_np = yva.numpy()
    best_score, best_state, bad, best_ep = -np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb in tr_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            if args.noise_std > 0:
                xb = xb + args.noise_std * torch.randn_like(xb)
            opt.zero_grad()
            loss = (ce(model(xb), yb) * wb).sum() / wb.sum()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xv_d)
            lv = ce_val(logits, yv_d).mean().item()
        if np.unique(yva_np).size > 1:
            pv = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            try:
                score = roc_auc_score(yva_np, pv)
            except ValueError:
                score = -lv
        else:
            score = -lv
        if score > best_score + 1e-4:
            best_score, best_state, bad, best_ep = score, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        elif ep >= args.min_epochs:
            bad += 1
            if bad >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.softmax(model(Xv_d), 1)[:, 1].cpu().numpy()
    return model, {"best_ep": best_ep, "val_score": best_score,
                   "n_train_seq": len(idx), "val_idx": list(va_idx),
                   "val_y": yva_np, "val_prob": val_prob}


@torch.no_grad()
def predict_proba(model, X_t, end_idx, args, device, bs=256):
    model.eval()
    arr = gather_windows(X_t, list(end_idx), args.seq_len)
    out = []
    for s in range(0, len(arr), bs):
        xb = arr[s: s + bs].to(device)
        p = torch.softmax(model(xb), dim=1)
        out.append(p[:, 1].cpu().tolist())
    return np.concatenate(out)


# =====================================================================
# 主流程
# =====================================================================
def main():
    args = parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print("=" * 56)
    print("BiGRU+Inception+Attention 新基础数据版（滚动fold评估）")
    print(f"device={device} | blocks={args.n_blocks} | hidden={args.hidden_size}"
          f" | seq={args.seq_len} | epochs<={args.epochs}")
    print(f"ensemble={max(1, args.n_seeds)}seeds | ls={args.label_smoothing}"
          f" | purge_gap={'seq_len' if args.purge_gap < 0 else args.purge_gap}"
          f" | cw={args.class_weight} | clip={args.grad_clip}")
    print("=" * 56)

    print("[1/4] 加载新基础数据...")
    feats, y, w, dates = load_aligned_data(args)
    folds = load_folds(args)

    date_arr = dates.values
    X_all = feats.values
    # 逐fold特征选择：每fold只用自身训练窗的Top-K（34脚本重要性表）
    fold_top: dict[int, list[str]] = {}
    if args.fold_importance_file:
        fi = pd.read_csv(DATA_NEW / "out" / args.fold_importance_file)
        for fd_id, g in fi.groupby("fold"):
            cols = [c for c in g.sort_values("gain", ascending=False)[
                "feature"] if c in feats.columns][: args.top_k]
            if len(cols) == args.top_k:
                fold_top[int(fd_id)] = cols
        print(f"  逐fold特征选择: {len(fold_top)} folds × Top{args.top_k}"
              f"（训练窗内重要性）")

    print(f"[2/4] 滚动训练: {len(folds)} folds ...")
    rows, preds = [], []
    for _, fd in folds.iterrows():
        s = int(np.searchsorted(date_arr, np.datetime64(fd["date_start"])))
        e = int(np.searchsorted(date_arr, np.datetime64(fd["date_end"]), "right")) - 1
        if s >= len(y) or e < s:
            continue
        tr_lo = 0 if args.max_train_days <= 0 else max(0, s - args.max_train_days)

        # 本fold特征子集（逐fold模式下每fold不同）
        X_raw = feats[fold_top[int(fd["fold"])]].values \
            if fold_top.get(int(fd["fold"])) else X_all

        # 标准化：只用训练段统计（防泄漏）；NaN填0(=均值)，极端值裁剪
        mu = np.nanmean(X_raw[tr_lo:s], axis=0)
        sd = np.nanstd(X_raw[tr_lo:s], axis=0) + 1e-8
        Xs = (X_raw - mu) / sd
        Xs = np.clip(np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0),
                     -8.0, 8.0).astype(np.float32)
        try:
            X_t = torch.from_numpy(Xs)
        except Exception:
            X_t = torch.tensor(Xs.tolist(), dtype=torch.float32)

        end_idx = np.arange(s, e + 1)
        probas, val_probas, best_eps, info = [], [], [], {}
        for k in range(max(1, args.n_seeds)):
            model, info = train_one_fold(
                args, X_t, y, w, tr_lo, s - 1, device,
                input_size=X_raw.shape[1], seed=args.seed + 1000 * k)
            if model is None:
                break
            probas.append(predict_proba(model, X_t, end_idx, args, device))
            val_probas.append(info.get("val_prob"))
            best_eps.append(info.get("best_ep") or 0)
        if not probas:
            print(f"  fold起点{s}训练样本不足，跳过")
            continue
        proba = np.mean(probas, axis=0)
        val_proba = np.mean(val_probas, axis=0)
        # 多seed平均后的验证段AUC（调参选参用，不看测试段）
        try:
            val_auc_fold = roc_auc_score(info.get("val_y"), val_proba)
        except Exception:
            val_auc_fold = np.nan
        fold_th = select_threshold(
            info.get("val_y"), val_proba, args.threshold_mode,
            args.decision_threshold)
        info = {"best_ep": int(np.mean(best_eps)),
                "n_train_seq": info.get("n_train_seq"),
                "threshold": fold_th}
        yp = (proba >= fold_th).astype(int)
        yt = y[end_idx]

        try:
            auc = roc_auc_score(yt, proba)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(yt, yp)
        f1m = f1_score(yt, yp, average="macro", zero_division=0)
        rows.append({"fold": fd["fold"], "date_start": str(fd["date_start"].date()),
                     "date_end": str(fd["date_end"].date()), "n_test": len(yt),
                     "accuracy": acc, "macro_f1": f1m, "auc": auc,
                     "val_auc": val_auc_fold,
                     "best_ep": info.get("best_ep"), "n_train_seq": info.get("n_train_seq"),
                     "threshold": fold_th})
        for i, pi, yi, di in zip(end_idx, proba, yt, date_arr[end_idx]):
            preds.append({"date": str(pd.Timestamp(di).date()), "row": int(i),
                          "y_true": int(yi), "proba_down": float(pi),
                          "pred": int(pi >= fold_th), "threshold": float(fold_th)})
        print(f"  fold {fd['fold']:>2} [{fd['date_start'].date()}~"
              f"{fd['date_end'].date()}] acc={acc:.4f} auc={auc:.4f} "
              f"ep={info.get('best_ep')} th={fold_th:.3f}", flush=True)

    prefix = f"{args.tag}_" if args.tag else ""
    fm = pd.DataFrame(rows)
    pr = pd.DataFrame(preds)
    fm.to_csv(OUT_DIR / f"{prefix}nnv2_fold_metrics.csv", index=False,
              encoding="utf-8-sig")
    pr.to_csv(OUT_DIR / f"{prefix}nnv2_test_predictions.csv", index=False,
              encoding="utf-8-sig")

    print("[3/4] 全局指标 ...")
    if len(pr):
        yt, yp, pb = pr["y_true"], pr["pred"], pr["proba_down"]
        g_acc = accuracy_score(yt, yp)
        g_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        g_auc = roc_auc_score(yt, pb)

        print("[4/4] 分年指标 ...")
        pr["year"] = pd.to_datetime(pr["date"]).dt.year
        year_stats = pr.groupby("year").apply(
            lambda g: pd.Series({
                "n": len(g),
                "acc": accuracy_score(g["y_true"], g["pred"]),
                "auc": roc_auc_score(g["y_true"], g["proba_down"])
                if g["y_true"].nunique() > 1 else np.nan,
            }), include_groups=False)

        lines = [
            "=" * 56,
            f"BiGRU+Inception+Attention 新基础数据 ({args.n_blocks}块, "
            f"hidden={args.hidden_size}, top-k="
            f"{'all' if args.full_features else args.top_k})",
            f"device={device} | ensemble={max(1, args.n_seeds)}seeds"
            f" | ls={args.label_smoothing} | purge=on | cw={args.class_weight}"
            f" | th={args.decision_threshold} | th_mode={args.threshold_mode}"
            f" | window={args.max_train_days}d",
            "-" * 56,
            f"全局: Acc={g_acc:.4f}  MacroF1={g_f1:.4f}  AUC={g_auc:.4f}",
            "新数据XGB基线: Acc=0.5236  MacroF1=0.5209  AUC=0.5277",
            "旧数据NN(BASE_ATTN): Acc=0.5385  MacroF1=0.5361  AUC=0.5745",
            "旧数据XGB基线: Acc=0.5635  MacroF1=0.5627  AUC=0.5822",
            "-" * 56,
            "分年:",
            year_stats.round(4).to_string(),
            "-" * 56,
            f"folds={len(fm)}  预测样本={len(pr)}  耗时={time.time()-t0:.0f}s",
            "=" * 56,
        ]
        report = "\n".join(lines)
        print(report)
        (OUT_DIR / f"{prefix}nnv2_metrics.txt").write_text(report, encoding="utf-8")
        print(f"\n输出: {OUT_DIR / f'{prefix}nnv2_metrics.txt'}")


if __name__ == "__main__":
    import traceback
    try:
        print("MAIN START", flush=True)
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
