# -*- coding: utf-8 -*-
"""
BiGRU + Inception + Self-Attention 神经网络版（仿 资料2.md 架构）
预测10年期国债收益率未来10天方向（三分类：上行=0 / 下行=1 / 持平=2）

相对二分类版（01_BiGRU训练_二分类_10天.py）的改动：
  1. 标签三分: Δ<-deadzone → 1(下行); |Δ|≤deadzone → 2(持平); Δ>deadzone → 0(上行)
     —— 二分类的死区样本(|Δ|<2bp 权重0.3)在三分类中升级为独立类别
  2. 模型 output_size=3（资料2.md 架构原生支持，CrossEntropyLoss 多类）
  3. 早停指标: 验证集 OVR macro-AUC（类别不足时回退 -loss）
  4. 评估双口径:
       - 三分类: Acc / MacroF1 / OVR macro-AUC / 3x3混淆矩阵
       - 折叠二分类: 按方向类概率 p(下行)/(p(上行)+p(下行)) 折算，
         与二分类版/XGB基线完全可比
  5. 输出: 逐样本 p_up/p_down/p_flat/pred3/proba_down_folded

其余（特征Top-K、滚动fold、purged验证、多seed集成、label smoothing、
 AdamW+余弦退火、梯度裁剪、窗口标准化）与二分类版一致。
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
XGB_DATA = BASE / "model_new" / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 参数
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--full-features", action="store_true")
    p.add_argument("--deadzone-bp", type=float, default=2.0,
                   help="三分类死区阈值(bp)：|Δ|≤此值 → 持平类(2)")
    # 模型结构
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--dropout", type=float, default=0.5)
    # 训练
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--val-rate", type=float, default=0.2)
    # 泛化
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weight", action="store_true", default=True)
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--purge-gap", type=int, default=-1)
    p.add_argument("--max-train-days", type=int, default=1260)
    p.add_argument("--seed", type=int, default=42)
    # 运行
    p.add_argument("--folds-file", default="xgb_fold_metrics_bin.csv")
    p.add_argument("--tag", default="H10C3_TRY")
    return p.parse_args()


# =====================================================================
# 模型结构（照搬 资料2.md，超参外部传入）
# =====================================================================
class InceptionModule(nn.Module):
    """4分支多尺度1D卷积 + 1x1残差 + BN + MaxPool（资料2.md 同款）"""

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
    """单头自注意力 Q/K/V + 1/sqrt(d) 缩放"""

    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=2)
        self.scale = embed_size ** -0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale)
        return torch.bmm(scores, V)


class BiGRUInceptionAttentionBlock(nn.Module):
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
        out, _ = self.gru(x, h0)
        out = out.permute(0, 2, 1)
        out = self.inception(out)
        out = out.permute(0, 2, 1)
        out = self.attention(out)
        return self.dropout(out)


class BiGRUInceptionAttentionModel(nn.Module):
    """N块堆叠 + 全局平均池化 + 全连接（output_size=3 三分类）"""

    def __init__(self, input_size, hidden_size, num_layers, output_size=3,
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
        return self.fc(out)                   # (B, 3) logits


# =====================================================================
# 数据对齐（特征/fold与XGB一致，标签三分类动态构造）
# =====================================================================
def load_aligned_data(args):
    feats = pd.read_csv(XGB_DATA / "xgb_train_features_bin.csv", index_col=0)
    df_data = pd.read_csv(XGB_DATA / "data.csv", usecols=["Unnamed: 0"])
    df_rate = pd.read_csv(XGB_DATA / "rate.csv")[["time", "rate"]]
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0",
                      right_on="time", how="inner").sort_values("Unnamed: 0")
    H = args.horizon
    merged["delta_h"] = merged["rate"].shift(-H) - merged["rate"]
    merged = merged.loc[merged["rate"].shift(-H).notna()].reset_index(drop=True)

    cutoff = pd.to_datetime(args.start_date)
    merged_f = merged.loc[merged["Unnamed: 0"].values >= np.datetime64(cutoff)]
    merged_f = merged_f.reset_index(drop=True)

    n = min(len(feats), len(merged_f))
    feats = feats.iloc[:n].reset_index(drop=True)
    merged_f = merged_f.iloc[:n].reset_index(drop=True)
    dates = merged_f["Unnamed: 0"]

    # 三分类标签: 上行=0 / 下行=1 / 持平=2（|Δ|≤deadzone）
    delta = merged_f["delta_h"].values
    dz = args.deadzone_bp / 100.0
    y3 = np.where(delta < -dz, 1, np.where(delta > dz, 0, 2)).astype(int)
    y_bin = (delta < 0).astype(int)           # 折叠二分类真值（与二分类版同口径）

    if not args.full_features:
        imp = pd.read_csv(XGB_DATA / "xgb_feature_importance_bin.csv")
        imp_col = "feature" if "feature" in imp.columns else imp.columns[0]
        top_cols = imp.sort_values("importance", ascending=False)[imp_col].tolist()
        top_cols = [c for c in top_cols if c in feats.columns][:args.top_k]
        feats = feats[top_cols]

    dist = np.bincount(y3, minlength=3)
    print(f"  horizon={H}天 | 样本: {len(feats)}, 特征: {feats.shape[1]}")
    print(f"  三分类分布: 上行(0)={dist[0]} 下行(1)={dist[1]} 持平(2)={dist[2]}"
          f" ({dist[2]/len(y3):.1%})")
    return feats, y3, y_bin, dates


# =====================================================================
# 滑动窗口序列
# =====================================================================
def gather_windows(X_t, indices, seq_len):
    rows = [list(range(i - seq_len + 1, i + 1)) for i in indices]
    idx_t = torch.tensor(rows, dtype=torch.long)
    return X_t[idx_t]


def seq_indices(y_len, end_lo, end_hi, seq_len):
    return [i for i in range(max(seq_len - 1, end_lo), end_hi + 1)
            if i < y_len]


# =====================================================================
# 单fold训练（多类版）
# =====================================================================
def train_one_fold(args, X_t, y, tr_lo, tr_hi, device, input_size, seed):
    idx = seq_indices(len(y), tr_lo, tr_hi, args.seq_len)
    if len(idx) < 40:
        return None, {}

    gap = args.purge_gap if args.purge_gap >= 0 else args.seq_len
    n_val = max(8, int(len(idx) * args.val_rate))
    tr_idx, va_idx = idx[:-(n_val + gap)], idx[-n_val:]
    if len(tr_idx) < 20:
        return None, {}
    Xtr = gather_windows(X_t, tr_idx, args.seq_len)
    Xv = gather_windows(X_t, va_idx, args.seq_len)
    ytr_np = np.array([int(y[i]) for i in tr_idx])
    yva_np = np.array([int(y[i]) for i in va_idx])
    yva = torch.tensor(yva_np, dtype=torch.long)   # Windows numpy int32→必须显式long

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    model = BiGRUInceptionAttentionModel(
        input_size=input_size, hidden_size=args.hidden_size,
        num_layers=args.gru_layers, dropout=args.dropout,
        n_blocks=args.n_blocks, output_size=3).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr)

    # 三类逆频率权重
    if args.class_weight:
        cnt = np.bincount(ytr_np, minlength=3).astype(np.float64)
        cw = len(ytr_np) / (3.0 * np.maximum(cnt, 1.0))
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        cw_t = None
    ce = nn.CrossEntropyLoss(weight=cw_t,
                             label_smoothing=args.label_smoothing)
    ce_val = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    tr_loader = DataLoader(
        TensorDataset(Xtr, torch.tensor(ytr_np, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True)

    Xv_d, yv_d = Xv.to(device), yva.to(device)
    best_score, best_state, bad, best_ep = -np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xv_d)
            lv = ce_val(logits, yv_d).mean().item()
            pv = torch.softmax(logits, 1).cpu().numpy()
        # 早停: 验证OVR macro-AUC（验证段三类齐全才可算，否则-loss）
        if np.unique(yva_np).size == 3:
            try:
                score = roc_auc_score(yva_np, pv, multi_class="ovr",
                                      average="macro")
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
        val_prob = torch.softmax(model(Xv_d), 1).cpu().numpy()
    return model, {"best_ep": best_ep, "val_score": best_score,
                   "n_train_seq": len(idx), "val_y": yva_np,
                   "val_prob": val_prob}


@torch.no_grad()
def predict_proba3(model, X_t, end_idx, args, device, bs=256):
    """返回 (n,3) 概率矩阵"""
    model.eval()
    arr = gather_windows(X_t, list(end_idx), args.seq_len)
    out = []
    for s in range(0, len(arr), bs):
        xb = arr[s: s + bs].to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


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
    print("BiGRU+Inception+Attention 三分类版（滚动fold评估）")
    print(f"device={device} | blocks={args.n_blocks} | hidden={args.hidden_size}"
          f" | seq={args.seq_len} | epochs<={args.epochs} | seeds={args.n_seeds}")
    print("=" * 56)

    print("[1/4] 加载数据 ...")
    feats, y3, y_bin, dates = load_aligned_data(args)
    fm_xgb = pd.read_csv(XGB_DATA / args.folds_file)
    fm_xgb["date_start"] = pd.to_datetime(fm_xgb["date_start"])
    fm_xgb["date_end"] = pd.to_datetime(fm_xgb["date_end"])

    date_arr = dates.values
    X_raw = feats.values

    print(f"[2/4] 滚动训练: {len(fm_xgb)} folds ...")
    rows, preds = [], []
    for _, fd in fm_xgb.iterrows():
        s = int(np.searchsorted(date_arr, np.datetime64(fd["date_start"])))
        e = int(np.searchsorted(date_arr, np.datetime64(fd["date_end"]), "right")) - 1
        if s >= len(y3) or e < s:
            continue
        tr_lo = 0 if args.max_train_days <= 0 else max(0, s - args.max_train_days)

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
        probas, best_eps, info = [], [], {}
        for k in range(max(1, args.n_seeds)):
            model, info = train_one_fold(
                args, X_t, y3, tr_lo, s - 1, device,
                input_size=X_raw.shape[1], seed=args.seed + 1000 * k)
            if model is None:
                break
            probas.append(predict_proba3(model, X_t, end_idx, args, device))
            best_eps.append(info.get("best_ep") or 0)
        if not probas:
            print(f"  fold起点{s}训练样本不足，跳过")
            continue
        P = np.mean(probas, axis=0)                    # (n,3) 集成概率
        pred3 = P.argmax(axis=1)
        # 折叠二分类: 方向概率归一（剔除持平类质量）
        pd_fold = P[:, 1] / np.clip(P[:, 0] + P[:, 1], 1e-8, None)
        pred_dir = (pd_fold >= 0.5).astype(int)
        yt3, yt_bin = y3[end_idx], y_bin[end_idx]

        acc3 = accuracy_score(yt3, pred3)
        f13 = f1_score(yt3, pred3, average="macro", zero_division=0)
        try:
            auc3 = roc_auc_score(yt3, P, multi_class="ovr", average="macro")
        except ValueError:
            auc3 = np.nan
        acc_b = accuracy_score(yt_bin, pred_dir)
        try:
            auc_b = roc_auc_score(yt_bin, pd_fold)
        except ValueError:
            auc_b = np.nan
        rows.append({"fold": fd["fold"], "date_start": str(fd["date_start"].date()),
                     "date_end": str(fd["date_end"].date()), "n_test": len(yt3),
                     "acc3": acc3, "macro_f1_3": f13, "auc_ovr3": auc3,
                     "acc_folded2": acc_b, "auc_folded2": auc_b,
                     "best_ep": int(np.mean(best_eps))})
        for i, pi, pdir, pdw, y3i, ybi, di in zip(
                end_idx, pred3, pred_dir, pd_fold, yt3, yt_bin, date_arr[end_idx]):
            preds.append({"date": str(pd.Timestamp(di).date()), "row": int(i),
                          "y_true3": int(y3i), "y_true_bin": int(ybi),
                          "p_up": float(P[i - s, 0]), "p_down": float(P[i - s, 1]),
                          "p_flat": float(P[i - s, 2]), "pred3": int(pi),
                          "pred_dir": int(pdir),
                          "proba_down_folded": float(pdw)})
        print(f"  fold {fd['fold']:>2} [{fd['date_start'].date()}~"
              f"{fd['date_end'].date()}] 3cls: acc={acc3:.4f} f1={f13:.4f} "
              f"auc={auc3:.4f} | 折叠2cls: acc={acc_b:.4f} auc={auc_b:.4f} "
              f"ep={int(np.mean(best_eps))}", flush=True)

    prefix = f"{args.tag}_" if args.tag else ""
    fm = pd.DataFrame(rows)
    pr = pd.DataFrame(preds)
    fm.to_csv(OUT_DIR / f"{prefix}nn_fold_metrics.csv", index=False,
              encoding="utf-8-sig")
    pr.to_csv(OUT_DIR / f"{prefix}nn_test_predictions.csv", index=False,
              encoding="utf-8-sig")

    print("[3/4] 全局指标 ...")
    if len(pr):
        # 三分类口径
        g_acc3 = accuracy_score(pr["y_true3"], pr["pred3"])
        g_f13 = f1_score(pr["y_true3"], pr["pred3"], average="macro",
                         zero_division=0)
        g_auc3 = roc_auc_score(pr["y_true3"],
                               pr[["p_up", "p_down", "p_flat"]].values,
                               multi_class="ovr", average="macro")
        cm = pd.crosstab(pd.Series(pr["y_true3"], name="真实"),
                         pd.Series(pr["pred3"], name="预测"))
        # 折叠二分类口径（与二分类版/XGB可比）
        g_accb = accuracy_score(pr["y_true_bin"], pr["pred_dir"])
        g_aucb = roc_auc_score(pr["y_true_bin"], pr["proba_down_folded"])
        flat_hit = accuracy_score(pr.loc[pr["y_true3"] == 2, "y_true3"],
                                  pr.loc[pr["y_true3"] == 2, "pred3"])

        print("[4/4] 分年指标 ...")
        pr["year"] = pd.to_datetime(pr["date"]).dt.year
        year_stats = pr.groupby("year").apply(
            lambda g: pd.Series({
                "n": len(g),
                "acc3": accuracy_score(g["y_true3"], g["pred3"]),
                "auc_folded2": roc_auc_score(g["y_true_bin"],
                                             g["proba_down_folded"])
                if g["y_true_bin"].nunique() > 1 else np.nan,
                "acc_folded2": accuracy_score(g["y_true_bin"], g["pred_dir"]),
            }), include_groups=False)

        lines = [
            "=" * 56,
            f"三分类版 BiGRU+Inception+Attention (blocks={args.n_blocks},"
            f" hidden={args.hidden_size}, seq={args.seq_len},"
            f" horizon={args.horizon}天, seeds={max(1, args.n_seeds)})",
            f"deadzone={args.deadzone_bp}bp | ls={args.label_smoothing}"
            f" | cw={args.class_weight} | window={args.max_train_days}d",
            "-" * 56,
            f"[三分类] 全局: Acc={g_acc3:.4f}  MacroF1={g_f13:.4f}"
            f"  OVR-AUC={g_auc3:.4f}",
            f"[三分类] 持平类命中率: {flat_hit:.4f}",
            "混淆矩阵 (行=真实 0上行/1下行/2持平, 列=预测):",
            cm.to_string(),
            "-" * 56,
            f"[折叠二分类] Acc={g_accb:.4f}  AUC={g_aucb:.4f}",
            "参照: XGB基线 Acc=0.5635 AUC=0.5822 | 二分类FINAL5 Acc=0.5913 AUC=0.6705",
            "-" * 56,
            "分年:",
            year_stats.round(4).to_string(),
            "-" * 56,
            f"folds={len(fm)}  预测样本={len(pr)}  耗时={time.time()-t0:.0f}s",
            "=" * 56,
        ]
        report = "\n".join(lines)
        print(report)
        (OUT_DIR / f"{prefix}nn_metrics.txt").write_text(report, encoding="utf-8")
        print(f"\n输出: {OUT_DIR / f'{prefix}nn_metrics.txt'}")


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
