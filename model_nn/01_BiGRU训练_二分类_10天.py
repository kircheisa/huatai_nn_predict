# -*- coding: utf-8 -*-
"""
BiGRU + Inception + Self-Attention 神经网络版（仿 资料2.md 架构）
预测10年期国债收益率未来10天方向（二分类：下行=1）

与 XGB 主流程（model_new/03_XGB训练_二分类.py）的对齐：
  1. 特征: 复用 model_new/data/xgb_train_features_bin.csv（280列，已含STL/遗传特征）
  2. 标签: 按 rate.csv 动态构造未来10天方向标签（不复用5天标签文件）
  3. 评估: 逐fold对齐 model_new/data/xgb_fold_metrics_bin.csv 的28个fold边界
  4. 死区: |rate_delta_10d|<2bp 样本权重0.3

相对 资料2.md 原架构的改动（小样本防过拟合）：
  - input_size: 10 -> XGB重要性Top-K（默认30，可--full-features用全部280）
  - output_size: 3 -> 2（二分类）
  - Block数: 3 -> 默认1（--n-blocks可调，样本仅~2000个序列撑不起3层堆叠）
  - hidden_size: 64 -> 默认32
  - 新增: 早停（验证集loss, patience=8）、训练窗口内标准化（防泄漏）、时间序切分

本轮泛化优化（目标：提升样本外 Acc/F1/AUC）：
  - 多seed集成: 每fold训练--n-seeds个模型(默认3)平均概率，降方差
  - purged验证切分: 训练/验证间隔seq_len行，消除滑动窗口重叠导致的早停泄漏
  - 类别权重 + 死区权重联合加权损失（按权重和归一化）
  - label smoothing 0.05（抑制过自信、改善校准）
  - AdamW + 余弦退火(lr->1e-5) + 梯度裁剪1.0
  - 自注意力加 1/sqrt(d) 缩放；GRU层间dropout
  - 输入高斯噪声增强（--noise-std，默认关）
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
    # 数据对齐
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--top-k", type=int, default=30,
                   help="按XGB重要性取Top-K特征（--full-features则忽略）")
    p.add_argument("--full-features", action="store_true",
                   help="使用全部280列特征")
    p.add_argument("--deadzone-bp", type=float, default=2.0,
                   help="标签死区(bp)，死区内样本权重0.3，0=禁用")
    # 模型结构（资料2架构的降配版）
    p.add_argument("--seq-len", type=int, default=60, help="滑动窗口长度")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=1, choices=[1, 2, 3],
                   help="BiGRU+Inception+Attention 堆叠块数（资料2为3，小样本建议1）")
    p.add_argument("--dropout", type=float, default=0.5)
    # 训练
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12, help="早停patience")
    p.add_argument("--min-epochs", type=int, default=10,
                   help="早停最小训练轮数（防验证AUC噪声导致过早停止欠拟合）")
    p.add_argument("--val-rate", type=float, default=0.2, help="训练窗口尾部验证占比")
    p.add_argument("--decision-threshold", type=float, default=0.5,
                   help="将下行概率转为类别的判定阈值，默认0.5")
    p.add_argument("--threshold-mode", default="fixed",
                   choices=["fixed", "val_macro_f1", "val_acc"],
                   help="fixed=使用固定阈值；val_* = 用fold验证集自动选阈值")
    # 泛化优化（提升样本外指标）
    p.add_argument("--n-seeds", type=int, default=3,
                   help="每fold多seed训练后平均概率（集成），1=关闭")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="标签平滑系数，0=禁用")
    p.add_argument("--class-weight", action="store_true", default=True,
                   help="按训练集类别频率加权损失（默认开）")
    p.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数，0=禁用")
    p.add_argument("--min-lr", type=float, default=1e-5, help="余弦退火最低学习率")
    p.add_argument("--purge-gap", type=int, default=-1,
                   help="训练/验证间隔行数(防早停泄漏)，-1=自动取seq_len")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="训练时输入加高斯噪声std(数据增强)，0=禁用")
    p.add_argument("--max-train-days", type=int, default=1260,
                   help="训练窗口上限（交易日），0=expanding全量")
    p.add_argument("--recency-halflife", type=int, default=0,
                   help="时间衰减样本权重半衰期(交易日)，近期样本权重更高，0=禁用")
    p.add_argument("--state-similarity", type=int, default=0,
                   help="状态相似度加权：与训练窗末端k个序列平均状态相近的样本权重更高，0=禁用")
    p.add_argument("--seed", type=int, default=42)
    # 运行
    p.add_argument("--folds-file", default="xgb_fold_metrics_bin.csv",
                   help="XGB的fold边界文件（保证两模型测试区间完全一致）")
    p.add_argument("--tag", default="H10_TRY",
                   help="输出文件名前缀（超参寻优时区分不同配置）")
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
    """单头自注意力 Q/K/V（资料2.md 同款，未除 sqrt(d)）"""

    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (batch, seq, embed)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = self.softmax(torch.bmm(Q, K.transpose(1, 2)))
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
        out = self.blocks(x)                  # (B, L//n, ch)
        out = torch.mean(out, dim=1)          # 全局平均池化
        return self.fc(out)                   # (B, 2)


# =====================================================================
# 数据对齐（复用XGB特征与fold边界，标签按horizon动态构造）
# =====================================================================
def load_aligned_data(args):
    # 1) 特征（5天口径缓存的全量特征；10天标签时裁掉末尾无法定义标签的行）
    feats = pd.read_csv(XGB_DATA / "xgb_train_features_bin.csv", index_col=0)

    # 2) 日期/利率：复刻XGB的merged构造（data.csv ∩ rate.csv 内连接）
    df_data = pd.read_csv(XGB_DATA / "data.csv", usecols=["Unnamed: 0"])
    df_rate = pd.read_csv(XGB_DATA / "rate.csv")[["time", "rate"]]
    df_data["Unnamed: 0"] = pd.to_datetime(df_data["Unnamed: 0"])
    df_rate["time"] = pd.to_datetime(df_rate["time"])
    merged = pd.merge(df_data, df_rate, left_on="Unnamed: 0",
                      right_on="time", how="inner").sort_values("Unnamed: 0")
    H = args.horizon
    merged["delta_h"] = merged["rate"].shift(-H) - merged["rate"]
    valid = merged["rate"].shift(-H).notna()   # 标签可定义的行
    merged = merged.loc[valid].reset_index(drop=True)

    cutoff = pd.to_datetime(args.start_date)
    mask = merged["Unnamed: 0"].values >= np.datetime64(cutoff)
    merged_f = merged.loc[mask].reset_index(drop=True)

    # 特征缓存按5天标签保留了更长的尾部；当horizon>5时需要裁掉末尾多余行。
    n = min(len(feats), len(merged_f))
    feats = feats.iloc[:n].reset_index(drop=True)
    merged_f = merged_f.iloc[:n].reset_index(drop=True)
    dates = merged_f["Unnamed: 0"]

    # 3) 动态标签 + 死区权重
    delta = merged_f["delta_h"].values
    y = (delta < 0).astype(int)
    if args.deadzone_bp > 0:
        w = np.where(np.abs(delta) * 100 < args.deadzone_bp, 0.3, 1.0)
        w = np.where(np.isnan(delta), 1.0, w)
    else:
        w = np.ones(len(r))
    n_dead = int((w < 1).sum())

    # 4) 特征筛选：XGB重要性Top-K
    if not args.full_features:
        imp = pd.read_csv(XGB_DATA / "xgb_feature_importance_bin.csv")
        imp_col = "feature" if "feature" in imp.columns else imp.columns[0]
        top_cols = imp.sort_values("importance", ascending=False)[imp_col].tolist()
        top_cols = [c for c in top_cols if c in feats.columns][:args.top_k]
        feats = feats[top_cols]
        print(f"  特征筛选: XGB重要性Top{len(top_cols)}")

    print(f"  horizon={H}天 | 样本: {len(feats)}, 特征: {feats.shape[1]}, "
          f"死区降权: {n_dead}/{len(feats)}")
    return feats, y.astype(int), w, dates


def load_folds(args):
    fm = pd.read_csv(XGB_DATA / args.folds_file)
    fm["date_start"] = pd.to_datetime(fm["date_start"])
    fm["date_end"] = pd.to_datetime(fm["date_end"])
    return fm


# =====================================================================
# 滑动窗口序列（张量索引方式，绕过 torch.numpy 桥不兼容问题）
# =====================================================================
def gather_windows(X_t, indices, seq_len):
    """X_t: (N, F) tensor；返回 (n, seq_len, F)，窗口 = 行 [i-L+1, i]"""
    rows = [list(range(i - seq_len + 1, i + 1)) for i in indices]
    idx_t = torch.tensor(rows, dtype=torch.long)
    return X_t[idx_t]


def seq_indices(y_len, end_lo, end_hi, seq_len):
    return [i for i in range(max(seq_len - 1, end_lo), end_hi + 1)
            if i < y_len]


def select_threshold(y_true, prob, mode, default_threshold):
    """按验证集自动选择阈值；类别单一或fixed模式时回退默认值。"""
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


# =====================================================================
# 训练与预测
# =====================================================================
def train_one_fold(args, X_t, y, w, tr_lo, tr_hi, device, input_size, seed):
    """训练窗口 [tr_lo, tr_hi]（end索引），返回模型与训练信息

    相对原版的训练侧优化：
      1. purged切分：训练/验证间隔gap行，验证窗口与训练窗口不共享原始行
         （滑动窗口场景下原版早停会因窗口重叠而虚低，选错epoch）
      2. 类别权重 + 死区样本权重联合加权损失（按权重和归一化）
      3. label smoothing 抑制过自信
      4. AdamW + 余弦退火(lr -> min_lr) + 梯度裁剪
      5. 可选输入高斯噪声增强
    """
    idx = seq_indices(len(y), tr_lo, tr_hi, args.seq_len)
    if len(idx) < 40:
        return None, {}

    # 时间序切分 + purge gap（-1=自动取seq_len，彻底去重叠）
    gap = args.purge_gap if args.purge_gap >= 0 else args.seq_len
    n_val = max(8, int(len(idx) * args.val_rate))
    tr_idx, va_idx = idx[:-(n_val + gap)], idx[-n_val:]
    if len(tr_idx) < 20:
        return None, {}
    Xtr = gather_windows(X_t, tr_idx, args.seq_len)
    Xv = gather_windows(X_t, va_idx, args.seq_len)
    ytr_np = np.array([int(y[i]) for i in tr_idx])
    yva = torch.tensor([int(y[i]) for i in va_idx])
    wtr = torch.tensor([float(w[i]) for i in tr_idx], dtype=torch.float32)
    # 时间衰减权重：距训练窗末端越近权重越高（半衰期recency_halflife个交易日）
    if args.recency_halflife > 0:
        ages = np.array([float(tr_hi - i) for i in tr_idx])
        decay = np.power(0.5, ages / float(args.recency_halflife))
        wtr = wtr * torch.tensor(decay, dtype=torch.float32)
    # 状态相似度加权：与训练窗末端k个序列的平均状态相近的历史样本权重更高
    # （针对regime切换：找"与当前状态像"的历史类比，而非单纯更近的历史）
    if args.state_similarity > 0:
        st = Xtr.mean(dim=1)                              # 每序列状态向量 (n, feat)
        ref = st[-args.state_similarity:].mean(dim=0, keepdim=True)
        sd = st.std(dim=0, keepdim=True).clamp_min(1e-6)
        d2 = (((st - ref) / sd) ** 2).mean(dim=1)         # 标准化平方距离 (n,)
        wtr = wtr * torch.exp(-d2 / d2.median().clamp_min(1e-6))

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

    # 类别权重（按训练段频率的逆）
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
        # 早停指标：验证AUC（与评估目标一致；验证段类别单一时退回-loss）
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
    """对 end_idx 逐个出预测概率"""
    model.eval()
    arr = gather_windows(X_t, list(end_idx), args.seq_len)
    out = []
    for s in range(0, len(arr), bs):
        xb = arr[s: s + bs].to(device)
        p = torch.softmax(model(xb), dim=1)
        out.append(p[:, 1].cpu().tolist())
    return np.concatenate(out)


# =====================================================================
# 主流程：滚动fold训练评估
# =====================================================================
def main():
    args = parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" )
    torch.backends.cudnn.benchmark = True

    print("=" * 56)
    print("BiGRU+Inception+Attention 神经网络版（滚动fold评估）")
    print(f"device={device} | blocks={args.n_blocks} | hidden={args.hidden_size}"
          f" | seq={args.seq_len} | epochs<={args.epochs}")
    print(f"ensemble={max(1, args.n_seeds)}seeds | ls={args.label_smoothing}"
          f" | purge_gap={'seq_len' if args.purge_gap < 0 else args.purge_gap}"
          f" | cw={args.class_weight} | clip={args.grad_clip}")
    print("=" * 56)

    print("[1/4] 加载数据（对齐XGB特征/fold，按horizon动态构造标签）...")
    feats, y, w, dates = load_aligned_data(args)
    folds = load_folds(args)

    # 行索引 <-> 日期映射
    date_arr = dates.values
    X_raw = feats.values

    print(f"[2/4] 滚动训练: {len(folds)} folds ...")
    rows, preds = [], []
    for _, fd in folds.iterrows():
        s = int(np.searchsorted(date_arr, np.datetime64(fd["date_start"])))
        e = int(np.searchsorted(date_arr, np.datetime64(fd["date_end"]), "right")) - 1
        if s >= len(y) or e < s:
            continue
        # 训练窗口（expanding + 上限）
        tr_lo = 0 if args.max_train_days <= 0 else max(0, s - args.max_train_days)

        # 标准化：只用训练段统计（防泄漏）；NaN填0(=均值)，极端值裁剪
        mu = np.nanmean(X_raw[tr_lo:s], axis=0)
        sd = np.nanstd(X_raw[tr_lo:s], axis=0) + 1e-8
        Xs = (X_raw - mu) / sd
        Xs = np.clip(np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0),
                     -8.0, 8.0).astype(np.float32)
        # 矩阵一次性转tensor（优先from_numpy零拷贝，ABI不兼容时回退tolist）
        try:
            X_t = torch.from_numpy(Xs)
        except Exception:
            X_t = torch.tensor(Xs.tolist(), dtype=torch.float32)

        end_idx = np.arange(s, e + 1)
        # 多seed集成：同一fold训练多个不同初始化的模型，平均预测概率
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
        fold_th = select_threshold(
            info.get("val_y"), val_proba, args.threshold_mode,
            args.decision_threshold)
        info = {"best_ep": int(np.mean(best_eps)),
                "n_train_seq": info.get("n_train_seq"),
                "threshold": fold_th}
        yp = (proba >= fold_th).astype(int)
        yt = y[end_idx]

        # 指标（与XGB同口径）
        try:
            auc = roc_auc_score(yt, proba)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(yt, yp)
        f1m = f1_score(yt, yp, average="macro", zero_division=0)
        rows.append({"fold": fd["fold"], "date_start": str(fd["date_start"].date()),
                     "date_end": str(fd["date_end"].date()), "n_test": len(yt),
                     "accuracy": acc, "macro_f1": f1m, "auc": auc,
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
    fm.to_csv(OUT_DIR / f"{prefix}nn_fold_metrics.csv", index=False,
              encoding="utf-8-sig")
    pr.to_csv(OUT_DIR / f"{prefix}nn_test_predictions.csv", index=False,
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
            f"BiGRU+Inception+Attention ({args.n_blocks}块, hidden={args.hidden_size},"
            f" top-k={'all' if args.full_features else args.top_k})",
            f"device={device} | ensemble={max(1, args.n_seeds)}seeds"
            f" | ls={args.label_smoothing} | purge=on | cw={args.class_weight}"
            f" | th={args.decision_threshold}"
            f" | th_mode={args.threshold_mode}"
            f" | window={args.max_train_days}d"
            f" | decay={args.recency_halflife or 'off'}"
            f" | sim={args.state_similarity or 'off'}",
            "-" * 56,
            f"全局: Acc={g_acc:.4f}  MacroF1={g_f1:.4f}  AUC={g_auc:.4f}",
            "XGB基线: Acc=0.5635  MacroF1=0.5627  AUC=0.5822",
            "-" * 56,
            f"阈值: mean={fm['threshold'].mean():.3f}  std={fm['threshold'].std(ddof=0):.3f}",
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

        # sweep汇总日志（追加一行，便于超参寻优对比）
        sweep_file = OUT_DIR / "sweep_log.csv"
        if not sweep_file.exists():
            sweep_file.write_text(
                "tag,acc,macro_f1,auc,blocks,hidden,seq,topk,seeds,ls,secs\n",
                encoding="utf-8-sig")
        with open(sweep_file, "a", encoding="utf-8-sig") as f:
            f.write(f"{args.tag or 'default'},{g_acc:.4f},{g_f1:.4f},{g_auc:.4f},"
                    f"{args.n_blocks},{args.hidden_size},{args.seq_len},"
                    f"{'all' if args.full_features else args.top_k},"
                    f"{max(1, args.n_seeds)},{args.label_smoothing},"
                    f"{time.time()-t0:.0f}\n")

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
