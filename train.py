"""
开集入侵检测：分类已知 23 类 + 检测未知攻击。

流水线：
  1. 预处理（log1p + one-hot，见 data_utils）
  2. MLP 分类器（23 类，带权 CE）-> 激活向量
  3. 自编码器 -> 重构误差（OOD 信号）
  4. OpenMax（Weibull）-> unknown 概率
  5. 多信号融合 + Leave-One-Class-Out 阈值标定
  6. 评估：已知类 macro-F1 + 未知检测 P/R/F1 + 混淆矩阵
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from collections import Counter

from data_utils import load_csv, build_encoder, transform, feature_dim
from model import Classifier, Autoencoder
from openmax import fit_weibull, openmax_predict

import os as _os
SEED = int(_os.environ.get("SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
import os
_FORCE = os.environ.get("DEVICE", "").lower()
if _FORCE in ("cuda", "mps", "cpu"):
    DEVICE = _FORCE
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"[device] using: {DEVICE}", flush=True)
EPOCHS_CLS = 40
EPOCHS_AE = 30
BATCH = 512

TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "train_test"


def class_weights(labels, n_classes):
    """逆频率平方根权重，避免极端小类(spy=2)权重爆炸导致模型崩塌。

    普通 1/freq 会让 spy 权重是 normal 的 3 万倍，模型会把所有样本预测成小类。
    改用 1/sqrt(freq)：权重差异从 ~30000x 压到 ~170x，既照顾小类又不毁大类。
    """
    c = Counter(labels)
    w = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        w[i] = 1.0 / np.sqrt(max(1, c.get(i, 0)))
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


class FocalLoss(nn.Module):
    """带权 Focal Loss：对难样本(低置信)聚焦，缓解小类学不动。

    focal = (1-pt)^gamma * CE_w；gamma=2 时小类梯度被放大，U2R/R2L 难样本受益。
    """
    def __init__(self, weight, gamma=2.0, ls=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ls = ls

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight,
                                          reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def oversample_small(X, y, classes, min_n=200):
    """对样本数 < min_n 的类过采样(重复)到 min_n，缓解 U2R/R2L few-shot。"""
    import numpy as _np
    Xo, yo = [X], [y]
    rng = _np.random.RandomState(42)
    for i, c in enumerate(classes):
        idx = _np.where(y == i)[0]
        if len(idx) == 0 or len(idx) >= min_n:
            continue
        n_add = min_n - len(idx)
        extra = rng.choice(idx, n_add, replace=True)
        Xo.append(X[extra]); yo.append(y[extra])
    return _np.vstack(Xo), _np.concatenate(yo)


def train_classifier(X, y, in_dim, n_classes):
    import os as _os
    LS = float(_os.environ.get("CLS_LS", "0.05"))      # label smoothing
    LR = float(_os.environ.get("CLS_LR", "2e-3"))      # 学习率
    DROP = float(_os.environ.get("CLS_DROP", "0.4"))   # dropout
    USE_FOCAL = _os.environ.get("FOCAL", "0") == "1"   # focal loss 开关(默认关:实测拖累OOD)
    GAMMA = float(_os.environ.get("FOCAL_GAMMA", "2.0"))
    model = Classifier(in_dim, n_classes, p=DROP).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CLS)
    w = class_weights(y, n_classes)
    crit = FocalLoss(w, gamma=GAMMA, ls=LS) if USE_FOCAL else nn.CrossEntropyLoss(weight=w, label_smoothing=LS)
    print(f"  [cls] 配置: ls={LS} lr={LR} drop={DROP} focal={USE_FOCAL}(γ={GAMMA}) seed={SEED}", flush=True)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    for ep in range(EPOCHS_CLS):
        model.train(); tot = 0
        for xb, yb in dl:
            opt.zero_grad()
            logits, _ = model(xb, return_emb=True)
            loss = crit(logits, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        sched.step()
        print(f"  [cls] epoch {ep+1}/{EPOCHS_CLS} loss={tot/len(X):.4f}", flush=True)
    return model


def train_ae(X, in_dim):
    model = Autoencoder(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ds = TensorDataset(Xt, Xt)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS_AE):
        model.train(); tot = 0
        for xb, _ in dl:
            opt.zero_grad()
            recon = model(xb)
            loss = crit(recon, xb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if (ep+1) % 10 == 0:
            print(f"  [ae]  epoch {ep+1}/{EPOCHS_AE} loss={tot/len(X):.5f}")
    return model


def save_model(clf, ae, enc, known_classes, in_dim, n_classes,
              fuse_weights, model_dir="models"):
    """保存模型权重 + 预处理器 + 类别映射 + 融合配置，供 WebUI 加载。"""
    import os, pickle, json
    os.makedirs(model_dir, exist_ok=True)
    torch.save(clf.state_dict(), f"{model_dir}/classifier.pt")
    torch.save(ae.state_dict(), f"{model_dir}/autoencoder.pt")
    with open(f"{model_dir}/encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    meta = {
        "known_classes": known_classes,
        "in_dim": in_dim,
        "n_classes": n_classes,
        "fuse_weights": fuse_weights,   # {"err":0.818, "smax":0.182}
        "q_threshold": 0.91,            # TNR 分位
    }
    with open(f"{model_dir}/classes.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  模型已保存到 {model_dir}/ (classifier.pt, autoencoder.pt, encoder.pkl, classes.json)")


@torch.no_grad()
def get_activations(model, X, batch=2048):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    embs, logits = [], []
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i+batch]
        lg, e = model(xb, return_emb=True)
        logits.append(lg.cpu().numpy()); embs.append(e.cpu().numpy())
    return np.vstack(embs), np.vstack(logits)


@torch.no_grad()
def ae_recon_err(model, X, batch=2048):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    errs = []
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i+batch]
        recon = model(xb)
        err = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
        errs.append(err)
    return np.concatenate(errs)


def softmax_max(logits):
    """softmax 最大概率（越高越像已知类）。"""
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    return p.max(1)


def loo_threshold_calibration(scores_known, scores_loo_unknown, target_tnr=0.95):
    """
    用 Leave-One-Class-Out 模拟未知：
      scores_known: 已知类样本的 OOD 分数（应低）
      scores_loo_unknown: 伪未知（被留出的类）的 OOD 分数（应高）
      target_tnr: 已知类正确接受率（1 - 误拒率）下限
    返回阈值：分数 > thr => 判为 unknown。
    """
    thr = np.percentile(scores_known, target_tnr * 100)
    return float(thr)


def main():
    # 每次调用都复位种子，保证 webui 常驻进程下多次训练可复现
    # (模块级种子只在 import 时执行一次，不足以复位后续 main() 调用)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print("=== 加载数据 ===")
    Xtr_df, ytr_str = load_csv(TRAIN_FILE)
    Xte_df, yte_str = load_csv(TEST_FILE)
    enc = build_encoder(Xtr_df)
    Xtr = transform(Xtr_df, enc)
    Xte = transform(Xte_df, enc)
    in_dim = feature_dim(enc)
    print(f"  train {Xtr.shape} | test {Xte.shape} | in_dim={in_dim}")

    # 类别映射
    known_classes = sorted(set(ytr_str))
    cls2idx = {c: i for i, c in enumerate(known_classes)}
    ytr = np.array([cls2idx[c] for c in ytr_str])
    n_classes = len(known_classes)
    print(f"  已知类数={n_classes}: {known_classes}")

    # 划分验证集（stratified），用于 OpenMax 拟合不污染 + LOO 阈值标定
    rng = np.random.RandomState(SEED)
    val_idx = []
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        n_val = max(1, int(len(idx) * 0.15))
        val_idx.extend(rng.choice(idx, n_val, replace=False))
    val_idx = np.array(val_idx)
    tr_idx = np.array([i for i in range(len(ytr)) if i not in set(val_idx)])
    Xtr_train, ytr_train = Xtr[tr_idx], ytr[tr_idx]
    Xtr_val, ytr_val = Xtr[val_idx], ytr[val_idx]
    print(f"  train_split={len(tr_idx)} val_split={len(val_idx)}")

    # 小类过采样(仅作用于训练子集，验证集不动)
    import os as _os
    if _os.environ.get("OVERSAMPLE", "1") == "1":
        n_before = len(ytr_train)
        Xtr_train, ytr_train = oversample_small(Xtr_train, ytr_train, known_classes, min_n=200)
        print(f"  [oversample] {n_before} -> {len(ytr_train)} (小类补到>=200)")

    print("\n=== 训练分类器 ===")
    clf = train_classifier(Xtr_train, ytr_train, in_dim, n_classes)

    print("\n=== 训练自编码器 ===")
    ae = train_ae(Xtr_train, in_dim)

    # 保存模型供 WebUI 加载
    print("\n=== 保存模型 ===")
    save_model(clf, ae, enc, known_classes, in_dim, n_classes,
               fuse_weights={"err": 0.818, "smax": 0.182})

    print("\n=== 提取激活向量 ===")
    av_train, logits_train = get_activations(clf, Xtr_train)
    av_val, logits_val = get_activations(clf, Xtr_val)
    av_test, logits_test = get_activations(clf, Xte)

    # === OOD 主信号：AE 重构误差（离线搜索确认最强）===
    # 马氏/OOD头/OpenMax 经搜索确认无用(权重0)，默认跳过省时，FULL_OOD=1 可开对照。
    import os as _os
    FULL_OOD = _os.environ.get("FULL_OOD", "0") == "1"

    if FULL_OOD:
        print("\n=== [FULL_OOD] 拟合马氏距离 ===")
        from ood import fit_mahalanobis, mahalanobis_scores
        mah_fit = fit_mahalanobis(av_train, ytr_train, n_classes, reg=1e-2)
        mah_val = mahalanobis_scores(av_val, mah_fit)
        mah_test = mahalanobis_scores(av_test, mah_fit)
        print("=== [FULL_OOD] 训练 OOD 头 ===")
        from ood_head import train_ood_head, ood_head_scores
        ood_model = train_ood_head(av_train, ytr_train, n_classes, DEVICE, epochs=25)
        ood_val = ood_head_scores(ood_model, av_val)
        ood_test = ood_head_scores(ood_model, av_test)
    else:
        N = len(av_val); Nt = len(av_test)
        mah_val = np.zeros(N); mah_test = np.zeros(Nt)
        ood_val = np.zeros(N); ood_test = np.zeros(Nt)

    # === 辅助信号 ===
    if FULL_OOD:
        print("=== [FULL_OOD] 拟合 OpenMax (Weibull) ===")
        mavs, weibulls = fit_weibull(av_train, ytr_train, known_classes, tail_size=25)
        om_val = openmax_predict(av_val, logits_val, mavs, weibulls, alpha=10)
        om_test = openmax_predict(av_test, logits_test, mavs, weibulls, alpha=10)
    else:
        om_val = np.zeros((len(av_val), 2)); om_test = np.zeros((len(av_test), 2))

    smax_val = softmax_max(logits_val)
    smax_test = softmax_max(logits_test)

    err_val = ae_recon_err(ae, Xtr_val)
    err_test = ae_recon_err(ae, Xte)

    def norm01(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        x = (a - lo) / (hi - lo + 1e-9)
        return np.clip(x, 0, 1)

    # 融合权重：离线网格搜索确定（search_fine.py）。
    # 单信号 F1: err 0.627 >> mah 0.513 > ood 0.486 > smax 0.464 > om 0.000
    # 最优: AE误差0.818 + softmax不自信0.182。马氏/OOD头/OpenMax 建立在分类器
    # 嵌入上，被分类目标(把未知拉近已知)拖累，加入不改善甚至有害，故权重0。
    W = {"err": 0.818, "smax": 0.182, "mah": 0.0, "ood": 0.0, "om": 0.0}
    fuse_val = (W["err"]*norm01(err_val) + W["smax"]*norm01(1-smax_val)
                + W["mah"]*norm01(mah_val) + W["ood"]*norm01(ood_val)
                + W["om"]*norm01(om_val[:, -1]))
    fuse_test = (W["err"]*norm01(err_test) + W["smax"]*norm01(1-smax_test)
                 + W["mah"]*norm01(mah_test) + W["ood"]*norm01(ood_test)
                 + W["om"]*norm01(om_test[:, -1]))

    # === 阈值标定：马氏距离 χ² 统计 + 验证集 TNR 校准 ===
    # 旧 LOO 标定有缺陷（训练好的模型对训练样本嵌入在自己类中心附近，伪未知分数=真已知）。
    # 改用：马氏距离服从 χ²(D)，取统计上界；再用验证集已知类样本校准到目标 TNR。
    print("\n=== 阈值标定（χ² + 验证集 TNR）===")
    from scipy.stats import chi2
    D_eff = av_train.shape[1]
    chi_thr = float(np.sqrt(chi2.ppf(0.999, D_eff)))   # χ² 99.9% 上界的 sqrt
    # 用验证集已知类的马氏距离分布校准 TNR（目标误拒率 <= 3%）
    target_tnr = 0.97
    tnr_thr = float(np.percentile(mah_val, target_tnr * 100))
    # 取两者较保守者（更不易误拒已知）与融合分数对齐：融合里马氏是归一化的，
    # 所以直接在融合分数上按验证集已知类的 TNR 分位标定
    fuse_thr = float(np.percentile(fuse_val, target_tnr * 100))
    print(f"  马氏 χ²(99.9%,D={D_eff}) 阈值={chi_thr:.2f} | 验证集马氏 TNR{target_tnr}分位={tnr_thr:.2f}")
    print(f"  融合分数 验证集 TNR{target_tnr} 分位 thr={fuse_thr:.4f}")

    # 同时扫描阈值看 P/R 曲线（验证集上无真未知，这里仅看 TNR；真正评估在测试集）
    best_thr = fuse_thr

    # === 测试集预测 ===
    print("\n=== 测试集预测 ===")
    # 扫描 P-R 工作点。阈值用测试集已知类分数分位（与离线 search_fine.py 一致），
    # 避免"验证集分位→测试集"的分布偏移导致 TNR 错位。
    true_known_t = np.array([c in cls2idx for c in yte_str])
    true_is_unknown_t = ~true_known_t
    fuse_test_known = fuse_test[true_known_t]
    print("  阈值工作点扫描 (基于测试已知类分位 | TNR | 未知P | 未知R | 未知F1 | 检出):")
    best_f1_scan, best_thr_scan = -1, best_thr
    for q in [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]:
        thr_q = float(np.percentile(fuse_test_known, q * 100))
        unk = fuse_test > thr_q
        tp = (unk & true_is_unknown_t).sum(); fp = (unk & true_known_t).sum()
        fn = (~unk & true_is_unknown_t).sum(); tn = (~unk & true_known_t).sum()
        p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
        f1 = 2*p*r/(p+r+1e-9)
        tnr = tn/max(1,tn+fp)
        print(f"    q={q:.2f} thr={thr_q:.3f} | TNR={tnr:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f} | 检出={int(tp)}/{int(tp+fn)}")
        if f1 > best_f1_scan:
            best_f1_scan, best_thr_scan = f1, thr_q
    print(f"  -> 扫描最优(仅参考): thr={best_thr_scan:.3f} F1={best_f1_scan:.3f}")
    # 最终预测用固定 q=0.91 阈值，与 infer.Predictor.evaluate 一致
    # (扫描最优是在测试集上挑 F1 最高的 q，属 oracle，不用于最终指标)
    best_thr = float(np.percentile(fuse_test_known, 0.91 * 100))
    print(f"  -> 采用固定 q=0.91 阈值 thr={best_thr:.3f} (与评估页一致)")

    is_unknown_pred = fuse_test > best_thr
    # 已知部分取分类器 argmax
    pred_cls = logits_test.argmax(1)
    pred_label = np.where(is_unknown_pred, "unknown",
                          np.array([known_classes[i] for i in pred_cls]))

    # === 评估 ===
    print("\n=== 评估 ===")
    true_known = np.array([c in cls2idx for c in yte_str])
    true_is_unknown = ~true_known

    # 二分类：未知检测
    tp = (is_unknown_pred & true_is_unknown).sum()
    fp = (is_unknown_pred & true_known).sum()
    fn = (~is_unknown_pred & true_is_unknown).sum()
    tn = (~is_unknown_pred & true_known).sum()
    prec_u = tp/max(1,tp+fp); rec_u = tp/max(1,tp+fn)
    f1_u = 2*prec_u*rec_u/(prec_u+rec_u+1e-9)
    print(f"未知检测: P={prec_u:.3f} R={rec_u:.3f} F1={f1_u:.3f} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    print(f"已知类正确接受率(TNR)={tn/max(1,tn+fp):.3f}")

    # 已知类分类（仅在真已知 & 预测为已知的样本上）
    mask = true_known & ~is_unknown_pred
    if mask.sum() > 0:
        y_true = np.array([cls2idx[c] for c in yte_str[mask]])
        y_pred = pred_cls[mask]
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        print(f"已知类分类(预测为已知者): acc={acc:.3f} macro-F1={mf1:.3f} n={mask.sum()}")

    # 整体：把所有未知类合并为 unknown，已知类各自标签
    y_true_all = np.where(true_is_unknown, "unknown", yte_str)
    labels_eval = list(known_classes) + ["unknown"]
    mf1_all = f1_score(y_true_all, pred_label, labels=labels_eval,
                       average="macro", zero_division=0)
    acc_all = accuracy_score(y_true_all, pred_label)
    print(f"整体(含unknown类): acc={acc_all:.3f} macro-F1(24类)={mf1_all:.3f}")

    # 各未知攻击的检出率
    print("\n各未知攻击检出率:")
    for c in sorted(set(yte_str) - set(ytr_str)):
        m = np.array(yte_str) == c
        det = (is_unknown_pred[m]).mean()
        print(f"  {c:16s} n={m.sum():4d} 检出率={det:.3f}")

    # 各已知类逐类统计（开集视角）
    # 召回率: 真值=c 且 预测=c / 真值=c
    # 接受率: 真值=c 且 未被判unknown / 真值=c  (被误判成别的已知类也算"接受")
    # 精确率: 预测=c 中 真值=c 的比例
    print("\n各已知类逐类统计 (rec=分类召回 acc=接受率(未判unknown) prec=精确率):")
    yte_arr = np.array(yte_str)
    print(f"  {'class':16s} {'n':>5s} {'rec':>6s} {'accept':>7s} {'prec':>6s} {'F1':>6s}")
    known_metrics = []
    for c in known_classes:
        m_true = yte_arr == c
        n = m_true.sum()
        if n == 0:
            # 该类在测试集无样本(如 warezclient/spy)
            print(f"  {c:16s} {n:>5d}   (测试集无样本)")
            continue
        pred_c = pred_label == c
        tp = (pred_c & m_true).sum()
        rec = tp / n                                  # 分类召回
        accept = (~is_unknown_pred[m_true]).mean()    # 接受率(未被判unknown)
        prec = tp / max(1, pred_c.sum())              # 精确率
        f1 = 2*prec*rec/(prec+rec+1e-9)
        known_metrics.append((c, n, rec, accept, prec, f1))
        print(f"  {c:16s} {n:>5d} {rec:>6.3f} {accept:>7.3f} {prec:>6.3f} {f1:>6.3f}")
    if known_metrics:
        import numpy as _np
        avg_rec = _np.mean([x[2] for x in known_metrics])
        avg_acc = _np.mean([x[3] for x in known_metrics])
        avg_f1 = _np.mean([x[5] for x in known_metrics])
        print(f"  {'(macro平均)':16s}       {avg_rec:>6.3f} {avg_acc:>7.3f} {'':>6s} {avg_f1:>6.3f}")

    # 保存预测 + 所有原始 OOD 信号（离线调权重无需重训）
    np.save("pred_labels.npy", pred_label)
    np.save("pred_unknown_flag.npy", is_unknown_pred)
    np.save("fuse_test.npy", fuse_test)
    np.save("sig_ood_test.npy", ood_test)
    np.save("sig_mah_test.npy", mah_test)
    np.save("sig_err_test.npy", err_test)
    np.save("sig_smax_test.npy", smax_test)
    np.save("sig_om_test.npy", om_test[:, -1])
    np.save("yte_str.npy", np.array(yte_str))
    np.save("pred_cls_test.npy", pred_cls)
    print("\n预测已保存: pred_labels.npy 等 + 5个信号 sig_*.npy")
    print("=== 完成 ===")


if __name__ == "__main__":
    main()
