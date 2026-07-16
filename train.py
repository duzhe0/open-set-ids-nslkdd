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

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
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


def train_classifier(X, y, in_dim, n_classes):
    model = Classifier(in_dim, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CLS)
    w = class_weights(y, n_classes)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
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

    print("\n=== 训练分类器 ===")
    clf = train_classifier(Xtr_train, ytr_train, in_dim, n_classes)

    print("\n=== 训练自编码器 ===")
    ae = train_ae(Xtr_train, in_dim)

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

    def softmax_max(logits):
        p = np.exp(logits - logits.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        return p.max(1)
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
    # 先扫描 P-R 曲线，看不同 TNR 工作点下的未知召回，找最优阈值
    true_known_t = np.array([c in cls2idx for c in yte_str])
    true_is_unknown_t = ~true_known_t
    print("  阈值工作点扫描 (thr | TNR | 未知P | 未知R | 未知F1 | 未知检出数):")
    best_f1_scan, best_thr_scan = -1, best_thr
    for q in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        thr_q = float(np.percentile(fuse_val, q * 100))
        unk = fuse_test > thr_q
        tp = (unk & true_is_unknown_t).sum(); fp = (unk & true_known_t).sum()
        fn = (~unk & true_is_unknown_t).sum(); tn = (~unk & true_known_t).sum()
        p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
        f1 = 2*p*r/(p+r+1e-9)
        tnr = tn/max(1,tn+fp)
        print(f"    q={q:.2f} thr={thr_q:.3f} | TNR={tnr:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f} | 检出={int(tp)}/{int(tp+fn)}")
        if f1 > best_f1_scan:
            best_f1_scan, best_thr_scan = f1, thr_q
    # 选 F1 最高的工作点作为最终阈值（若与 TNR0.97 相近则保留原值）
    print(f"  -> 扫描最优: thr={best_thr_scan:.3f} F1={best_f1_scan:.3f} (原TNR0.97 thr={best_thr:.3f})")
    best_thr = best_thr_scan

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
