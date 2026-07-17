"""
类条件密度检测 (P0 核心)。

动机：guess_passwd/warezmaster/snmpguess 在窗口统计特征 f22~f40 上与 normal 完全可分
(test vs normal CV AUC≈0.97)，但当前全局 AE + softmax 抓不住——因为 one-hot service
(70维)主导相似度，count 维度的差异被淹没，且分类器把它们误分到 normal。

方案：在**原始窗口特征子空间**(不经过分类目标，分布干净)上，对每个已知类拟合稳健
马氏距离 (MCD)。推理时算样本到其**预测类**训练分布的马氏距离 —— 偏离则判 unknown。
guess_passwd 被分到 normal 时，其 count=1 偏离 normal 的 count=37 分布 → 距离大 → 检出。

不依赖分类器分对，这是与旧马氏(建在分类器嵌入上)的本质区别。
"""
import numpy as np

# 窗口/主机统计特征 (2s 窗口 + 目标主机历史) —— 弱时序，但对 R2L 区分力极强
WINDOW_FEATS = [f"f{i}" for i in range(22, 41)]  # f22..f40, 共19维


def _log1p_std(vals, means, stds):
    return ((np.log1p(vals) - means) / stds).astype(np.float32)


def fit_window_encoder(X_train_df):
    """在训练集上拟合窗口特征的 log1p+standardize 统计量。"""
    vals = np.log1p(X_train_df[WINDOW_FEATS].values.astype(np.float64))
    means = vals.mean(axis=0).astype(np.float64)
    stds = vals.std(axis=0).astype(np.float64)
    stds[stds < 1e-6] = 1.0
    return {"means": means, "stds": stds}


def transform_window(X_df, wenc):
    return _log1p_std(X_df[WINDOW_FEATS].values.astype(np.float64),
                      wenc["means"], wenc["stds"])


def fit_cond_mahalanobis(X_win, y, n_classes, reg=1e-2):
    """对每个已知类在窗口子空间拟合稳健马氏距离。

    X_win: (N, D) 已标准化的窗口特征
    y: (N,) 类别索引
    返回 fits: list of dict(mean, cov_inv, n, use_mcd)
    """
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
    D = X_win.shape[1]
    fits = []
    for c in range(n_classes):
        mask = y == c
        n = int(mask.sum())
        xc = X_win[mask]
        if n == 0:
            # 训练集无样本(不会发生，已知类都有)，用全局 fallback
            fits.append({"mean": np.zeros(D), "cov_inv": np.eye(D) / reg, "n": 0, "use_mcd": False})
            continue
        mean = xc.mean(axis=0)
        # MCD 需要样本数足够；小类用经验协方差 + L2 正则
        if n >= 2 * D:
            try:
                mcd = MinCovDet(support_fraction=None, random_state=42).fit(xc)
                cov = mcd.covariance_
            except Exception:
                cov = np.cov(xc, rowvar=False)
        else:
            cov = np.cov(xc, rowvar=False)
        # L2 正则保证可逆
        cov = cov + reg * np.eye(D)
        try:
            cov_inv = np.linalg.inv(cov)
        except Exception:
            cov_inv = np.linalg.pinv(cov)
        fits.append({"mean": mean, "cov_inv": cov_inv, "n": n, "use_mcd": n >= 2 * D})
    return fits


def _mahalanobis(X, mean, cov_inv):
    """批量马氏距离。X:(M,D) -> (M,)"""
    diff = X - mean
    # (M,D) @ (D,D) -> (M,D); 再 * diff 求和
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, cov_inv, diff), 0))


def cond_mahalanobis_scores(X_win, pred_cls, fits):
    """类条件马氏距离：每样本算到其【预测类】训练分布的距离。

    X_win: (M, D), pred_cls: (M,) 预测类索引
    返回 (M,) 距离，越大越像 unknown。
    """
    M = len(X_win)
    scores = np.zeros(M, dtype=np.float64)
    for c in range(len(fits)):
        mask = pred_cls == c
        if not mask.any():
            continue
        scores[mask] = _mahalanobis(X_win[mask], fits[c]["mean"], fits[c]["cov_inv"])
    return scores


def min_mahalanobis_scores(X_win, fits):
    """到最近已知类的马氏距离(备用信号)。"""
    all_d = np.stack([_mahalanobis(X_win, f["mean"], f["cov_inv"]) for f in fits], axis=1)
    return all_d.min(axis=1)
