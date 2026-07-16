"""
基于嵌入空间的 OOD 检测：马氏距离 + MAV。

比 OpenMax(softmax 缩减)更可靠：过拟合分类器的 softmax 对 OOD 不敏感，
但 penultimate embedding 的几何结构对未知仍有区分力。

每类：均值 MAV + 类内协方差 Sigma_c。
样本到类 c 的马氏距离：d = sqrt((x-mav)^T Sigma_c^{-1} (x-mav))。
OOD 分数 = min_c d  （离最近已知类中心的马氏距离）。

协方差正则化：Sigma + eps*I，防小样本类奇异。
全局协方差作为 fallback（样本不足时）。
"""
import numpy as np


def fit_mahalanobis(av, labels, n_classes, reg=1e-2):
    """对每个类拟合 MAV + 协方差。返回 params。"""
    D = av.shape[1]
    # 全局协方差（用于小样本类 fallback / 共享）
    global_mean = av.mean(axis=0)
    global_cov = np.cov(av, rowvar=False) + reg * np.eye(D)
    global_inv = np.linalg.inv(global_cov)

    params = []
    for c in range(n_classes):
        mask = labels == c
        n = mask.sum()
        if n < 10:
            # 小样本类：用全局协方差，均值仍用类内
            mav = av[mask].mean(axis=0) if n > 0 else global_mean
            params.append({"mav": mav, "inv": global_inv, "n": int(n), "shared": True})
            continue
        cls_av = av[mask]
        mav = cls_av.mean(axis=0)
        cov = np.cov(cls_av, rowvar=False) + reg * np.eye(D)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = global_inv
        params.append({"mav": mav, "inv": inv, "n": int(n), "shared": False})
    return {"classes": params, "global_inv": global_inv, "global_mean": global_mean}


def mahalanobis_scores(av, fit):
    """返回 (N,) 最小马氏距离（到最近已知类中心）。"""
    N, D = av.shape
    classes = fit["classes"]
    # 预计算每个类的 (mav, inv)
    dists = np.zeros((N, len(classes)), dtype=np.float64)
    for ci, p in enumerate(classes):
        diff = av - p["mav"]                       # (N, D)
        # d_i = diff @ inv @ diff^T 对角
        tmp = diff @ p["inv"]                       # (N, D)
        d = np.einsum("ij,ij->i", tmp, diff)        # (N,)
        dists[:, ci] = np.sqrt(np.maximum(d, 0))
    return dists.min(axis=1)
