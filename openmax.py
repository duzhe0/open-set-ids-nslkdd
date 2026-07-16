"""
OpenMax 开集识别。

思路（Bendale & Boult 2015）：
1) 取分类器倒数第二层激活向量 AV（activation vector）。
2) 对每个已知类，计算该类训练样本到其 MAV(mean activation vector)的距离，
   取最远的 tail_size 个距离，用 Weibull 分布拟合 -> 描述该类"正常范围"。
3) 推理时：对样本的 softmax 概率，按各类的 Weibull CDF 做"缩减"，
   缩减出来的概率质量汇入新增的 unknown 类，得到 (n_classes+1) 的分布。

只用 numpy/scipy，无需额外库。
"""
import numpy as np
from scipy.stats import weibull_min


def fit_weibull(av, labels, classes, tail_size=20):
    """
    av: (N, D) 激活向量
    labels: (N,) 类别索引
    返回 mavs: (C, D), weibulls: list of (c, loc, shape) 实际存 params
    """
    mavs = np.zeros((len(classes), av.shape[1]), dtype=np.float32)
    weibulls = []
    for i, c in enumerate(classes):
        mask = labels == c
        if mask.sum() < 2:
            # 小样本类：无法拟合 Weibull，填全键，标记为不可用
            weibulls.append({
                "c": c, "idx": i, "mav": np.zeros(av.shape[1], dtype=np.float32),
                "shape": None, "loc": 0.0, "scale": 1.0, "params": None,
            })
            continue
        cls_av = av[mask]
        mav = cls_av.mean(axis=0)
        mavs[i] = mav
        # 每个样本到 MAV 的距离
        dists = np.linalg.norm(cls_av - mav, axis=1)
        dists = np.sort(dists)[::-1]  # 从大到小
        tail = dists[:max(1, min(tail_size, len(dists)))]
        # Weibull 拟合: 对距离拟合
        try:
            c_shape, loc, scale = weibull_min.fit(tail, floc=0.0)
        except Exception:
            c_shape, loc, scale = 1.0, 0.0, float(np.max(tail) + 1e-6)
        weibulls.append({
            "c": c, "idx": i,
            "mav": mav,
            "shape": float(c_shape),
            "loc": float(loc),
            "scale": float(scale),
        })
    return mavs, weibulls


def _weibull_cdf(wb, dist):
    if wb["params"] is None and wb.get("shape") is None:
        return 0.0
    shape = wb["shape"]; loc = wb["loc"]; scale = wb["scale"]
    if scale <= 0:
        scale = 1e-6
    return float(weibull_min.cdf(dist, shape, loc=loc, scale=scale))


def openmax_predict(av_test, logits_test, mavs, weibulls, alpha=10, threshold=None):
    """
    av_test: (M, D)
    logits_test: (M, C)
    返回 openmax_scores: (M, C+1), 最后一列=unknown
    """
    M, C = logits_test.shape
    # 先 softmax
    probs = np.exp(logits_test - logits_test.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    om_scores = np.zeros((M, C + 1), dtype=np.float32)
    for r in range(M):
        av = av_test[r]
        # 每个 class 的修正权重 w
        rev_scores = probs[r].copy()
        ws = np.zeros(C, dtype=np.float32)
        # 按 logits 排序取 top-alpha 个类做缩减
        ranked = np.argsort(-probs[r])
        for rank, ci in enumerate(ranked):
            if rank >= alpha:
                break
            wb = None
            for w in weibulls:
                if w["idx"] == ci and w.get("shape") is not None:
                    wb = w
                    break
            if wb is None or wb.get("mav") is None:
                continue
            dist = np.linalg.norm(av - wb["mav"])
            w_cdf = _weibull_cdf(wb, dist)
            ws[ci] = w_cdf
        # 修正
        revised = rev_scores * (1 - ws)
        unknown_mass = (rev_scores * ws).sum()
        om_scores[r, :C] = revised
        om_scores[r, C] = unknown_mass
    # 归一化
    s = om_scores.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    om_scores = om_scores / s
    return om_scores
