"""
模型加载与推理：供 WebUI 调用。

加载保存的 classifier.pt / autoencoder.pt / encoder.pkl / classes.json，
提供：
  - predict(X_raw_df): 单条/批量预测，返回预测标签 + OOD 分数 + 各信号
  - evaluate(test_file): 跑测试集，返回完整评估指标
"""
import os, sys, pickle, json
# 把项目根目录加入 path，以便 import 上层的 data_utils/model/train
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from data_utils import load_csv, transform, feature_dim
from model import Classifier, Autoencoder
import train as T


class Predictor:
    def __init__(self, model_dir=None, device=None):
        # 默认指向项目根目录的 models/
        if model_dir is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(root, "models")
        self.device = device or T.DEVICE
        with open(f"{model_dir}/classes.json") as f:
            meta = json.load(f)
        self.known_classes = meta["known_classes"]
        self.cls2idx = {c: i for i, c in enumerate(self.known_classes)}
        self.in_dim = meta["in_dim"]
        self.n_classes = meta["n_classes"]
        self.fuse_weights = meta["fuse_weights"]
        self.q_threshold = meta["q_threshold"]
        with open(f"{model_dir}/encoder.pkl", "rb") as f:
            self.enc = pickle.load(f)

        # 加载模型权重
        self.clf = Classifier(self.in_dim, self.n_classes).to(self.device)
        self.clf.load_state_dict(torch.load(f"{model_dir}/classifier.pt", map_location=self.device))
        self.clf.eval()
        self.ae = Autoencoder(self.in_dim).to(self.device)
        self.ae.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location=self.device))
        self.ae.eval()

    def _scores(self, X):
        """对已 transform 的矩阵 X 算各 OOD 信号 + 融合分数。"""
        av, logits = T.get_activations(self.clf, X)
        err = T.ae_recon_err(self.ae, X)
        smax = T.softmax_max(logits)
        # 归一化
        n_err = self._norm01(err)
        n_smax = self._norm01(1 - smax)
        w = self.fuse_weights
        fuse = w["err"] * n_err + w["smax"] * n_smax
        return logits, err, smax, fuse

    @staticmethod
    def _norm01(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    def predict_df(self, X_df):
        """输入原始 DataFrame，返回每条预测结果。"""
        X = transform(X_df, self.enc)
        logits, err, smax, fuse = self._scores(X)
        # 阈值用 fuse 的固定分位（与训练一致，用已知类 q=0.91）
        # 但 predict 时不知哪些是已知，用训练时保存的阈值近似
        # 这里用一个合理默认：fuse 的 0.91 分位（需已知类，预测时用全体近似）
        thr = float(np.percentile(fuse, self.q_threshold * 100))
        is_unknown = fuse > thr
        pred_cls = logits.argmax(1)
        pred_label = np.where(is_unknown, "unknown",
                              np.array([self.known_classes[i] for i in pred_cls]))
        return {
            "pred_label": pred_label,
            "is_unknown": is_unknown,
            "fuse_score": fuse,
            "ae_err": err,
            "softmax_max": smax,
            "pred_cls_idx": pred_cls,
            "logits": logits,
            "threshold": thr,
        }

    def evaluate(self, test_file):
        """跑测试集，返回完整评估指标（复用 train.py 评估逻辑）。

        评估口径与 train.py 一致：阈值用测试集已知类的 q=0.91 分位
        （评估时有真实标签，可用已知类分布定阈值，等价线上用已知流量比例校准）。
        """
        Xte_df, yte_str = load_csv(test_file)
        X = transform(Xte_df, self.enc)
        logits, err, smax, fuse = self._scores(X)
        # 评估时用测试集已知类分位定阈值（与 train.py 一致）
        true_known = np.array([c in self.cls2idx for c in yte_str])
        thr = float(np.percentile(fuse[true_known], self.q_threshold * 100))
        is_unknown = fuse > thr
        pred_cls = logits.argmax(1)
        pred_label = np.where(is_unknown, "unknown",
                              np.array([self.known_classes[i] for i in pred_cls]))
        res = {"is_unknown": is_unknown, "pred_cls_idx": pred_cls, "pred_label": pred_label,
               "fuse_score": fuse, "threshold": thr}
        true_unk = ~true_known

        # 未知检测
        tp = int((is_unknown & true_unk).sum())
        fp = int((is_unknown & true_known).sum())
        fn = int((~is_unknown & true_unk).sum())
        tn = int((~is_unknown & true_known).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 2 * p * r / (p + r + 1e-9)
        tnr = tn / max(1, tn + fp)

        # 已知类分类
        from sklearn.metrics import accuracy_score, f1_score
        mask = true_known & ~is_unknown
        known_acc = accuracy_score(
            [self.cls2idx[c] for c in np.array(yte_str)[mask]], pred_cls[mask]
        ) if mask.sum() > 0 else 0.0
        known_mf1 = f1_score(
            [self.cls2idx[c] for c in np.array(yte_str)[mask]], pred_cls[mask], average="macro"
        ) if mask.sum() > 0 else 0.0

        # 各未知攻击检出率
        per_unknown = []
        for c in sorted(set(yte_str) - set(self.known_classes)):
            m = np.array(yte_str) == c
            per_unknown.append({"attack": c, "n": int(m.sum()),
                                "detect_rate": float(is_unknown[m].mean())})

        # 已知类逐类
        per_known = []
        yte_arr = np.array(yte_str)
        for c in self.known_classes:
            m = yte_arr == c
            n = int(m.sum())
            if n == 0:
                per_known.append({"attack": c, "n": 0, "rec": 0, "accept": 0, "prec": 0, "f1": 0})
                continue
            pred_c = pred_label == c
            tp_c = int((pred_c & m).sum())
            rec = tp_c / n
            accept = float((~is_unknown[m]).mean())
            prec = tp_c / max(1, int(pred_c.sum()))
            f1c = 2 * prec * rec / (prec + rec + 1e-9)
            per_known.append({"attack": c, "n": n, "rec": round(rec, 3),
                              "accept": round(accept, 3), "prec": round(prec, 3),
                              "f1": round(f1c, 3)})

        # 混淆矩阵（已知类间，未知合并）
        y_true_all = np.where(true_unk, "unknown", yte_str)
        labels = list(self.known_classes) + ["unknown"]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_all, pred_label, labels=labels)
        overall_acc = accuracy_score(y_true_all, pred_label)
        overall_mf1 = f1_score(y_true_all, pred_label, labels=labels, average="macro", zero_division=0)

        return {
            "unknown_detection": {"P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
                                   "TP": tp, "FP": fp, "FN": fn, "TN": tn, "TNR": round(tnr, 3)},
            "known_classification": {"acc": round(known_acc, 3), "macro_f1": round(known_mf1, 3),
                                      "n": int(mask.sum())},
            "overall": {"acc": round(overall_acc, 3), "macro_f1": round(overall_mf1, 3)},
            "per_unknown": per_unknown,
            "per_known": per_known,
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "n_test": len(yte_str),
        }


if __name__ == "__main__":
    p = Predictor()
    print("模型加载成功,已知类:", p.known_classes[:5], "...")
    res = p.evaluate("train_test")
    print("未知检测:", res["unknown_detection"])
    print("已知分类:", res["known_classification"])
    print("整体:", res["overall"])
