"""诊断：训练后分类器在训练集自身的准确率，判断是没学到还是评估口径错。"""
import numpy as np, torch
from collections import Counter
from data_utils import load_csv, build_encoder, transform, feature_dim
from model import Classifier
import train as T

Xtr_df, ytr_str = load_csv("KDDTrain+.txt")
Xte_df, yte_str = load_csv("train_test")
enc = build_encoder(Xtr_df)
Xtr = transform(Xtr_df, enc); Xte = transform(Xte_df, enc)
in_dim = feature_dim(enc)
known = sorted(set(ytr_str))
c2i = {c:i for i,c in enumerate(known)}
ytr = np.array([c2i[c] for c in ytr_str])

print("训练分类器...")
clf = T.train_classifier(Xtr, ytr, in_dim, len(known))
av, logits = T.get_activations(clf, Xtr)
pred = logits.argmax(1)
acc_tr = (pred == ytr).mean()
print(f"训练集自身 acc={acc_tr:.4f}")
# 每类准确率
for c in known:
    m = ytr == c2i[c]
    if m.sum()>0:
        a = (pred[m]==c2i[c]).mean()
        print(f"  {c:16s} n={m.sum():5d} acc={a:.3f}")

# 测试集已知类
yte_idx = np.array([c2i[c] if c in c2i else -1 for c in yte_str])
mask = yte_idx >= 0
av_t, logits_t = T.get_activations(clf, Xte)
pred_t = logits_t.argmax(1)
print(f"\n测试集已知类 acc={ (pred_t[mask]==yte_idx[mask]).mean():.4f} n={mask.sum()}")
