"""诊断:guess_passwd/warezmaster 被分到哪了,以及和谁的激活向量最近。"""
import numpy as np
from data_utils import load_csv, build_encoder, transform, feature_dim

Xtr_df, ytr_str = load_csv("KDDTrain+.txt")
Xte_df, yte_str = load_csv("train_test")
enc = build_encoder(Xtr_df)
Xtr = transform(Xtr_df, enc); Xte = transform(Xte_df, enc)
ytr = np.array(ytr_str); yte = np.array(yte_str)
known = sorted(set(ytr_str)); c2i = {c:i for i,c in enumerate(known)}

# 原始特征空间:guess_passwd测试样本 到各已知类中心的距离
for target in ["guess_passwd", "warezmaster"]:
    te_mask = yte == target
    if te_mask.sum()==0: continue
    te_mean = Xte[te_mask].mean(0)
    print(f"\n=== {target} (test n={te_mask.sum()}, train n={(ytr==target).sum()}) ===")
    dists = []
    for kc in known:
        kc_mean = Xtr[ytr==kc].mean(0)
        d = np.linalg.norm(te_mean - kc_mean)
        dists.append((kc, d))
    dists.sort(key=lambda x: x[1])
    print("  到各已知类中心距离(前6):")
    for kc, d in dists[:6]:
        print(f"    {kc:16s} d={d:.2f}")
    # 训练集 guess_passwd 自身分布 vs 测试集
    if (ytr==target).sum()>0:
        tr_mean = Xtr[ytr==target].mean(0)
        print(f"  train {target}中心 到 test {target}中心 距离={np.linalg.norm(tr_mean-te_mean):.2f}")
        print(f"  (小说明: 距离大=训练/测试该类分布漂移严重)")
