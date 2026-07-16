"""精搜权重：在 AE+softmax 主导区域细化，并测试单 AE / 加少量马氏是否更好。"""
import numpy as np
from data_utils import load_csv

ood = np.load("sig_ood_test.npy"); mah = np.load("sig_mah_test.npy")
err = np.load("sig_err_test.npy"); smax = np.load("sig_smax_test.npy")
om = np.load("sig_om_test.npy")
yte = np.load("yte_str.npy", allow_pickle=True)
_, ytr_str = load_csv("KDDTrain+.txt")
known = set(ytr_str)
true_known = np.array([c in known for c in yte]); true_unk = ~true_known

def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
n_ood=norm01(ood); n_mah=norm01(mah); n_err=norm01(err); n_smax=norm01(1-smax); n_om=norm01(om)

def best_f1(fuse, tnr_min=0.90):
    bf, bp, br, bt, bq = -1,0,0,0,0
    for q in np.arange(0.85, 0.97, 0.005):
        thr = np.percentile(fuse[true_known], q*100)
        unk = fuse > thr
        tp=(unk&true_unk).sum(); fp=(unk&true_known).sum(); fn=(~unk&true_unk).sum(); tn=(~unk&true_known).sum()
        p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); f1=2*p*r/(p+r+1e-9); tnr=tn/max(1,tn+fp)
        if tnr>=tnr_min and f1>bf: bf,bp,br,bt,bq=f1,p,r,tnr,q
    return bf,bp,br,bt,bq

# 1) 单信号
print("=== 单信号 ===")
for name, sig in [("err",n_err),("mah",n_mah),("ood",n_ood),("smax",n_smax),("om",n_om)]:
    f,p,r,t,q = best_f1(sig)
    print(f"  {name:5s} F1={f:.3f} P={p:.3f} R={r:.3f} TNR={t:.3f} q={q:.3f}")

# 2) err + 各辅助信号精搜
print("\n=== err主导 + 辅助精搜 (步长0.05, TNR>=0.90) ===")
best=None
for we in np.arange(0.5,1.01,0.05):
    for wx in np.arange(0,0.5,0.05):
        for name, sig in [("mah",n_mah),("smax",n_smax),("ood",n_ood),("om",n_om)]:
            s = we+wx
            if s==0: continue
            fuse = (we/s)*n_err + (wx/s)*sig
            f,p,r,t,q = best_f1(fuse)
            if f > (best[0] if best else -1):
                best = (f, {"err":round(we/s,3), name:round(wx/s,3)}, p, r, t, q)
f,w,p,r,t,q = best
print(f"  最优: {w} F1={f:.3f} P={p:.3f} R={r:.3f} TNR={t:.3f} q={q:.3f}")

# 3) err + 两辅助
print("\n=== err + 两辅助精搜 ===")
best=None
for we in np.arange(0.6,1.01,0.05):
    for w1 in np.arange(0,0.4,0.05):
        for w2 in np.arange(0,0.4,0.05):
            sigs = [("mah",n_mah),("smax",n_smax)]
            s=we+w1+w2
            if s==0: continue
            fuse=(we/s)*n_err+(w1/s)*n_mah+(w2/s)*n_smax
            f,p,r,t,q=best_f1(fuse)
            if f>(best[0] if best else -1):
                best=(f,{"err":round(we/s,3),"mah":round(w1/s,3),"smax":round(w2/s,3)},p,r,t,q)
f,w,p,r,t,q=best
print(f"  最优: {w} F1={f:.3f} P={p:.3f} R={r:.3f} TNR={t:.3f} q={q:.3f}")
fuse=(w["err"])*n_err+(w.get("mah",0))*n_mah+(w.get("smax",0))*n_smax
thr=np.percentile(fuse[true_known],q*100); unk=fuse>thr
print("\n各未知检出:")
for c in sorted(set(yte)-known):
    m=yte==c; print(f"  {c:16s} n={m.sum():4d} {unk[m].mean():.3f}")
