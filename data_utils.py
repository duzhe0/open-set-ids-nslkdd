"""
NSL-KDD 数据加载与预处理。

特征：41维（1-3 符号列 duration/protocol_type/service/flag，4-41 数值）+ 标签（第42列）+ 难度（第43列）。
关键：train/test 分布漂移（service 等），需 one-hot 对齐到训练集词表，未见类别归入 "unseen" 桶。
"""
import numpy as np
import pandas as pd

COLS = (
    ["duration", "protocol_type", "service", "flag"]
    + [f"f{i}" for i in range(4, 41)]   # 4..40 -> 37 列数值
    + ["label", "difficulty"]
)
SYMBOLIC = ["protocol_type", "service", "flag"]
ALL_NUMERIC = ["duration"] + [f"f{i}" for i in range(4, 41)]
# f19 (num_outbound_cmds) 在训练集为常数，删除
DROP_NUMERIC = {"f19"}


def load_csv(path):
    df = pd.read_csv(path, header=None, names=COLS)
    labels = df["label"].astype(str).values
    X = df.drop(columns=["label", "difficulty"])
    return X, labels


def numeric_cols():
    return [c for c in ALL_NUMERIC if c not in DROP_NUMERIC]


def build_encoder(X_train_df):
    """在训练集上拟合 one-hot + log1p 标准化器，返回 encoder。

    数值处理：log1p 压长尾（src_bytes/dst_bytes 等可达 1e9），
    再做 mean/std 标准化。比直接 standardize 抗极值得多。
    """
    cats = {}
    for c in SYMBOLIC:
        cats[c] = sorted(X_train_df[c].astype(str).unique().tolist())
    num = numeric_cols()
    vals = np.log1p(X_train_df[num].values.astype(np.float64))
    means = vals.mean(axis=0).astype(np.float32)
    stds = vals.std(axis=0).astype(np.float32)
    stds[stds < 1e-6] = 1.0
    return {"cats": cats, "num": num, "means": means, "stds": stds}


def transform(X_df, enc):
    """将 df 转为数值矩阵。未见符号类别 -> 全零（OOD 信号之一）。"""
    parts = []
    for c in SYMBOLIC:
        vocab = enc["cats"][c]
        idx = {v: i for i, v in enumerate(vocab)}
        col = X_df[c].astype(str).values
        mat = np.zeros((len(X_df), len(vocab)), dtype=np.float32)
        for r, v in enumerate(col):
            j = idx.get(v)
            if j is not None:
                mat[r, j] = 1.0
        parts.append(mat)
    num = enc["num"]
    vals = np.log1p(X_df[num].values.astype(np.float64))
    vals = (vals - enc["means"]) / enc["stds"]
    parts.append(vals.astype(np.float32))
    return np.hstack(parts)


def feature_dim(enc):
    return sum(len(v) for v in enc["cats"].values()) + len(enc["num"])
