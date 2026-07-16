"""
深度模型：
1) Classifier: MLP，输出 23 类 logits + 激活向量(penultimate)。
2) Autoencoder: 重构输入，重构误差作为 OOD 信号之一。

类别不平衡 -> 带权交叉熵。
"""
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden // 2, n_classes)
        self.emb_dim = hidden // 2

    def forward(self, x, return_emb=False):
        e = self.net(x)
        logits = self.fc(e)
        if return_emb:
            return logits, e
        return logits


class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden=128, code=32, p=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, code), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(code, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

    def encode(self, x):
        return self.enc(x)
