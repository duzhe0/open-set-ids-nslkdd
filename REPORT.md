# NSL-KDD 开集入侵检测 — 尝试报告

> 任务:用 KDDTrain+(23 类:22 攻击 + normal)训练深度模型,对 `train_test` 分类。
> 要求:已知 23 类识别 + 未知攻击标注为 unknown。
> 环境:uv + PyTorch(MPS / Mac GPU)。纯深度学习方案。

## 数据摸底

| 项 | 训练 KDDTrain+ | 测试 train_test |
|---|---|---|
| 样本数 | 125,973 | 22,544 |
| 类数 | 23(含 normal) | 38(含 normal) |
| 特征 | 41(3 符号 + 38 数值) | 同 |
| 未知攻击(test 独有) | — | 17 种,共 3,750 条 |

**关键事实:**
- 数值长尾极端:`src_bytes`/`dst_bytes` median=0/44 但 max≈1.3e9 → 必须 `log1p` 压缩
- `f19`(num_outbound_cmds)训练集 0 方差 → 删除
- 类别极不平衡:normal 67k、neptune 41k,而 spy=2、perl=3、phf=4
- train↔test 分布漂移:`guess_passwd` 53→1231、`warezmaster` 20→944;service 频率漂移明显;`warezclient`/`spy` 在 test 缺失
- 未知攻击与已知语义相近:saint≈satan、mscan≈ipsweep、apache2≈smurf、snmpguess≈guess_passwd → 单一 OOD 信号不够

## 方案架构

1. 预处理:`log1p` + standardize(数值) + one-hot(符号,train 词表,test 未见类别全零)
2. MLP 分类器(23 类,BN+Dropout,带权 CE)
3. 自编码器 → 重构误差(OOD 辅助信号)
4. OpenMax(Weibull,激活向量)→ unknown 概率
5. 多信号融合 + LOO 阈值标定
6. 评估:未知检测 P/R/F1 + 已知类 macro-F1 + 各未知攻击检出率

---

## 尝试记录

### 尝试 1:初始方案(逆频率带权 CE + OpenMax + 融合)

**结果:**
```
未知检测: P=0.228 R=0.143 F1=0.176 (TP=537 FP=1815 FN=3213 TN=16979)
已知类正确接受率(TNR)=0.903
已知类分类: acc=0.004 macro-F1=0.200   ← 几乎全错
整体(含unknown): acc=0.027 macro-F1(24类)=0.139
```

**问题诊断:** 跑诊断脚本发现分类器在**训练集自身** acc 仅 1.68%。小类(spy/perl/phf 等 acc=1.0)全对,大类(normal/neptune/satan acc=0.0)全错 —— 模型崩塌。

**根因:** `1/freq` 的带权 CE 让 spy(2 样本)权重是 normal(67k)的 ~3 万倍,模型把几乎所有样本都预测成小类以压低小类 loss。loss=72 也异常高。

### 尝试 2:修复类别权重(逆频率平方根)

**改动:** `class_weights` 从 `1/freq` 改为 `1/sqrt(freq)`,权重差异从 ~30000× 压到 ~170×。

**诊断结果(分类器单独):**
```
训练集自身 acc=0.9981   (大类全部 ≥0.997)
测试集已知类 acc=0.8378
```
分类器修复成功。

### 尝试 3:完整流水线(修复后)

**结果:**
```
未知检测: P=0.120 R=0.017 F1=0.029 (TP=63 FP=464 FN=3687 TN=18330)
已知类正确接受率(TNR)=0.975
已知类分类: acc=0.858 macro-F1=0.586   ← 分类器正常了
整体(含unknown): acc=0.700 macro-F1(24类)=0.481

LOO 标定: thr=0.3553 | LOO伪未知 F1=0.002 | 伪未知 recall=0.001   ← 标定失败
```

各未知攻击检出率(apache2 0.058、mailbomb 0.000、mscan 0.003、processtable 0.000、snmpgetattack 0.000、snmpguess 0.000 ...)几乎全挂。

**问题诊断:** 分类器修好了(已知类 acc 0.858),但**未知检测反过来几乎检不出**(R=0.017)。根因在 LOO 标定:`伪未知 recall=0.001`。

**根因分析:**
1. 分类器在训练集上 acc=0.998、softmax 极度自信 → softmax 最大概率对 OOD 几乎无区分力(业界已知:过拟合分类器的 softmax 对 OOD 不敏感)
2. OpenMax 的 Weibull tail 在分类器几乎完美分类时也不显著
3. **LOO 标定方法本身有根本缺陷**:用训练好的分类器对**训练集自身**的留出类算分数,但这些样本分类器见过、嵌入就在自己类中心附近 → "伪未知"分数和真已知一样低 → 阈值被卡到很高 → 真未知全被当已知

---

## 当前结论

- ✅ 数据预处理、分类器训练、AE 训练均已可用
- ✅ 已知 23 类分类:测试已知类 acc=0.858, macro-F1=0.586
- ❌ 未知检测失效:OOD 信号(softmax/OpenMax/AE)对未知攻击无区分力,LOO 标定方法有缺陷

## 下一步方向(进行中)

1. **改 OOD 主信号为马氏距离**:用分类器 penultimate embedding,每类拟合 MAV + 类内协方差 → 马氏距离。比 softmax/OpenMax 对 OOD 更敏感(几何结构而非概率饱和)。
2. **改阈值标定**:马氏距离有 χ² 理论分位点,可直接用统计阈值(如 99.5% 分位)替代有缺陷的 LOO;用验证集已知类样本校准 TNR。
3. 协方差正则化(小样本类用全局协方差 fallback)。

> 状态:OOD 模块 `ood.py`(马氏距离)已写好,正接入 `train.py` 替换 OpenMax 主信号。
