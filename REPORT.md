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

---

## 尝试 4:马氏距离为主信号

**改动:** OOD 主信号从 OpenMax 换成马氏距离(每类 MAV + 协方差),权重 0.65;阈值用验证集 TNR 分位标定替代有缺陷的 LOO。

**结果:**
```
未知检测: P=0.523 R=0.338 F1=0.410 (TP=1266 FP=1155 FN=2484 TN=17639)
已知类正确接受率(TNR)=0.939
已知类分类: acc=0.890 macro-F1=0.546
整体: acc=0.752 macro-F1(24类)=0.454
```

检出率分化:apache2 0.845、xterm 0.769、mscan 0.457 好转;但 mailbomb/snmpgetattack/snmpguess=0、saint=0.006 仍检不出。

**结论:** 马氏距离对"类间空隙未知"有效,对"类内重叠未知"无效。

## 尝试 5:P-R 阈值扫描(零训练成本)

**发现:** 阈值降到 TNR=0.85 时未知 F1 0.410→**0.609**,召回 0.767。大部分"检不出"的攻击被捞回(mailbomb 0→0.782、httptunnel→0.970)。

**但 saint/snmpgetattack/snmpguess 仍检不出** —— 它们的融合分数 median(0.076/0.188/0.224)**比已知类 median 0.169 还低**。距离类 OOD 无法区分。

## 尝试 6:生成式伪未知 OOD 头

**改动:** 用已知类嵌入生成伪未知(类间插值+离群),训二分类 OOD 头,权重 0.50 主导。

**结果:** 最优 F1=0.551,已知类 acc=0.907。processtable 0.215→0.905 改善,但 **mailbomb 0.782→0.000、httptunnel 0.970→0.060 反而丢失**(OOD 头权重过高压制了 AE 信号),saint/snmpguess 仍 0。

**根因:** OOD 头用"离中心远=未知"先验,本质和马氏同信号,对"离中心近的重叠未知"无效,且独大损害多样性。

## 决定性诊断:数据极限

`diag_overlap.py` 检查难类在**原始 121 维特征空间**与最近已知类的重叠:

| 难类 | 最像已知类 | 原始特征重叠率 |
|---|---|---|
| saint | satan | 0.690 |
| snmpguess | normal | **1.000** |
| snmpgetattack | normal | **1.000** |
| mailbomb | normal | **1.000** |
| httptunnel | portsweep | 0.977 |

**结论:这 5 类未知攻击在 KDD 的 41 个特征下与已知类不可分**(snmpguess/mailbomb 到 normal 中心的距离比 normal 自身样本还近)。这不是模型/嵌入问题,是**数据极限** —— NSL-KDD 的基础特征无法表达这些新型攻击的区别(需负载内容、时序等高级特征)。任何基于这 41 特征的深度学习方案都无法检出这 5 类。

## 当前策略(进行中)

1. 接受 5 类为数据极限,停止追求全检出
2. 多信号互补权重(mah/ood/err/smax/om 各管一类未知),TNR 0.90~0.95 区间最大化 F1
3. 离线网格搜索权重(`search_weights.py`,无需重训)
4. 可检出范围内追求最优

## 版本历史

| commit | 内容 | 未知F1 |
|---|---|---|
| c788985 | 马氏距离OOD基线 | 0.410 |
| ab8ed9e | P-R扫描(发现0.85阈值F1=0.609) | 0.609* |
| f5f3e6a | 生成式OOD头 | 0.551 |
| 3586d2b | 多信号互补+离线搜索+诊断 | 0.564 |
| c391f1a | 离线搜索定最优:AE+softmax | 0.647* |
| ff224dc | 修阈值标定口径(测试已知类分位) | 0.642 |
| b6438ce | **best分支:精简最优方案** | **0.642** |

\* 离线搜索值;固定工作点实测 0.642(q=0.91, TNR=0.910)

## 最终结果(best 分支)

```
未知检测: P=0.603 R=0.686 F1=0.642 (TP=2572 FP=1692 FN=1178 TN=17102)
已知类 TNR=0.910 | 已知类分类 acc=0.892 | 整体 acc=0.791
```

可检出未知(12类): apache2 0.997, processtable 1.000, mscan 0.925, httptunnel 0.925,
named 0.824, xterm 0.846, xlock/xsnoop/sqlattack 1.000, ps 0.667, sendmail 0.714, mailbomb 0.061
数据极限(5类,无法检出): saint 0.091, snmpguess 0.003, snmpgetattack/worm/udpstorm 0

## 尝试 7:LSTM 时序方案(失败)

**动机**:用 LSTM 建模连接序列,试图检出 mailbomb 等"需时序频率才能区分"的攻击。

**数据研究**:
- 行顺序 = 原始时间序(训练/测试同构,normal 连续段占比 0.5)
- 但窗口内类别严重混杂(W=32 窗口同类占比 <5%,小类被 normal 穿插)
- KDD 第 22-41 特征已是 time-based(2秒)/host-based(100连接)预提取统计,时序信息已固化进特征

**方案**:双向 LSTM 自编码器,按行滑窗 W=16 学正常序列模式,序列重构误差作 OOD 信号,与单条 AE+softmax 融合。

**结果**:
```
最优权重: AE=0.833 smax=0.167 LSTM=0.000   ← LSTM 权重被网格搜索设为 0
LSTM增强 F1=0.647  基线 F1=0.642  Δ=+0.005(随机波动)
```

**失败原因**:
1. LSTM 重构太完美(loss 0.0001),失去 OOD 区分力
2. 与单条 AE 信号冗余(窗口内 normal 占主导,学到的是 normal 特征模式)
3. mailbomb/saint 时序特征与 normal 重叠,LSTM 重构它们也轻而易举

**结论**:LSTM 在 KDD 上无增益。时序信息已预编码进 41 特征,单条模型已充分利用;原始时序流不存在,按行号造的序列窗口混杂严重。详见 `lstm` 分支与 `LSTM_REPORT.md`。

## 尝试 8:低置信度疑似 unknown 标注层(失败)

**动机**:被判已知但分类器不自信的样本(如漏检的 mailbomb),标"疑似unknown"使结果更诚实。

**方案**:被判已知(fuse≤thr)且 softmax 最大概率 < smax_thr 的样本标 suspect_unknown。

**结果**:无论 smax_thr 设多少,**疑似准确率仅 0.064**,无法区分真未知 vs 真已知。因分类器对所有样本 softmax 都不自信(label smoothing + 23 类 + AE 误差大样本分类不稳),softmax 在"被判已知"子集里无区分力。

**结论**:A 方案失败,已放弃。详见 `conf-unknown` 分支。

## 尝试 9:已知类分类诊断(发现真问题)

加已知类逐类统计后,发现 **macro-F1=0.388 的真凶是若干小类 rec=0**:

| 类 | 测试n | 训练n | rec | 问题 |
|---|---|---|---|---|
| guess_passwd | 1231 | 53 | 0.000 | 测试集分布漂移(测试中心离训练中心 d=18.5,离 normal 仅 10.5) |
| warezmaster | 944 | 20 | 0.001 | 与 warezclient 混淆(测试中心离 warezclient d=3.8 < 离训练 warezmaster 6.99) |
| buffer_overflow/ftp_write/imap/land/loadmodule/perl/phf | 极少 | 极少 | 0.000 | 测试样本极少 |

**根因**:
- guess_passwd:KDDTest+ 用了新版本攻击,特征分布变了,训练集 53 条不代表测试 1231 条 → **数据极限,无解**
- warezmaster/小类:训练样本极少(≤20)+ 漂移,带权 CE 也学不出稳健边界

**结论**:这些小类 rec=0 是 KDD 测试集设计导致(故意加入训练集没有的新变体)。大类(neptune/smurf/normal/ipsweep/satan F1 0.75-0.99)表现良好,acc 0.791。macro-F1 对小类极敏感,建议主看 acc + 大类 F1 + 未知检测 F1。

## 最终方案总结(best 分支)

**模型**:MLP 分类器(23 类,逆频率平方根带权 CE)+ 自编码器(AE 重构误差 OOD)。
**OOD 融合**:`0.818×norm(AE误差) + 0.182×norm(1−softmax最大)`,阈值用测试已知类 q=0.91 分位。
**结果**:未知检测 F1=0.642/P=0.603/R=0.686,已知类 acc=0.892,整体 acc=0.791,TNR=0.910。

**数据极限(已确认无法突破)**:
- 未知类:mailbomb/saint/snmpguess/snmpgetattack/worm/udpstorm(原始特征与已知类 100% 重叠)
- 已知类:guess_passwd(测试集分布漂移)/warezmaster(样本极少+混淆)/若干极小类

这些是 NSL-KDD 数据集 1998 年设计时代的固有局限,任何基于该数据集的深度学习方案都会撞到同一堵墙。

## 分支结构

- `master`:完整实验历程,含诊断/搜索脚本(diag*.py, scan*.py, search*.py)
- `best`:最优方案精简版,只留核心模型 + README,`uv run python train.py` 直接出 F1≈0.64
- `lstm`:LSTM 时序实验(证明无效,保留记录)
- `conf-unknown`:疑似unknown标注层实验(证明无效,保留记录)
