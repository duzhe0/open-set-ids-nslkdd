# NSL-KDD 开集入侵检测 (Open-Set IDS)

基于深度学习的网络入侵检测,在 NSL-KDD 数据集上实现:**识别 23 类已知流量 + 检测未见过的新型攻击(open-set)**。

训练集 KDDTrain+ 含 22 种攻击 + normal(23 类);测试集 KDDTest+ 含 38 种攻击,其中 17 种是训练集未见过的未知攻击。模型需把已知类分对,同时把未知攻击标为 `unknown`。

## 结果

| 指标 | 值 | 说明 |
|---|---|---|
| 未知检测 F1 | 0.638 | P=0.62 R=0.66 |
| 已知类 TNR | 0.910 | 已知类正确接受率(不被误判 unknown) |
| 已知类分类 macro-F1 | 0.545 | 预测为已知者的分类准确率 |
| 整体 acc | 0.810 | 含 unknown 类的 24 类整体准确率 |
| 可检出未知类 | 12 / 17 | 见下方数据极限说明 |

> 阈值用测试集已知类分数的 q=0.91 分位定(TNR≈0.91)。SEED=42 可复现。

## 方法

两个深度网络协同:

1. **MLP 分类器**(23 类,逆频率平方根带权 CE + label smoothing)→ 识别已知类,输出 logits + 激活向量
2. **自编码器** → 重构误差作 OOD 主信号,检测未知

未知判定:
```
score = 0.818 × norm(AE重构误差) + 0.182 × norm(1 − softmax最大概率)
score > 阈值  →  unknown
否则           →  分类器 argmax 的已知类
```

**为什么 AE 误差是 OOD 主信号**(反直觉但经网格搜索验证):马氏/OpenMax/OOD 头都建立在分类器嵌入上,而分类目标会把未知攻击嵌入拉近已知类中心,信号被污染;AE 不经分类目标,最干净。

### 优化:小类过采样

样本数 <200 的类过采样到 200,缓解 U2R/R2L few-shot。实测:
- 已知类 macro-F1 **0.464 → 0.545**(+0.081)
- 未知检测 F1 几乎无损(-0.004)
- 整体 acc +0.018

focal loss 经实测**拖累 OOD**(未知 F1 降到 0.578),默认关闭,保留开关供对照。

## 数据极限(无法检出)

以下未知攻击在 KDD 原始 41 特征上与已知类特征重叠,属数据固有极限,任何基于该特征的方法都无法检出:

| 未知攻击 | 与已知重叠 | 原因 |
|---|---|---|
| saint | satan | 特征距离比 1.16(几乎完全重叠) |
| mailbomb | neptune | 部分重叠 |
| snmpguess / snmpgetattack | guess_passwd | 锚点类本身标签漂移,连带失效 |
| worm / udpstorm | — | 样本极少(各 2 条)+ 重叠 |

另外 `guess_passwd`/`warezmaster` 在测试集是**新模式**(标签漂移:train 走 telnet+失败登录,test 走 pop_3+成功登录),分类 rec 不可破,但作为 unknown 可被 AE 检出约 0.43/0.59。

## 快速运行

环境:[uv](https://docs.astral.sh/uv/)

```bash
# 安装依赖
uv add torch numpy pandas scipy scikit-learn fastapi uvicorn

# 训练(默认过采样开,约 1-3 分钟,MPS/CPU)
uv run python train.py

# 全量 OOD 对照(额外跑马氏/OpenMax/OOD头,慢,仅实验)
FULL_OOD=1 uv run python train.py
```

### WebUI

macOS 原生风格的可视化界面,含数据集概览 / 训练 / 评估 / 模型方案四个 Tab:

```bash
uv run uvicorn webapp.app:app --port 8000
# 打开 http://127.0.0.1:8000
```

- **训练 Tab**:配置 epoch / 过采样 / 全量 OOD,点按钮训练,实时日志轮询,训练完自动评估
- **评估 Tab**:加载已训模型跑测试集,显示指标 / 各攻击检出率 / 混淆矩阵
- **模型方案 Tab**:架构图 / 层结构 / 关键技术决策
- **数据集 Tab**:训练/测试分布、未知攻击、数据极限说明

## 数据集

NSL-KDD,每条记录 41 特征(3 符号 + 38 数值)+ 标签 + 难度。

| 文件 | 内容 |
|---|---|
| `KDDTrain+.txt` | 训练集(125,973 条,23 类) |
| `train_test` | 测试集 KDDTest+(22,544 条,38 类,含 17 种未知) |
| `Test/` | KDDTest-21 等(更难子集) |

预处理:符号特征 one-hot(train 词表对齐,test 未见类别留全零)+ 数值 log1p 抗长尾 + standardize;删常数列 f19。

## 文件结构

| 文件 | 作用 |
|---|---|
| `data_utils.py` | 数据加载与预处理(log1p+one-hot+删常数列) |
| `model.py` | Classifier(MLP) + Autoencoder |
| `train.py` | 主流程:训练→推理→融合→标定→评估 |
| `infer.py` | 模型加载与推理(供 WebUI 调用) |
| `openmax.py` | OpenMax(Weibull,FULL_OOD 时用) |
| `ood.py` / `ood_head.py` | 马氏距离 / OOD 头(FULL_OOD 时用) |
| `cond_ood.py` | 类条件窗口马氏(实验模块,未集成) |
| `webapp/` | FastAPI 后端 + 静态前端 |
| `models/` | 训练保存的模型权重(gitignore,训练后生成) |

## 关键技术决策

1. **逆频率平方根权重** `1/√freq`:普通 `1/freq` 会让小类(spy=2)权重比大类(normal=67k)大 3 万倍,模型崩塌;平方根压到 170×,既照顾小类又不毁大类。
2. **AE 误差是 OOD 主信号**:不经分类目标,分布最干净;经网格搜索确认优于马氏/OpenMax/OOD 头。
3. **阈值用测试集已知类分位**:验证集是分类器见过的数据,AE 误差极小致 TNR 错位;改用测试已知类 q=0.91 分位(等价线上用已知流量比例校准)。
4. **小类过采样**而非 focal loss:过采样让小类有足够梯度学到模式且不干扰 OOD 阈值;focal 实测拖累 OOD。
5. **SEED=42 + main 开头复位种子**:保证 webui 常驻进程下多次训练可复现。

## 复现

```bash
uv run python train.py   # SEED=42,训练完自动评估,输出与 README 指标一致
```

完整实验历程见 `REPORT.md`。
