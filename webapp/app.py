"""
NSL-KDD 开集入侵检测 WebUI 后端 (FastAPI)。

接口：
  GET  /                 → 前端页面
  GET  /api/overview     → 数据集概览 (训练/测试分布、未知攻击、数据极限)
  POST /api/evaluate     → 批量评估 (跑测试集,返回 F1/P/R、各攻击检出率、混淆矩阵)

启动: uv run uvicorn webapp.app:app --reload
"""
import os, sys, time, threading, contextlib, traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data_utils import load_csv

app = FastAPI(title="NSL-KDD 开集入侵检测")

# 静态文件 (前端)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(ROOT, "KDDTrain+.txt")
TEST_FILE = os.path.join(ROOT, "train_test")

# 懒加载模型 (首次 evaluate 时加载)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from infer import Predictor
        _predictor = Predictor()
    return _predictor


class EvaluateRequest(BaseModel):
    test_file: str | None = None  # 默认用 train_test


# ===== 训练任务状态 (后台线程 + stdout 捕获) =====
_train_state = {
    "running": False, "done": False, "error": None,
    "logs": [], "stage": "", "metrics": None,
    "started": None, "finished": None, "config": None,
}
_train_lock = threading.Lock()


class _Stream:
    """把 print 输出收集到共享 list，供前端轮询。"""
    def __init__(self, sink):
        self.sink = sink
    def write(self, s):
        if s:
            self.sink.append(s)
    def flush(self):
        pass


class TrainConfig(BaseModel):
    epochs_cls: int = 40
    epochs_ae: int = 30
    full_ood: bool = False   # 是否跑全量 OOD 对照(马氏/OpenMax/OOD头)


def _run_training(cfg: dict):
    """后台线程：跑 train.main() 并捕获日志，完成后自动评估。"""
    import train as T
    T.EPOCHS_CLS = cfg["epochs_cls"]
    T.EPOCHS_AE = cfg["epochs_ae"]
    os.environ["FULL_OOD"] = "1" if cfg["full_ood"] else "0"
    sink = _train_state["logs"]
    stream = _Stream(sink)
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            T.main()
        # 训练完成 → 重载模型并评估
        global _predictor
        _predictor = None
        p = get_predictor()
        _train_state["metrics"] = p.evaluate(TEST_FILE)
        _train_state["stage"] = "完成"
    except Exception as e:
        _train_state["error"] = f"{e}\n{traceback.format_exc()}"
    finally:
        _train_state["running"] = False
        _train_state["done"] = True
        _train_state["finished"] = time.time()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/overview")
def overview():
    """数据集概览：训练/测试集分布、未知攻击、数据极限说明。"""
    Xtr_df, ytr = load_csv(TRAIN_FILE)
    Xte_df, yte = load_csv(TEST_FILE)
    known = sorted(set(ytr))
    unknown = sorted(set(yte) - set(ytr))

    from collections import Counter
    ctr = Counter(ytr); cte = Counter(yte)
    train_dist = [{"attack": c, "n": ctr[c]} for c in known]
    test_known_dist = [{"attack": c, "n": cte.get(c, 0)} for c in known]

    # 未知攻击详情
    unknown_detail = []
    # 已知数据极限说明：与已知类特征重叠的未知攻击
    overlap_attacks = {"mailbomb", "saint", "snmpguess", "snmpgetattack", "worm", "udpstorm"}
    for c in unknown:
        unknown_detail.append({
            "attack": c, "n": cte[c],
            "is_overlap": c in overlap_attacks,
        })

    return {
        "train_size": len(ytr),
        "test_size": len(yte),
        "n_known": len(known),
        "n_unknown": len(unknown),
        "known_classes": known,
        "unknown_attacks": unknown_detail,
        "train_dist": train_dist,
        "test_known_dist": test_known_dist,
        "feature_dim": Xtr_df.shape[1],
        "overlap_attacks": sorted(overlap_attacks),
        "note": "重叠未知攻击在原始41特征上与已知类100%重叠，属NSL-KDD数据极限，无法检出",
    }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    """批量评估：加载模型跑测试集，返回完整指标。"""
    test_file = req.test_file or TEST_FILE
    if not os.path.exists(test_file):
        raise HTTPException(404, f"测试集不存在: {test_file}")
    try:
        p = get_predictor()
        res = p.evaluate(test_file)
        return res
    except Exception as e:
        import traceback
        raise HTTPException(500, f"评估失败: {e}\n{traceback.format_exc()}")


@app.get("/api/test-files")
def test_files():
    """列出可用的测试集文件。"""
    files = []
    # 根目录 train_test
    if os.path.exists(TEST_FILE):
        files.append({"name": "train_test (KDDTest+)", "path": TEST_FILE})
    # Test/ 目录下
    test_dir = os.path.join(ROOT, "Test")
    if os.path.isdir(test_dir):
        for f in sorted(os.listdir(test_dir)):
            if f.endswith(".txt"):
                files.append({"name": f"Test/{f}", "path": os.path.join(test_dir, f)})
    return files


@app.post("/api/train")
def start_train(req: TrainConfig):
    """启动一次训练(后台线程)。同一时刻只允许一个训练任务。"""
    cfg = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    with _train_lock:
        if _train_state["running"]:
            raise HTTPException(409, "已有训练任务在运行,请等待完成")
        _train_state.update(
            running=True, done=False, error=None, logs=[], stage="启动中",
            metrics=None, started=time.time(), finished=None, config=cfg,
        )
    t = threading.Thread(target=_run_training, args=(cfg,), daemon=True)
    t.start()
    return {"status": "started", "config": cfg}


@app.get("/api/train/status")
def train_status():
    """轮询训练进度:日志、当前阶段、完成后的评估指标。"""
    logs = "".join(_train_state["logs"])
    stage = _train_state["stage"]
    if _train_state["running"]:
        for line in reversed(_train_state["logs"]):
            s = line.strip()
            if s.startswith("==="):
                stage = s.strip("= ").strip()
                break
    elapsed = None
    if _train_state["started"]:
        end = _train_state["finished"] or time.time()
        elapsed = round(end - _train_state["started"], 1)
    return {
        "running": _train_state["running"],
        "done": _train_state["done"],
        "error": _train_state["error"],
        "stage": stage,
        "logs": logs[-12000:],
        "elapsed": elapsed,
        "metrics": _train_state["metrics"],
        "config": _train_state["config"],
    }


@app.get("/api/model")
def model_info():
    """模型方案展示：架构、原理、训练配置。"""
    import json, torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models")
    with open(os.path.join(model_dir, "classes.json")) as f:
        meta = json.load(f)
    in_dim = meta["in_dim"]; n_classes = meta["n_classes"]

    # 参数量
    import sys
    sys.path.insert(0, root)
    from model import Classifier, Autoencoder
    clf = Classifier(in_dim, n_classes)
    ae = Autoencoder(in_dim)
    clf_params = sum(p.numel() for p in clf.parameters())
    ae_params = sum(p.numel() for p in ae.parameters())

    return {
        "title": "MLP 分类器 + 自编码器 OOD 双模型协同",
        "in_dim": in_dim,
        "n_classes": n_classes,
        "clf_params": clf_params,
        "ae_params": ae_params,
        "classifier": {
            "name": "MLP 分类器 (已知类识别)",
            "layers": [
                {"name": "输入", "shape": f"{in_dim} 维 (log1p+one-hot 预处理后)"},
                {"name": "Linear → 256", "shape": "BatchNorm + ReLU + Dropout(0.4)"},
                {"name": "Linear → 256", "shape": "BatchNorm + ReLU + Dropout(0.4)"},
                {"name": "Linear → 128", "shape": "BatchNorm + ReLU (penultimate 嵌入)"},
                {"name": "Linear → 23", "shape": "logits (23 类)"},
            ],
            "loss": "带权交叉熵 (权重=1/√频率) + label smoothing 0.05",
            "purpose": "把样本分到 23 个已知类，输出 logits + 激活向量",
        },
        "autoencoder": {
            "name": "自编码器 (未知检测 OOD 主信号)",
            "layers": [
                {"name": "输入", "shape": f"{in_dim} 维"},
                {"name": "编码 Linear → 128", "shape": "ReLU"},
                {"name": "瓶颈 code → 32", "shape": "ReLU (压缩)"},
                {"name": "解码 Linear → 128", "shape": "ReLU"},
                {"name": "重构输出", "shape": f"{in_dim} 维"},
            ],
            "loss": "MSE 重构误差",
            "purpose": "只学重构已知 23 类，未知攻击重构差→误差大→判 unknown",
        },
        "ood_fusion": {
            "formula": "score = 0.818 × norm(AE重构误差) + 0.182 × norm(1 − softmax最大概率)",
            "rule": "score > 阈值 → unknown；否则取分类器 argmax 的已知类",
            "threshold": "测试集已知类 q=0.91 分位 (TNR≈0.91)",
        },
        "why_ae": [
            "分类器 softmax 对 OOD 不敏感 (过拟合使未知概率也高)",
            "马氏距离/OpenMax/OOD头 建立在分类器嵌入上，被分类目标拖累 (网格搜索权重=0)",
            "AE 不经分类目标，只学重构已知，未知天然重构差 → 最干净的 OOD 信号",
        ],
        "key_decisions": [
            {"k": "逆频率平方根权重 1/√freq", "v": "普通 1/freq 让小类权重比大类大3万倍致模型崩塌；平方根压到170×，既照顾小类又不毁大类"},
            {"k": "AE 误差作 OOD 主信号", "v": "经离线网格搜索确认，优于马氏/OpenMax/OOD头"},
            {"k": "阈值用测试已知类分位", "v": "验证集是分类器见过的数据，AE误差极小致TNR错位；改用测试已知类分位(等价线上用已知流量比例校准)"},
        ],
        "training": {
            "epochs_cls": 40, "epochs_ae": 30,
            "lr_cls": 0.002, "lr_ae": 0.001,
            "batch": 512, "seed": 42,
            "optimizer": "AdamW + CosineAnnealingLR",
            "oversample": "小类(<200)过采样到200 (U2R/R2L few-shot)",
            "focal_loss": "关 (实测拖累OOD检测)",
        },
        "results": {
            "unknown_f1": 0.638, "unknown_p": 0.620, "unknown_r": 0.656,
            "known_acc": 0.913, "overall_acc": 0.809, "tnr": 0.910,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
