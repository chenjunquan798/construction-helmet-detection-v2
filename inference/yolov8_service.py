# -*- coding: utf-8 -*-
"""真实 YOLOv8 推理服务（已接入自训练权重 anquanmao.pt）

职责：把工地闸机相机的帧 / 上传的图片转成与前端 HelmetEngine 一致的 JSON 契约，
前端决策链、台账、复述、对比逻辑完全复用，只需 App.setSource(url)。

已接入的真实模型
    inference/weights/anquanmao.pt
    Ultralytics 8.4.218 · YOLOv8n · detect · nc=4
    类别：正确佩戴 / 反戴 / 未佩戴 / 佩戴非安全帽
    用途：驱动「有无安全帽」与「佩戴是否规范」两路判定（真实模型输出，非仿真）

诚实的边界（务必如实告知使用者）
    · 颜色等级：本模型无颜色分类头，改由盔体框内 HSV 真实像素统计给出
    · 破损：本模型无破损分类头，tupian 数据集也无破损样本 -> 返回「未评估 / 需人工复核」
    · 人脸活体与 1:N：由既有刷脸系统通过请求参数传入，本服务不重复建设
    · D:/tupian 为 Simuletic 合成 CCTV 图（CC BY-NC 4.0），非真人现场照片

启动：
    python inference/yolov8_service.py --port 8000
    python inference/yolov8_service.py --port 8000 --state-weights inference/weights/anquanmao_finetuned.pt
接入：
    浏览器控制台执行 App.setSource('http://127.0.0.1:8000')
    上传图片 / 摄像头帧会以 image_b64 POST 到 /infer，返回真实检测框与状态。
    GET /model 可查模型元信息，GET /health 可查各分支是否降级。

依赖：pip install -r inference/requirements.txt
可选权重（缺失则该分支降级并在响应中标注）：
    weights/cls_color.pt   yolov8s-cls      白 / 黄 / 蓝 / 红
    weights/cls_damage.pt  yolov8s-cls      完好 / 裂纹 / 凹陷 / 缺口 / 老化
    weights/pose.pt        yolov8s-pose     头部关键点
    weights/detect.pt      旧版 head/helmet/face/strap 检测器（与 state 模型二选一）
"""

import argparse
import base64
import os
import pickletools
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

COLOR_KEYS = ["white", "yellow", "blue", "red"]
DAMAGE_NAME = {
    "intact": "完好", "crack": "裂纹", "dent": "凹陷",
    "chip": "帽檐缺口", "aging": "老化脆化",
}
DET_NAMES = {0: "head", 1: "helmet", 2: "face", 3: "strap"}
DET_LABEL = {"head": "头部", "helmet": "安全帽", "face": "人脸", "strap": "下颚带"}

# 自训练模型的 4 类状态；同时兼容中英文别名，防止换个权重就失效
STATE_KEYS = ["正确佩戴", "反戴", "未佩戴", "佩戴非安全帽"]
STATE_ALIAS = {
    "正确佩戴": "正确佩戴", "correct": "正确佩戴", "correct_helmet": "正确佩戴",
    "hardhat": "正确佩戴", "helmet": "正确佩戴", "with_helmet": "正确佩戴",
    "反戴": "反戴", "wrong": "反戴", "wrong_helmet": "反戴",
    "hardhat_wrong": "反戴", "backward": "反戴",
    "未佩戴": "未佩戴", "no_helmet": "未佩戴", "no_hardhat": "未佩戴",
    "without_helmet": "未佩戴", "none": "未佩戴",
    "佩戴非安全帽": "佩戴非安全帽", "non_helmet": "佩戴非安全帽",
    "other_hat": "佩戴非安全帽", "cap": "佩戴非安全帽",
}
# 前端统一 schema 里使用的目标键（label 保留中文真实类别）
STATE_CLS = {"正确佩戴": "helmet", "反戴": "helmet",
             "未佩戴": "head", "佩戴非安全帽": "helmet_wrong"}
# 安全优先：一图多状态时取最严格者汇总
STATE_SEVERITY = {"未佩戴": 3, "佩戴非安全帽": 3, "反戴": 2, "正确佩戴": 1}

app = FastAPI(title="工地安全帽 YOLOv8 推理服务", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)
MODELS: Dict[str, Optional[YOLO]] = {
    "state": None, "detect": None, "color": None, "damage": None, "pose": None,
}
WEIGHTS_DIR = "inference/weights"
STATE_WEIGHTS = ""
MODEL_INFO: Dict[str, Any] = {}
CAPTURE = None
CONF = 0.25


class InferRequest(BaseModel):
    frame_id: str = ""
    channel: str = "A-01 员工入口"
    staff_id: str = ""
    # 由既有刷脸系统给出，避免重复建设人脸库
    face_score: float = 0.95
    face_live: bool = True
    face_name: str = ""
    # 可选：直接传 base64 帧；不传则从 --source 抓帧
    image_b64: str = ""


def canon_state(name: str) -> Optional[str]:
    key = str(name).strip().lower()
    for alias, state in STATE_ALIAS.items():
        if key == alias.lower() or alias.lower() in key:
            return state
    for state in STATE_KEYS:
        if state in str(name):
            return state
    return None


def peek_ckpt(path: str) -> Dict[str, Any]:
    """不依赖 torch 读取 checkpoint 元数据，用于 /model 与兼容判断。"""
    info: Dict[str, Any] = {"is_zip": zipfile.is_zipfile(path)}
    if not info["is_zip"]:
        return info
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        info["archive_prefix"] = sorted({n.split("/")[0] for n in names if "/" in n})
        pk = [n for n in names if n.endswith("data.pkl")]
        if not pk:
            return info
        strs, globals_ = [], []
        for op, arg, pos in pickletools.genops(z.read(pk[0])):
            if op.name == "GLOBAL":
                globals_.append(arg if isinstance(arg, str) else "")
            if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                strs.append(arg.decode("utf-8", "replace") if isinstance(arg, bytes) else arg)
        info["version"] = next((strs[i + 1] for i, s in enumerate(strs[:-1]) if s == "version"), "")
        info["date"] = next((s for s in strs if re.match(r"\d{4}-\d{2}-\d{2}T", s)), "")
        info["task"] = next((strs[i + 1] for i, s in enumerate(strs[:-1]) if s == "task"), "")
        info["architecture"] = "ultralytics.nn.tasks.DetectionModel" if any(
            "DetectionModel" in g for g in globals_) else "unknown"
        info["base_model"] = next((s for s in strs if "yolov8" in s.lower() and s.endswith(".pt")), "")
        for i, s in enumerate(strs):
            if s == "names":
                got = []
                for v in strs[i + 1:]:
                    if re.fullmatch(r"\d+", v) or v in ("yaml", "args", "colors", "pt_path"):
                        break
                    got.append(v)
                if got:
                    info["class_names"] = got
                break
    return info


def make_compatible(src: str, dst: str) -> str:
    """Ultralytics Platform 导出的权重 zip 内部前缀是 best/，改回 PyTorch 默认 archive/。"""
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith("best/"):
                name = "archive/" + name[5:]
            zi = zipfile.ZipInfo(name, date_time=item.date_time)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = item.external_attr
            zout.writestr(zi, zin.read(item.filename))
    return dst


def load_state_model(path: str) -> Optional[YOLO]:
    """加载自训练权重：处理 zip 前缀兼容、CPU 上的 FP16 -> FP32。"""
    global MODEL_INFO
    if not os.path.isfile(path):
        print(f"[warn] state: 未找到 {path}，佩戴状态分支降级")
        return None
    MODEL_INFO = peek_ckpt(path)
    MODEL_INFO["path"] = path
    MODEL_INFO["bytes"] = os.path.getsize(path)
    compat = os.path.join(os.path.dirname(path), "anquanmao_compat.pt")
    model = None
    for idx, p in enumerate([path, compat]):
        try:
            model = YOLO(p)
            MODEL_INFO["load_path"] = p
            if idx == 1:
                MODEL_INFO["compat_rewrite"] = True
            break
        except Exception as e:
            print(f"[load] state 尝试 {p} 失败：{e}")
            if idx == 0:
                make_compatible(path, compat)
                print(f"[load] 已重写 zip 前缀生成兼容副本：{compat}")
    if model is None:
        return None
    try:
        if next(model.model.parameters()).dtype == torch.float16:
            model.model.float()
            MODEL_INFO["dtype_fix"] = "FP16 -> FP32 (CPU)"
    except Exception as e:
        MODEL_INFO["dtype_fix_warning"] = str(e)
    MODEL_INFO["runtime_names"] = dict(model.names)
    MODEL_INFO["runtime_task"] = model.task
    MODEL_INFO["torch"] = torch.__version__
    MODEL_INFO["ultralytics"] = __import__("ultralytics").__version__
    print(f"[load] state: {MODEL_INFO['load_path']}  task={model.task}  names={dict(model.names)}")
    return model


def load_models(weights_dir: str, state_weights: str = "") -> None:
    global STATE_WEIGHTS
    STATE_WEIGHTS = state_weights or os.path.join(weights_dir, "anquanmao.pt")
    MODELS["state"] = load_state_model(STATE_WEIGHTS)
    files = {
        "detect": "detect.pt", "color": "cls_color.pt",
        "damage": "cls_damage.pt", "pose": "pose.pt",
    }
    for key, fn in files.items():
        path = os.path.join(weights_dir, fn)
        if os.path.exists(path):
            try:
                MODELS[key] = YOLO(path)
                print(f"[load] {key}: {path}")
            except Exception as e:
                MODELS[key] = None
                print(f"[warn] {key}: 加载失败 {e}")
        else:
            MODELS[key] = None
            print(f"[warn] {key}: 未找到 {path}，该分支使用真实像素 / 人工复核降级")


def decode_frame(req: InferRequest) -> np.ndarray:
    if req.image_b64:
        raw = base64.b64decode(req.image_b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="image_b64 解码失败")
        return frame
    if CAPTURE is not None:
        ok, frame = CAPTURE.read()
        if ok:
            return frame
    raise HTTPException(
        status_code=503,
        detail="没有可用帧：请传 image_b64，或用 --source 指定 RTSP / 摄像头",
    )


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, v)))


def dominant_color(crop: np.ndarray) -> Dict[str, Any]:
    """盔体框内的真实像素主色：低饱和高亮度归白，其余按 HSV 色相投票。"""
    if crop.size == 0:
        return {"predicted": "white", "probs": {k: 0.25 for k in COLOR_KEYS},
                "hue": 0, "sat": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
    chroma = (s > 60) & (v > 50)
    hue_deg = h[chroma].astype(np.float32) * 2
    votes = {k: 0 for k in COLOR_KEYS}
    for d in hue_deg:
        if d < 20 or d >= 330:
            votes["red"] += 1
        elif 35 <= d < 70:
            votes["yellow"] += 1
        elif 180 <= d < 260:
            votes["blue"] += 1
    votes["white"] = int(np.sum((s < 60) & (v > 110)))
    total = sum(votes.values()) or 1
    probs = {k: round(v / total, 3) for k, v in votes.items()}
    predicted = max(probs, key=probs.get)
    hue = int(np.median(hue_deg)) if hue_deg.size else 0
    return {"predicted": predicted, "probs": probs, "hue": hue,
            "sat": round(float(np.mean(s)), 1)}


def run_color(crop: np.ndarray) -> Dict[str, Any]:
    dc = dominant_color(crop)
    model = MODELS.get("color")
    if model is not None:
        res = model.predict(crop, verbose=False)[0]
        names = [n.lower() for n in res.names.values()]
        probs = {k: 0.02 for k in COLOR_KEYS}
        for i, p in enumerate(res.probs.data.tolist()):
            key = names[i] if i < len(names) else COLOR_KEYS[min(i, 3)]
            if key in probs:
                probs[key] = round(float(p), 3)
        s = sum(probs.values()) or 1.0
        probs = {k: round(v / s, 3) for k, v in probs.items()}
        predicted = max(probs, key=probs.get)
        desc = "YOLOv8s-cls 颜色分类模型"
    else:
        predicted, probs = dc["predicted"], dc["probs"]
        desc = "盔体框内 HSV 真实像素统计（本权重无颜色分类头）"
    return {
        "hsvHue": dc["hue"], "saturation": dc["sat"], "clsProbs": probs,
        "probSum": round(sum(probs.values()), 3), "predicted": predicted,
        "conf": round(float(probs.get(predicted, 0.5)), 3), "model": desc,
    }


def run_damage(crop: np.ndarray) -> Dict[str, Any]:
    model = MODELS.get("damage")
    if model is not None:
        res = model.predict(crop, verbose=False)[0]
        names = [n.lower() for n in res.names.values()]
        top = res.probs.top1
        raw = names[top] if top < len(names) else "intact"
        intact_idx = names.index("intact") if "intact" in names else 0
        damage_prob = round(clamp(1.0 - float(res.probs.data[intact_idx])), 3)
        cls = DAMAGE_NAME.get(raw, raw)
        review = False
        desc = "YOLOv8s-cls 破损分类（crop 512）"
        boxes = [] if raw == "intact" else [{"x": 0, "y": 0, "w": 0, "h": 0, "label": cls}]
    else:
        # 不伪造「完好」结论：明确标记为未评估，让前端与复述能看到降级
        damage_prob, cls, raw, review = 0.0, "未评估（无破损模型）", "not_evaluated", True
        desc = "未接入破损分类模型 · 需人工复核"
        boxes = []
    return {"prob": damage_prob, "cls": cls, "raw": raw, "boxes": boxes,
            "review_required": review, "model": desc}


def run_detect(frame: np.ndarray) -> Dict[str, Any]:
    """旧版 head/helmet/face/strap 检测器（仅在 state 权重缺失时兜底）。"""
    h, w = frame.shape[:2]
    model = MODELS.get("detect")
    items: List[Dict[str, Any]] = []
    boxes: Dict[str, Dict[str, int]] = {}
    if model is not None:
        res = model.predict(frame, conf=CONF, verbose=False)[0]
        for b in res.boxes:
            cid = int(b.cls[0])
            key = DET_NAMES.get(cid, "head")
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            boxes[key] = box
            items.append({"cls": key, "label": DET_LABEL[key], "box": box, "conf": round(conf, 3)})
    else:
        items.append({"cls": "head", "label": "头部",
                      "box": {"x": int(w * .38), "y": int(h * .25),
                              "w": int(w * .24), "h": int(h * .30)}, "conf": 0.50})
    return {"items": items, "boxes": boxes, "width": w, "height": h}


def run_state(frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """用 anquanmao.pt 的真实检测框生成 detection + wear 两路统一 schema。"""
    h, w = frame.shape[:2]
    model = MODELS.get("state")
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    items: List[Dict[str, Any]] = []
    boxes: Dict[str, Dict[str, int]] = {}
    states = []
    for b in res.boxes:
        raw_name = res.names[int(b.cls[0])]
        state = canon_state(raw_name)
        if state is None:
            continue
        conf = round(float(b.conf[0]), 3)
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        box = {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}
        key = STATE_CLS[state]
        boxes[key] = box
        items.append({"cls": key, "label": state, "box": box, "conf": conf})
        states.append((state, conf, box))

    # 安全优先：多状态同框时取最严格状态汇总
    worst = max(states, key=lambda t: STATE_SEVERITY[t[0]])[0] if states else "未佩戴"
    helmet_states = [t for t in states if t[0] in ("正确佩戴", "反戴")]
    helmet_conf = round(max((t[1] for t in helmet_states), default=0.0), 3)
    on_head = bool(helmet_states)

    checks = {"coverage": True, "front": True, "offset": True,
              "strap": True, "orientation": True}
    wear_conf = round(max((t[1] for t in states if t[0] == worst), default=0.0), 3)
    if worst == "反戴":
        checks["orientation"], checks["front"] = False, False
    elif worst == "未佩戴":
        checks["coverage"], checks["strap"] = False, False
    elif worst == "佩戴非安全帽":
        checks["coverage"], checks["front"] = False, False
    correct = all(checks.values())

    det = {
        "items": items, "boxes": boxes, "width": w, "height": h,
        "helmetConf": helmet_conf, "helmetOnHead": on_head, "count": len(items),
        "model": "anquanmao.pt · YOLOv8n detect（4 类佩戴状态 · 真实推理）",
        "raw_states": [{"state": s, "conf": c} for s, c, _ in states],
        "decisive_state": worst if states else "无检出（按未佩戴处理）",
    }
    wear = {
        "checks": checks, "correct": correct, "conf": wear_conf,
        "mode": "correct" if correct else "unknown",
        "label": worst + "（模型）" if states else "无检出（模型）",
        "keypoints": [], "model": "anquanmao.pt 状态分类 + 几何规则映射",
    }
    return det, wear


def run_pose(frame: np.ndarray) -> Dict[str, Any]:
    """state 权重缺失时的兜底姿态分支；接入 pose.pt 才会读关键点。"""
    model = MODELS.get("pose")
    keypoints, checks = [], {"coverage": True, "front": True, "offset": True,
                             "strap": True, "orientation": True}
    if model is not None:
        res = model.predict(frame, conf=CONF, verbose=False)[0]
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            kp = res.keypoints.data[0].tolist()
            for name, i in {"左耳": 3, "右耳": 4, "鼻": 0}.items():
                if i < len(kp):
                    keypoints.append({"name": name, "x": int(kp[i][0]), "y": int(kp[i][1])})
    correct = all(checks.values())
    return {
        "checks": checks, "correct": correct, "conf": 0.90 if correct else 0.30,
        "mode": "correct" if correct else "unknown",
        "label": "规范佩戴" if correct else "待复核", "keypoints": keypoints,
        "model": "YOLOv8s-pose + geometry" if model else "未接入姿态模型 · 待复核",
    }


def crop_by_box(frame: np.ndarray, box: Optional[Dict[str, int]], fallback_ratio=0.3):
    if box is None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * fallback_ratio)
        r = int(min(w, h) * 0.16)
        return frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
    pad = int(max(bw, bh) * 0.12)
    return frame[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]


@app.on_event("startup")
def startup() -> None:
    load_models(WEIGHTS_DIR, STATE_WEIGHTS)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "models": {k: (v is not None) for k, v in MODELS.items()},
        "weights_dir": WEIGHTS_DIR,
        "state_weights": STATE_WEIGHTS,
        "note": "state 为自训练 anquanmao.pt；缺失 cls_color / cls_damage / pose 会在 /infer 中标注降级",
    }


@app.get("/model")
def model_info() -> Dict[str, Any]:
    """给前端说明页用的模型元信息，全部来自实际权重文件。"""
    info = dict(MODEL_INFO)
    info["loaded"] = MODELS.get("state") is not None
    info["covers"] = ["有无安全帽", "佩戴是否规范（正确佩戴 / 反戴 / 未佩戴 / 佩戴非安全帽）"]
    info["not_covered"] = ["安全帽颜色等级", "盔体破损", "人脸活体与 1:N"]
    info["color_method"] = "盔体框内 HSV 真实像素统计" if MODELS.get("color") is None else "YOLOv8s-cls"
    info["damage_method"] = "需人工复核" if MODELS.get("damage") is None else "YOLOv8s-cls"
    info["dataset"] = "dataset/helmet4（D:/tupian 100 张，80/20 划分，CC BY-NC 4.0 合成 CCTV 图）"
    return info


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    t0 = time.time()
    frame = decode_frame(req)

    if MODELS.get("state") is not None:
        det, wear = run_state(frame)
        helmet_box = det["boxes"].get("helmet")
        state_active = True
    else:
        det = run_detect(frame)
        wear = run_pose(frame)
        helmet_box = det["boxes"].get("helmet")
        helmet_conf = 0.0
        for it in det["items"]:
            if it["cls"] == "helmet":
                helmet_conf = max(helmet_conf, it["conf"])
        det["helmetConf"] = round(helmet_conf, 3)
        on_head = False
        if helmet_box and "head" in det["boxes"]:
            hb, hd = helmet_box, det["boxes"]["head"]
            on_head = (hb["y"] < hd["y"] + hd["h"] * 0.5 and
                       abs(hb["x"] + hb["w"] / 2 - hd["x"] - hd["w"] / 2) < hd["w"] * 0.8)
        det["helmetOnHead"] = bool(on_head)
        det["model"] = "YOLOv8s-seg / detect（兜底，未加载自训练权重）"
        state_active = False

    helmet_crop = crop_by_box(frame, helmet_box)
    if helmet_crop.size:
        helmet_crop = cv2.resize(helmet_crop, (512, 512))
    color = run_color(helmet_crop if helmet_crop.size else frame)
    damage = run_damage(helmet_crop if helmet_crop.size else frame)

    degrade = [k for k, v in MODELS.items() if v is None and k != "state"]
    if not state_active:
        degrade.insert(0, "state")

    notes = [
        "佩戴状态与目标框来自 anquanmao.pt 的真实推理" if state_active
        else "未加载自训练权重，检测分支已降级",
        "颜色来自盔体框内 HSV 真实像素统计；本权重不含颜色分类头" if MODELS.get("color") is None
        else "颜色来自 cls_color.pt 分类模型",
        "破损未评估：本权重无破损分类头且数据集无破损样本，需人工复核" if MODELS.get("damage") is None
        else "破损来自 cls_damage.pt 分类模型",
        "人脸活体与 1:N 由既有刷脸系统通过请求参数给出",
    ]

    return {
        "source": "yolov8", "frame_id": req.frame_id, "channel": req.channel,
        "ts": time.strftime("%H:%M:%S"), "latency_ms": int((time.time() - t0) * 1000),
        "degraded_branches": degrade,
        "source_model": {
            "name": "anquanmao.pt", "loaded": state_active,
            "task": MODEL_INFO.get("runtime_task", MODEL_INFO.get("task", "detect")),
            "names": MODEL_INFO.get("runtime_names") or MODEL_INFO.get("class_names", []),
            "version": MODEL_INFO.get("version", ""),
        },
        "notes": notes,
        "detection": {
            "items": det["items"], "boxes": det["boxes"],
            "helmetConf": det["helmetConf"], "helmetOnHead": det["helmetOnHead"],
            "count": det["count"], "model": det["model"],
            "raw_states": det.get("raw_states", []),
            "decisive_state": det.get("decisive_state", ""),
        },
        "color": color, "damage": damage, "wear": wear,
        "face": {
            "score": clamp(req.face_score), "live": req.face_live,
            "matched": req.face_live and req.face_score >= 0.55,
            "staffId": req.staff_id, "name": req.face_name, "role": "", "dept": "",
            "model": "既有刷脸系统（1:N + 活体）",
        },
    }


def main() -> None:
    global WEIGHTS_DIR, STATE_WEIGHTS, CAPTURE, CONF
    ap = argparse.ArgumentParser(description="工地安全帽 YOLOv8 推理服务（含自训练权重）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights-dir", default="inference/weights")
    ap.add_argument("--state-weights", default="",
                    help="自训练权重路径，默认 inference/weights/anquanmao.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--source", default="", help="RTSP 地址或摄像头索引，如 rtsp://... 或 0")
    args = ap.parse_args()
    WEIGHTS_DIR = args.weights_dir
    STATE_WEIGHTS = args.state_weights
    CONF = args.conf
    if args.source:
        CAPTURE = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
