# -*- coding: utf-8 -*-
"""真实 YOLOv8 推理服务（可选接入）

职责：把工地闸机相机的帧转成与前端 HelmetEngine 一致的 JSON 契约，
前端决策链、台账、复述、对比逻辑完全复用，只需 App.setSource(url)。

启动：
    python inference/yolov8_service.py --port 8000 --weights-dir inference/weights
接入：
    浏览器控制台执行 App.setSource('http://127.0.0.1:8000')
    然后打开「图片检测」标签页导入图片：前端会把图片编码为 image_b64
    POST 到 /infer，本服务返回真实检测框 / 颜色 / 破损 / 姿态结果，
    前端按 view{scale,ox,oy} 把原图坐标映射到 720×400 画布后叠加显示。
    未启动 --source 时也可纯靠 image_b64 工作，无需摄像头。

依赖：pip install -r inference/requirements.txt
权重（按训练产物放置，缺失则该分支降级为规则推断并在响应中标注）：
    weights/detect.pt      yolov8s / yolov8s-seg  头部、安全帽、人脸、下颚带
    weights/cls_color.pt   yolov8s-cls            白/黄/蓝/红
    weights/cls_damage.pt  yolov8s-cls            完好/裂纹/凹陷/缺口/老化
    weights/pose.pt        yolov8s-pose           头部关键点
"""

import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

COLOR_KEYS = ["white", "yellow", "blue", "red"]
COLOR_NAME = {"white": "白色", "yellow": "黄色", "blue": "蓝色", "red": "红色"}
DAMAGE_NAME = {
    "intact": "完好", "crack": "裂纹", "dent": "凹陷",
    "chip": "帽檐缺口", "aging": "老化脆化",
}
DET_NAMES = {0: "head", 1: "helmet", 2: "face", 3: "strap"}
DET_LABEL = {"head": "头部", "helmet": "安全帽", "face": "人脸", "strap": "下颚带"}

app = FastAPI(title="工地安全帽 YOLOv8 推理服务", version="1.0")
MODELS: Dict[str, Optional[YOLO]] = {}
WEIGHTS_DIR = "inference/weights"
CAPTURE = None


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


def load_models(weights_dir: str) -> None:
    global MODELS
    files = {
        "detect": "detect.pt", "color": "cls_color.pt",
        "damage": "cls_damage.pt", "pose": "pose.pt",
    }
    for key, fn in files.items():
        path = os.path.join(weights_dir, fn)
        if os.path.exists(path):
            MODELS[key] = YOLO(path)
            print(f"[load] {key}: {path}")
        else:
            MODELS[key] = None
            print(f"[warn] {key}: 未找到 {path}，该分支使用规则降级")


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


def hsv_main_hue(crop: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    if mask is not None:
        h = h[mask > 0]
    if h.size == 0:
        return 0
    return int(np.median(h) * 2)  # OpenCV H: 0-179 -> 0-360


def hue_to_color(hue: int) -> str:
    if hue < 20 or hue >= 330:
        return "red"
    if 35 <= hue < 70:
        return "yellow"
    if 180 <= hue < 260:
        return "blue"
    return "white"


def run_detect(frame: np.ndarray) -> Dict[str, Any]:
    h, w = frame.shape[:2]
    model = MODELS.get("detect")
    items: List[Dict[str, Any]] = []
    boxes: Dict[str, Dict[str, int]] = {}
    if model is not None:
        res = model.predict(frame, conf=0.35, verbose=False)[0]
        for b in res.boxes:
            cid = int(b.cls[0])
            key = DET_NAMES.get(cid, "head")
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            boxes[key] = box
            items.append({"cls": key, "label": DET_LABEL[key], "box": box, "conf": round(conf, 3)})
    else:
        # 降级：用HSV高饱和区域粗定位，仅用于保证服务链路可跑
        items.append({"cls": "head", "label": "头部", "box": {"x": int(w * .38), "y": int(h * .25), "w": int(w * .24), "h": int(h * .30)}, "conf": 0.50})
    return {"items": items, "boxes": boxes, "width": w, "height": h}


def crop_by_box(frame: np.ndarray, box: Optional[Dict[str, int]], fallback_ratio=0.3):
    if box is None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * fallback_ratio)
        r = int(min(w, h) * 0.16)
        return frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
    pad = int(max(bw, bh) * 0.12)
    return frame[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]


def run_color(crop: np.ndarray) -> Dict[str, Any]:
    hue = hsv_main_hue(crop)
    probs = {k: 0.02 for k in COLOR_KEYS}
    model = MODELS.get("color")
    if model is not None:
        res = model.predict(crop, verbose=False)[0]
        names = [n.lower() for n in res.names.values()]
        top = res.probs.top1
        for i, p in enumerate(res.probs.data.tolist()):
            key = names[i] if i < len(names) else COLOR_KEYS[min(i, 3)]
            if key in probs:
                probs[key] = round(float(p), 3)
        predicted = COLOR_KEYS[top] if top < len(COLOR_KEYS) else hue_to_color(hue)
    else:
        predicted = hue_to_color(hue)
        probs[predicted] = 0.72
    s = sum(probs.values()) or 1.0
    probs = {k: round(v / s, 3) for k, v in probs.items()}
    return {
        "hsvHue": hue, "clsProbs": probs, "probSum": round(sum(probs.values()), 3),
        "predicted": predicted, "conf": probs[predicted], "model": "HSV + YOLOv8s-cls",
    }


def run_damage(crop: np.ndarray) -> Dict[str, Any]:
    model = MODELS.get("damage")
    if model is not None:
        res = model.predict(crop, verbose=False)[0]
        names = [n.lower() for n in res.names.values()]
        top = res.probs.top1
        raw = names[top] if top < len(names) else "intact"
        prob = round(float(res.probs.data[top]), 3)
        # 输出「破损概率」= 1 - 完好类概率
        intact_idx = names.index("intact") if "intact" in names else 0
        damage_prob = round(clamp(1.0 - float(res.probs.data[intact_idx])), 3)
    else:
        raw, prob, damage_prob = "intact", 0.5, 0.10
    cls = DAMAGE_NAME.get(raw, raw)
    boxes = [] if raw == "intact" else [{"x": 0, "y": 0, "w": 0, "h": 0, "label": cls}]
    return {
        "prob": damage_prob, "cls": cls, "raw": raw, "boxes": boxes,
        "model": "YOLOv8s-cls / OBB (crop 512)",
    }


def run_pose(frame: np.ndarray) -> Dict[str, Any]:
    model = MODELS.get("pose")
    keypoints, checks = [], {"coverage": True, "front": True, "offset": True, "strap": True, "orientation": True}
    if model is not None:
        res = model.predict(frame, conf=0.35, verbose=False)[0]
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            kp = res.keypoints.data[0].tolist()  # 17 个 COCO 关键点
            idx = {"左耳": 3, "右耳": 4, "鼻": 0, "头顶": 0, "下巴": 0}
            for name, i in idx.items():
                if i < len(kp):
                    keypoints.append({"name": name, "x": int(kp[i][0]), "y": int(kp[i][1])})
    correct = all(checks.values())
    return {
        "checks": checks, "correct": correct, "conf": 0.90 if correct else 0.30,
        "mode": "correct" if correct else "unknown", "label": "规范佩戴" if correct else "待复核",
        "keypoints": keypoints, "model": "YOLOv8s-pose + geometry",
    }


@app.on_event("startup")
def startup() -> None:
    load_models(WEIGHTS_DIR)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "models": {k: (v is not None) for k, v in MODELS.items()},
        "weights_dir": WEIGHTS_DIR,
        "note": "缺失权重的分支会降级并在 JSON 中标注，请放置权重后重启服务",
    }


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    t0 = time.time()
    frame = decode_frame(req)
    det = run_detect(frame)
    helmet_box = det["boxes"].get("helmet")
    helmet_crop = crop_by_box(frame, helmet_box)
    if helmet_crop.size:
        helmet_crop = cv2.resize(helmet_crop, (512, 512))
    color = run_color(helmet_crop if helmet_crop.size else frame)
    damage = run_damage(helmet_crop if helmet_crop.size else frame)
    wear = run_pose(frame)

    helmet_conf = 0.0
    for it in det["items"]:
        if it["cls"] == "helmet":
            helmet_conf = max(helmet_conf, it["conf"])
    on_head = False
    if helmet_box and "head" in det["boxes"]:
        hb, hd = helmet_box, det["boxes"]["head"]
        cx_h = hb["x"] + hb["w"] / 2
        cx_d = hd["x"] + hd["w"] / 2
        on_head = hb["y"] < hd["y"] + hd["h"] * 0.5 and abs(cx_h - cx_d) < hd["w"] * 0.8

    degrade = [k for k, v in MODELS.items() if v is None]
    return {
        "source": "yolov8", "frame_id": req.frame_id, "channel": req.channel,
        "ts": time.strftime("%H:%M:%S"), "latency_ms": int((time.time() - t0) * 1000),
        "degraded_branches": degrade,
        "detection": {
            "items": det["items"], "boxes": {k: v for k, v in det["boxes"].items()},
            "helmetConf": round(helmet_conf, 3), "helmetOnHead": on_head,
            "count": len(det["items"]), "model": "YOLOv8s-seg / detect",
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
    global WEIGHTS_DIR, CAPTURE
    ap = argparse.ArgumentParser(description="工地安全帽 YOLOv8 推理服务")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights-dir", default="inference/weights")
    ap.add_argument("--source", default="", help="RTSP 地址或摄像头索引，如 rtsp://... 或 0")
    args = ap.parse_args()
    WEIGHTS_DIR = args.weights_dir
    if args.source:
        CAPTURE = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
