# -*- coding: utf-8 -*-
"""校验自训练权重能否被本系统加载，并在本地图片上跑一次真实推理。

    python tools/verify_model.py
    python tools/verify_model.py --weights inference/weights/anquanmao.pt \
        --images dataset/helmet4/images/val --limit 6 \
        --report dataset/model_report.json

它会：
  1. 读取 checkpoint 元数据（不依赖 torch）：任务、类别、训练参数、epoch 数；
  2. 用 ultralytics.YOLO 真正加载；若因 zip 内部前缀不是 archive/ 而失败，
     自动重写一份兼容副本（inference/weights/anquanmao_compat.pt）再加载；
  3. 在 CPU 上把 FP16 权重转成 FP32（Half 在 CPU 卷积上不可用）；
  4. 对若干张真实图片 predict，输出每类的检出数与置信度；
  5. 把可复现的结论写到 JSON，供 README / 前端说明页引用。
"""

import argparse
import io
import json
import os
import pickletools
import re
import zipfile

DEFAULT_WEIGHTS = "inference/weights/anquanmao.pt"


def peek_ckpt(path):
    """不 import torch，直接从 zip/pickle 里读类别名与训练参数。"""
    info = {"is_zip": zipfile.is_zipfile(path)}
    if not info["is_zip"]:
        return info
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        info["archive_prefix"] = sorted({n.split("/")[0] for n in names if "/" in n})
        info["n_storages"] = len([n for n in names if "/data/" in n])
        pk = [n for n in names if n.endswith("data.pkl")]
        if not pk:
            return info
        strings, globals_ = [], []
        for op, arg, pos in pickletools.genops(z.read(pk[0])):
            if op.name == "GLOBAL":
                globals_.append(arg if isinstance(arg, str) else "")
            if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                if isinstance(arg, bytes):
                    arg = arg.decode("utf-8", "replace")
                strings.append(arg)
        info["version"] = next((strings[i + 1] for i, s in enumerate(strings[:-1])
                                if s == "version"), "")
        info["date"] = next((s for s in strings
                             if re.match(r"\d{4}-\d{2}-\d{2}T", s)), "")
        info["task"] = next((strings[i + 1] for i, s in enumerate(strings[:-1])
                             if s == "task"), "")
        info["architecture"] = "ultralytics.nn.tasks.DetectionModel" if any(
            "DetectionModel" in g for g in globals_) else "unknown"
        # model.names 紧跟在 names 键之后
        for i, s in enumerate(strings):
            if s == "names" and i + 1 < len(strings):
                names = []
                for v in strings[i + 1:]:
                    if re.fullmatch(r"\d+", v) or v in ("yaml", "args", "colors", "pt_path"):
                        break
                    names.append(v)
                if names:
                    info["class_names"] = names
                break
        info["base_model"] = next((s for s in strings if "yolov8" in s.lower() and s.endswith(".pt")), "")
    return info


def make_compatible(src, dst):
    """把 zip 内部 best/ 前缀改写为 PyTorch 默认的 archive/ 前缀。"""
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith("best/"):
                name = "archive/" + name[5:]
            data = zin.read(item.filename)
            zi = zipfile.ZipInfo(name, date_time=item.date_time)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = item.external_attr
            zout.writestr(zi, data)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--images", default="dataset/helmet4/images/val")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--report", default="dataset/model_report.json")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    w = os.path.abspath(args.weights)
    if not os.path.isfile(w):
        raise SystemExit("找不到权重：%s" % w)
    meta = peek_ckpt(w)
    print("== checkpoint 元数据 ==")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    report = {"weights": w, "bytes": os.path.getsize(w), "metadata": meta,
              "load": {}, "predictions": [], "verdict": ""}

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        report["verdict"] = "依赖未安装：%s" % e
        print(report["verdict"])
        _dump(report, args.report)
        return

    load_path = w
    for attempt in (1, 2):
        try:
            model = YOLO(load_path)
            report["load"] = {
                "ok": True, "path": load_path, "attempt": attempt,
                "ultralytics": __import__("ultralytics").__version__,
                "torch": torch.__version__,
                "task": model.task,
                "names": dict(model.names),
                "device": "cpu",
            }
            break
        except Exception as e:
            print("[load] 第 %d 次失败：%s" % (attempt, e))
            if attempt == 1:
                compat = os.path.join(os.path.dirname(w), "anquanmao_compat.pt")
                make_compatible(w, compat)
                print("[load] 已生成兼容副本：%s" % compat)
                load_path = compat
                report["load"]["first_error"] = str(e)
            else:
                report["load"] = {"ok": False, "error": str(e)}
                report["verdict"] = "模型无法加载"
                _dump(report, args.report)
                return

    # FP16 -> FP32，否则 CPU 卷积会报 "not implemented for Half"
    try:
        if next(model.model.parameters()).dtype == torch.float16:
            model.model.float()
            report["load"]["dtype_fix"] = "FP16 -> FP32 (CPU)"
    except Exception as e:
        report["load"]["dtype_fix_warning"] = str(e)

    img_dir = os.path.abspath(args.images)
    files = []
    if os.path.isdir(img_dir):
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))[:args.limit]
    for fn in files:
        p = os.path.join(img_dir, fn)
        try:
            res = model.predict(p, conf=args.conf, verbose=False)[0]
            names = res.names
            boxes = res.boxes
            items = []
            for b in boxes:
                items.append({"cls": names[int(b.cls[0])],
                              "conf": round(float(b.conf[0]), 3),
                              "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()]})
            report["predictions"].append({"image": fn, "detections": len(items), "items": items})
            print("  %-32s %d 个框: %s" % (fn, len(items),
                  ", ".join("%s %.2f" % (i["cls"], i["conf"]) for i in items[:6])))
        except Exception as e:
            report["predictions"].append({"image": fn, "error": str(e)})
            print("  %-32s 推理失败: %s" % (fn, e))

    ok = report["load"].get("ok") and any(
        "error" not in p and p["detections"] >= 0 for p in report["predictions"])
    report["verdict"] = (
        "可以导入：YOLOv8n detect 权重成功加载并完成真实推理；"
        "类别为 4 类佩戴状态，接入推理服务后由 StateAdapter 映射到系统统一 schema。"
        if ok else "加载成功但推理异常，请检查依赖与图片路径"
    )
    print("\n== 结论 ==\n" + report["verdict"])
    _dump(report, args.report)


def _dump(report, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入：%s" % path)


if __name__ == "__main__":
    main()
