# -*- coding: utf-8 -*-
"""在 dataset/helmet4 的 val 划分上评估自训练权重，输出真实指标。

    python tools/evaluate.py
    python tools/evaluate.py --weights inference/weights/anquanmao_finetuned.pt \
        --out dataset/eval_finetuned.json

产出 JSON 含：每类 precision/recall/mAP50/mAP50-95、整体指标、逐图检出明细。
微调前后各跑一次即可得到 A/B 对比；数字来自 ultralytics model.val()，不是仿真。
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "inference", "weights", "anquanmao.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "helmet4", "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "eval_report.json"))
    ap.add_argument("--no-val", action="store_true", help="只做逐图推理统计，跳过 model.val()")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit("找不到权重：%s" % args.weights)
    if not os.path.isfile(args.data):
        raise SystemExit("找不到数据集，请先运行 python tools/prepare_dataset.py")

    # data.yaml 的 path 相对仓库根；固定 cwd 避免 Ultralytics 解析到错误目录
    os.chdir(ROOT)
    from ultralytics import YOLO
    import torch

    model = YOLO(args.weights)
    try:
        if next(model.model.parameters()).dtype == torch.float16:
            model.model.float()
    except Exception:
        pass

    report = {"weights": args.weights, "data": args.data,
              "conf": args.conf, "imgsz": args.imgsz, "device": args.device,
              "names": dict(model.names)}

    if not args.no_val:
        # Windows 上多进程 dataloader 容易卡住，显式 workers=0 / 小 batch
        metrics = model.val(data=args.data, split="val", imgsz=args.imgsz,
                            conf=args.conf, device=args.device, verbose=False,
                            workers=0, batch=1, plots=False)
        try:
            mp, mr, map50, map95 = metrics.mean_results()
            report["overall"] = {
                "precision": round(float(mp), 4), "recall": round(float(mr), 4),
                "mAP50": round(float(map50), 4), "mAP50-95": round(float(map95), 4),
            }
            report["per_class"] = {}
            for i, name in model.names.items():
                if i < len(metrics.box.ap50):
                    report["per_class"][name] = {
                        "mAP50": round(float(metrics.box.ap50[i]), 4),
                        "mAP50-95": round(float(metrics.box.ap[i]), 4),
                    }
        except Exception as e:
            report["overall_error"] = str(e)
        print("整体指标：", report.get("overall"))

    # 逐图推理明细，作为「跑得通」的直接证据
    img_dir = os.path.join(ROOT, "dataset", "helmet4", "images", "val")
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    details = []
    for fn in files:
        p = os.path.join(img_dir, fn)
        res = model.predict(p, conf=args.conf, verbose=False, device=args.device)[0]
        items = [{"cls": res.names[int(b.cls[0])], "conf": round(float(b.conf[0]), 3)}
                 for b in res.boxes]
        # 图级真值来自文件名前缀，仅用于展示对照，不参与 mAP 计算
        prefix = next((k for k in ("correct_helmet", "wrong_helmet", "no_helmet")
                       if fn.startswith(k)), "")
        details.append({"image": fn, "filename_category": prefix,
                        "detections": len(items), "items": items})
    report["per_image"] = details
    report["image_count"] = len(details)
    report["total_detections"] = sum(d["detections"] for d in details)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("评估图片：%d 张，检出框：%d 个" % (len(details), report["total_detections"]))
    print("报告：", args.out)


if __name__ == "__main__":
    main()
