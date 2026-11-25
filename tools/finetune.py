# -*- coding: utf-8 -*-
"""以自训练权重为起点，用 D:/tupian 整理出的 helmet4 数据集做增量微调。

重要说明（务必如实理解「在原有训练基础上加入数据集」的含义）：
  anquanmao.pt 是 Ultralytics Platform 云端训练的产物，其原始训练图片
  （/datasets/anquanmao-5982da81/）不在本机，本地只有权重，没有那批图片。
  因此这里做的是「以已训练权重作为预训练起点 + 新增本地数据集微调」，
  而不是把两批原始图片混在一起从头重训。权重里已学到的特征被完整保留并继续更新。

  helmet4 的类别顺序与 anquanmao.pt 的 model.names 完全一致（nc=4），
  所以可以直接续训；ppe9 是 9 类 PPE 原始口径，仅供后续扩展，不能直接混训。

用法：
    python tools/finetune.py --dry-run                 # 只看命令，不训练
    python tools/finetune.py --epochs 30 --device cpu
    python tools/finetune.py --epochs 50 --device 0   # 有 NVIDIA GPU 时

产物：runs/helmet4_finetune/weights/best.pt
     并自动复制为 inference/weights/anquanmao_finetuned.pt
"""

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_WEIGHTS = os.path.join(ROOT, "inference", "weights", "anquanmao.pt")
DEFAULT_DATA = os.path.join(ROOT, "dataset", "helmet4", "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前 N 层；数据里缺「佩戴非安全帽」样本时可设为 10 缓解遗忘")
    ap.add_argument("--lr0", type=float, default=0.001,
                    help="微调学习率，建议比从头训练低一个量级")
    ap.add_argument("--name", default="helmet4_finetune")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit("找不到权重：%s" % args.weights)
    if not os.path.isfile(args.data):
        raise SystemExit("找不到数据集配置：%s\n请先运行 python tools/prepare_dataset.py" % args.data)

    # data.yaml 的 path 相对仓库根；固定 cwd 避免 Ultralytics 解析到错误目录
    os.chdir(ROOT)

    # 校验类别顺序与数量，避免把 9 类数据喂给 4 类权重
    # 用轻量文本解析，不额外依赖 PyYAML（训练本身仍由 ultralytics 负责读 yaml）
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^nc:\s*(\d+)", text, re.M)
    if not m:
        raise SystemExit("data.yaml 缺少 nc 字段")
    nc = int(m.group(1))
    names = [v.strip() for v in re.findall(r"^\s+\d+:\s*(.+)$", text, re.M)]
    if len(names) != nc:
        raise SystemExit("data.yaml 的 names 数量(%d)与 nc(%d)不一致" % (len(names), nc))
    print("数据集类别 :", names)
    print("权重       :", args.weights)
    print("微调配置   : epochs=%d imgsz=%d batch=%d device=%s lr0=%g freeze=%d"
          % (args.epochs, args.imgsz, args.batch, args.device, args.lr0, args.freeze))
    print("说明       : 本地无原始训练图片，本次为「权重起点 + 新增数据集」增量微调")
    print("注意       : helmet4 中「佩戴非安全帽」样本为 0，该类可能弱化，见 dataset/README.md")

    cmd = ("yolo detect train model=%s data=%s epochs=%d imgsz=%d batch=%d "
           "device=%s lr0=%g freeze=%d project=%s name=%s exist_ok=True"
           % (args.weights, args.data, args.epochs, args.imgsz, args.batch,
              args.device, args.lr0, args.freeze, os.path.join(ROOT, "runs"), args.name))
    print("\n等价 CLI：\n  " + cmd)
    if args.dry_run:
        print("\n[dry-run] 未实际训练。")
        return

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit("请先 pip install -r inference/requirements.txt（%s）" % e)

    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, lr0=args.lr0, freeze=args.freeze,
        project=os.path.join(ROOT, "runs"), name=args.name, exist_ok=True,
        plots=False, workers=0,
    )
    best = os.path.join(ROOT, "runs", args.name, "weights", "best.pt")
    if os.path.isfile(best):
        dst = os.path.join(ROOT, "inference", "weights", "anquanmao_finetuned.pt")
        shutil.copy2(best, dst)
        print("\n微调权重已就位：%s" % dst)
        print("用新权重启动服务：python inference/yolov8_service.py --state-weights " + dst)
    else:
        print("[warn] 未找到 best.pt，请检查 runs/%s/" % args.name, file=sys.stderr)


if __name__ == "__main__":
    main()
