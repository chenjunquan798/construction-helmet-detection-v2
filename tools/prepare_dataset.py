# -*- coding: utf-8 -*-
"""把 D:/tupian 的 100 张 PPE 图片整理成两套可直接训练的 YOLO 数据集。

用法：
    python tools/prepare_dataset.py                       # 默认读 D:/tupian
    python tools/prepare_dataset.py --source D:/tupian --out dataset --seed 42

产出：
    dataset/ppe9/    原始 9 类 PPE 标注（Simuletic 口径，保留全部框）
        images/train|val/*.jpg    labels/train|val/*.txt    data.yaml
    dataset/helmet4/ 与自训练权重 anquanmao.pt 对齐的 4 类「佩戴状态」标注
        images/train|val/*.jpg    labels/train|val/*.txt    data.yaml
    dataset/remap_report.json    映射统计与一致性检查（可复现的证据）

为什么要重映射：
    anquanmao.pt 的类别是 正确佩戴 / 反戴 / 未佩戴 / 佩戴非安全帽（nc=4），
    而 tupian 是 9 类 PPE 多目标标注。两者类别体系不同，直接混训会把类别打乱。

为什么必须重组目录：
    原始 tupian 把 JPG 放在根目录、txt 放在 labels/ 子目录，且 data.yaml 里
    train/val 都写 "."。Ultralytics 依据 images/ -> labels/ 的路径替换规则找标签，
    这种布局会导致「找不到任何标签」。重组后才是标准 YOLO 结构。
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict

# tupian/data.yaml 原始 9 类
PPE9_NAMES = [
    "person", "hardhat", "hardhat_wrong", "no_hardhat",
    "vest", "vest_wrong", "no_vest", "gloves", "no_gloves",
]

# anquanmao.pt 训练时的 4 个类别（从 checkpoint 内 model.names 读出）
HELMET4_NAMES = ["正确佩戴", "反戴", "未佩戴", "佩戴非安全帽"]

# 9 类 -> 4 类 的框级映射；未列出的类别与安全帽状态无关，直接丢弃
PPE_TO_STATE = {
    1: 0,  # hardhat        -> 正确佩戴
    2: 1,  # hardhat_wrong  -> 反戴
    3: 2,  # no_hardhat     -> 未佩戴（人头框，语义是「这个人没戴帽」）
}
# 「佩戴非安全帽」(3) 在 tupian 中没有对应标注。
# 说明文档提到 no_helmet 类里含棒球帽/针织帽代替，但其标注统一为 no_hardhat，
# 无法可靠区分，因此不伪造该类样本，写入报告并在 README 中显式声明。

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def category_of(name):
    """从文件名前缀判断图级类别，仅用于分层划分与一致性核对。"""
    n = name.lower()
    for key in ("correct_helmet", "wrong_helmet", "no_helmet"):
        if n.startswith(key):
            return key
    return "other"


def read_label(path):
    """读 YOLO txt，返回 [(cls, [5 个浮点]), ...]，并校验格式。"""
    rows, bad = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
                vals = [float(v) for v in parts[1:5]]
            except (ValueError, IndexError):
                bad.append((ln, line))
                continue
            if len(vals) != 4 or cls < 0 or cls >= len(PPE9_NAMES):
                bad.append((ln, line))
                continue
            if any(v < 0 or v > 1.0001 for v in vals):
                bad.append((ln, line))
                continue
            rows.append((cls, vals))
    return rows, bad


def write_yaml(path, names, dataset_path, extra_comment=""):
    # Ultralytics 的 check_det_dataset 会把相对 path 相对「当前工作目录」解析，
    # 因此这里写相对仓库根的路径；所有训练 / 评估命令都从仓库根执行。
    lines = ["# 由 tools/prepare_dataset.py 生成，请勿手工改动路径部分",
             "# path 相对仓库根目录，命令请在仓库根执行（脚本会自动 chdir）"]
    if extra_comment:
        lines += ["# " + c for c in extra_comment.strip().splitlines()]
    lines += [
        "path: " + dataset_path,
        "train: images/train",
        "val: images/val",
        "nc: %d" % len(names),
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append("  %d: %s" % (i, n))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="D:/tupian", help="原始 tupian 目录")
    ap.add_argument("--out", default="dataset", help="输出根目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    out = os.path.abspath(args.out)
    if not os.path.isdir(src):
        raise SystemExit("找不到源目录：%s" % src)

    images = sorted(
        f for f in os.listdir(src)
        if os.path.splitext(f)[1].lower() in IMAGE_EXT
        and os.path.isfile(os.path.join(src, f))
    )
    if not images:
        raise SystemExit("源目录没有图片：%s" % src)
    label_dir = os.path.join(src, "labels")

    # 分层划分：保证 train/val 三类比例一致，结果由 seed 固定可复现
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for f in images:
        by_cat[category_of(f)].append(f)
    split = {}
    for cat, files in by_cat.items():
        files = sorted(files)
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * args.val_ratio))) if len(files) >= 5 else 0
        for f in files:
            split[f] = "val" if f in set(files[:n_val]) else "train"

    ppe9 = os.path.join(out, "ppe9")
    h4 = os.path.join(out, "helmet4")
    for root in (ppe9, h4):
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)

    rep = {
        "source": src,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "image_count": len(images),
        "split": {"train": 0, "val": 0},
        "by_category": dict(Counter(split.values())),
        "category_split": {},
        "ppe9": {"boxes": Counter(), "images": {"train": 0, "val": 0}},
        "helmet4": {"boxes": Counter(), "images": {"train": 0, "val": 0},
                    "dropped_boxes": 0, "empty_images": []},
        "consistency": {"checked": 0, "mismatch": []},
        "bad_label_lines": [],
        "warnings": [],
    }
    for cat, files in by_cat.items():
        rep["category_split"][cat] = {
            "total": len(files),
            "train": sum(1 for f in files if split[f] == "train"),
            "val": sum(1 for f in files if split[f] == "val"),
        }

    for f in images:
        part = split[f]
        rep["split"][part] += 1
        stem = os.path.splitext(f)[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        rows, bad = ([], []) if not os.path.isfile(label_path) else read_label(label_path)
        for ln, line in bad:
            rep["bad_label_lines"].append({"file": stem + ".txt", "line": ln, "text": line})

        # --- ppe9：原样保留 9 类 ---
        shutil.copy2(os.path.join(src, f), os.path.join(ppe9, "images", part, f))
        with open(os.path.join(ppe9, "labels", part, stem + ".txt"), "w", encoding="utf-8") as g:
            for cls, vals in rows:
                rep["ppe9"]["boxes"][PPE9_NAMES[cls]] += 1
                g.write("%d %.6f %.6f %.6f %.6f\n" % (cls, *vals))
        rep["ppe9"]["images"][part] += 1

        # --- helmet4：框级重映射到 4 类 ---
        shutil.copy2(os.path.join(src, f), os.path.join(h4, "images", part, f))
        mapped = []
        for cls, vals in rows:
            if cls in PPE_TO_STATE:
                mapped.append((PPE_TO_STATE[cls], vals))
            else:
                rep["helmet4"]["dropped_boxes"] += 1
        with open(os.path.join(h4, "labels", part, stem + ".txt"), "w", encoding="utf-8") as g:
            for cls, vals in mapped:
                rep["helmet4"]["boxes"][HELMET4_NAMES[cls]] += 1
                g.write("%d %.6f %.6f %.6f %.6f\n" % (cls, *vals))
        rep["helmet4"]["images"][part] += 1
        if not mapped:
            rep["helmet4"]["empty_images"].append(f)

        # 一致性核对：图级前缀 vs 框级映射结果
        cat = category_of(f)
        state_ids = set(c for c, _ in mapped)
        expect = {"correct_helmet": {0}, "wrong_helmet": {1}, "no_helmet": {2}}.get(cat)
        if expect is not None:
            rep["consistency"]["checked"] += 1
            if not state_ids & expect:
                rep["consistency"]["mismatch"].append({
                    "file": f, "category": cat, "mapped_states": sorted(state_ids),
                })

    # 序列化 Counter
    for key in ("ppe9", "helmet4"):
        rep[key]["boxes"] = dict(rep[key]["boxes"])

    if not rep["helmet4"]["boxes"].get("佩戴非安全帽"):
        rep["warnings"].append(
            "helmet4 中「佩戴非安全帽」样本为 0：tupian 无对应标注，未伪造。"
            "微调会削弱该类，请补充真实样本或冻结分类头。"
        )
    if rep["helmet4"]["empty_images"]:
        rep["warnings"].append(
            "%d 张图在 4 类映射后无框（仅含 person/vest/gloves 等），已保留为空标签"
            % len(rep["helmet4"]["empty_images"])
        )
    if rep["bad_label_lines"]:
        rep["warnings"].append("发现 %d 行不合法标注，已跳过" % len(rep["bad_label_lines"]))

    write_yaml(os.path.join(ppe9, "data.yaml"), PPE9_NAMES, "dataset/ppe9",
               "原始 9 类 PPE 标注（Simuletic CCTV 样本），来源与许可见 dataset/README.md")
    write_yaml(os.path.join(h4, "data.yaml"), HELMET4_NAMES, "dataset/helmet4",
               "与 inference/weights/anquanmao.pt 的 model.names 对齐\n"
               "tools/finetune.py 使用本文件做增量微调")

    # 保留溯源材料
    for fn in ("说明文档.txt", "dataset_manifest.csv", "data.yaml"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, fn))
    with open(os.path.join(out, "remap_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("源目录      :", src)
    print("图片        :", len(images), " train/val =", rep["split"])
    print("ppe9  框统计:", rep["ppe9"]["boxes"])
    print("helmet4 框统计:", rep["helmet4"]["boxes"])
    print("丢弃无关框  :", rep["helmet4"]["dropped_boxes"])
    print("一致性异常  :", len(rep["consistency"]["mismatch"]))
    for w in rep["warnings"]:
        print("[warn]", w)
    print("输出        :", out)


if __name__ == "__main__":
    main()
