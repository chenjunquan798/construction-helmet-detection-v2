# 本地数据集：D:/tupian → 标准 YOLO 结构

本目录由 `tools/prepare_dataset.py` 从本机 `D:/tupian` 生成，用于给自训练权重
`inference/weights/anquanmao.pt` 做增量微调与真实评估。生成过程可重复、有报告：
`dataset/remap_report.json`。

## 1. 先把边界说清楚（重要）

| 项目 | 事实 |
| --- | --- |
| 图片数量 | 100 张 JPG，约 36 MB，1672 × 941 |
| 图像性质 | **合成（synthetic）CCTV 监控视角图，不是真人真实现场照片** |
| 场景 | 建筑工地 / 仓库物流 / 工厂车间，含白天与夜间红外 |
| 来源 | Simuletic CCTV PPE Compliance Detection Dataset 免费样本（200 张中筛选 100 张） |
| 项目地址 | https://github.com/Simuletic/cctv-ppe-compliance-detection-dataset |
| 许可 | **CC BY-NC 4.0**：允许非商业用途分享与改编，**须署名 Simuletic**；商业项目需向原作者取得商业许可 |
| 破损样本 | **0 张**。公开样本中没有「安全帽明显破损仍佩戴」类别，系统里该分支仍需人工复核或另行采集 |

因此，这批数据让系统「更真实」的含义是：**训练/评估链路跑在真实图片与真实标注上、
指标来自真实推理**，而不是把仿真数字换成随机噪声。它并不能替代真实现场采集。
100 张的样本量也只够做格式验证、流程调试与小样本微调，不足以训练可部署检测器
（这一点原始说明文档亦明确写出）。

## 2. 为什么要重组目录

原始 `D:/tupian` 的布局是：JPG 在根目录、TXT 在 `labels/` 子目录，
`data.yaml` 里 `train: .` / `val: .`。Ultralytics 通过 `images/ → labels/` 的
路径替换规则定位标签，这种布局会**找不到任何标签**，且 train/val 是同一批数据，
评估会泄漏。重组后：

```
dataset/
├── ppe9/                    # 原始 9 类 PPE 口径（保留全部框，供后续扩展）
│   ├── images/train|val/*.jpg
│   ├── labels/train|val/*.txt
│   └── data.yaml            # nc=9, person/hardhat/.../no_gloves
├── helmet4/                 # 与 anquanmao.pt 对齐的 4 类「佩戴状态」
│   ├── images/train|val/*.jpg
│   ├── labels/train|val/*.txt
│   └── data.yaml            # nc=4, 正确佩戴/反戴/未佩戴/佩戴非安全帽
├── remap_report.json        # 划分、框统计、一致性检查、警告
├── 说明文档.txt              # 原始说明（已随仓库保留）
├── dataset_manifest.csv     # 原始文件名溯源表
└── data.yaml                # 原始 9 类配置（存档）
```

划分：按 `correct_helmet / wrong_helmet / no_helmet` 图级类别**分层**取 20% 作 val，
`seed=42` 固定可复现 → **train 80 张 / val 20 张**。

## 3. 类别映射（9 类 → 4 类）

`anquanmao.pt` 的 `model.names` 是 `正确佩戴 / 反戴 / 未佩戴 / 佩戴非安全帽`，
与 tupian 的 9 类 PPE 口径不同，直接混训会打乱类别。脚本按**框级**做映射：

| tupian 原始类 | 映射到 | 说明 |
| --- | --- | --- |
| `1 hardhat` | `0 正确佩戴` | 规范佩戴的盔体框 |
| `2 hardhat_wrong` | `1 反戴` | 不规范 / 反戴 |
| `3 no_hardhat` | `2 未佩戴` | 人头框，语义是「此人未戴帽」 |
| `4 佩戴非安全帽` | — | **无对应标注，不伪造样本** |
| `0 person`、`4 vest`、`5 vest_wrong`、`6 no_vest`、`7 gloves`、`8 no_gloves` | 丢弃 | 与安全帽状态无关 |

实际统计（`remap_report.json`）：

- `ppe9`：person 225、hardhat 180、hardhat_wrong 30、no_hardhat 27、vest 198、
  vest_wrong 12、no_vest 15、gloves 232、no_gloves 135 个框
- `helmet4`：**正确佩戴 180、反戴 30、未佩戴 27** 个框，丢弃无关框 817 个
- 图级前缀与框级映射的一致性检查：**100 张全部核对，异常 0 张**

⚠️ `helmet4` 中第 3 类「佩戴非安全帽」样本为 **0**。以该数据集微调会弱化这个类，
务必补充真实样本，或在 `tools/finetune.py` 里用 `--freeze 10` 冻结部分层、
降低 `--lr0` 缓解遗忘。这一点在报告 `warnings` 里也写明，**不掩盖**。

## 4. 复现命令

```bash
# 1) 从 D:/tupian 重新生成数据集（可重复执行）
python tools/prepare_dataset.py --source D:/tupian --out dataset --seed 42

# 2) 校验自训练权重能否加载 + 真实推理一张图
python tools/verify_model.py

# 3) 评估（微调前），输出 dataset/eval_report.json
python tools/evaluate.py

# 4) 以 anquanmao.pt 为起点做增量微调（先看命令可加 --dry-run）
python tools/finetune.py --epochs 30 --device cpu

# 5) 评估微调后权重，与第 3 步做 A/B
python tools/evaluate.py --weights inference/weights/anquanmao_finetuned.pt \
    --out dataset/eval_finetuned.json
```

**关于「在原有训练基础上加入数据集」**：`anquanmao.pt` 来自 Ultralytics Platform
云端训练（`train_args.data = /datasets/anquanmao-5982da81/data.yaml`），
**那批原始训练图片不在本机**，本地只有权重。因此这里做的是
「以已训练权重作为预训练起点 + 新增本地数据集微调」，权重已学到的特征被完整保留
并继续更新；不是把两批原始图片混在一起从头重训。若后续拿到原训练集，
把两者的 `images/` 合并后再跑同一脚本即可。

## 5. 署名

数据集来源：Simuletic — *CCTV PPE Compliance Detection Dataset*（免费样本），
CC BY-NC 4.0，非商业用途，保留署名。本仓库仅收录其 100 张样本用于流程验证与
教学演示；如用于商业部署，请替换为自有现场数据并取得相应授权。
