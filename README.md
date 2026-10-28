# ICML2026（4433均分3.5）｜BasisBFN

[![自动测试](https://github.com/Q531531/BasisBFN/actions/workflows/ci.yml/badge.svg)](https://github.com/Q531531/BasisBFN/actions/workflows/ci.yml)

当前版本：**v1.0.0**。

**面向恶劣天气图像复原的图像依赖基底贝叶斯残差精修**

BasisBFN 在冻结的 SSR 复原主干上添加轻量精修模块：一次生成四个 RGB
残差基底，在空间系数上执行四步贝叶斯更新，再通过后验条件门控合成校正。
本仓库提供方法实现、参数匹配的消融配置、训练／推理／评测入口及验证工具。

[方法细节](docs/BASISBFN.md) · [数据准备](docs/DATASETS.md) ·
[发布内容与复现要求](docs/REPRODUCIBILITY.md) · [开发指南](CONTRIBUTING.md)

![BasisBFN 训练与推理框架](docs/assets/basisbfn_architecture.png)

## 核心实现

- **冻结复原主干：** SSR 产生基础复原图像及 48 通道解码特征，训练时保持冻结。
- **单次条件 U-Net：** 61 通道条件输入经过 U-Net，生成共享上下文与四个
  图像依赖的 RGB 残差基底；重型 U-Net 和基底各计算一次。
- **系数空间 BFN：** 实现系数目标投影、Gaussian sender、噪声状态、
  共享 observer、离散 BFN 损失和四步贝叶斯精度加权融合。
- **后验条件门控：** 使用上下文、最终系数均值与对数精度，逐像素控制校正强度。
- **初始化保持恒等：** observer 输出投影零初始化，保证初始校正严格为零。
- **轻量精修分支：** 完整新增分支包含 **624,217 个可训练参数**；
  该数值不包含冻结 SSR 主干。

四个消融分支共享参数规模与训练目标，仅改变状态更新规则和门控是否应用：

| 分支 | 系数更新 | 接受门控 |
| --- | --- | --- |
| `D4` | 确定性加法 | 关闭 |
| `D4-G` | 确定性加法 | 开启 |
| `B4` | 贝叶斯融合 | 关闭 |
| `B4-G` | 贝叶斯融合 | 开启 |

## 快速验证：无需数据集和权重

下载仓库后，从仓库根目录执行：

```bash
git clone https://github.com/Q531531/BasisBFN.git
cd BasisBFN

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"

python3 scripts/smoke_basisbfn.py --device cpu
python3 -m pytest -q
```

快速自检运行真实的新增 readout 模块，检查形状、零校正初始化、四步更新、
后验精度、门控范围及损失反向传播。它使用小型测试张量，**不生成复原图片，
不衡量 PSNR，也不代替包含 SSR 的完整训练／推理**。

GitHub Actions 自动运行 CPU 回归、模块自检、主要命令入口检查和 wheel 构建。

## 完整运行的环境

建议使用 Python 3.10 或 3.11。CPU 可以运行上述验证；完整 SSR 主干依赖
NVIDIA CUDA 和 `mamba-ssm`。SSR 官方源码与预训练权重需要单独准备：

```bash
export SSR_REPO=/path/to/SSR
export SSR_CHECKPOINT="$SSR_REPO/Allweather/pretrained_models/net_g_best.pth"
export SGB_DATA_ROOT=/path/to/allweather
export ALLWEATHER_TEST_ROOT=/path/to/testallweather
```

[`scripts/setup_ssr_foundation.sh`](scripts/setup_ssr_foundation.sh)
提供官方 SSR 源码、依赖和权重的准备流程，并检查权重能被正确加载。
若需要 LPIPS 或可选 CLIP 功能，可安装 `python3 -m pip install -e ".[all]"`。

## 数据准备

训练数据使用 All-Weather 合集，
包含 Outdoor-Rain、Snow100K 和 RainDrop，共 **18,069 对**图像。

[数据配对与划分说明](docs/DATASETS.md)

当前配置要求以下目录与数量：

```text
allweather/
├── input/
├── gt/
├── train_pairs.tsv     # 17,166 对
└── val_pairs.tsv       # 903 对
```

四个模型训练种子 `42/43/44/45` 共用同一份数据划分。发布包不分发原实验
TSV 清单；精确复现论文划分需要原始清单。

正式测试使用 Snow100K-S、Snow100K-L、Outdoor-Rain Test1 和 RainDrop
Test-A，与训练／验证数据分别保存。

## 论文配置

主模型为 `B4-G`：

| 配置项 | 取值 |
| --- | --- |
| 残差基底／系数更新步数 | `K = 4`／`T = 4` |
| Gate 输入 | 上下文、后验均值、对数精度 |
| 残差尺度 | `0.2` |
| 系数目标 | 停止梯度的 ridge 投影，ridge `1e-3` |
| 系数 BFN 损失配置权重 | `4e-5`，公式解释见方法文档 |
| 优化器 | AdamW，学习率 `2e-4`，weight decay `1e-4` |
| 训练 | 6 epochs，batch 16，BF16，EMA `0.999` |
| 模型训练种子 | `42, 43, 44, 45` |
| 固定评测端点 | epoch-6 EMA checkpoint |

## 生成配置、预检与训练

以 `B4-G`、seed 42 为例：

```bash
python3 scripts/make_basisbfn_config.py \
  --arm B4-G --seed 42 \
  --output configs/generated/basisbfn_b4g_seed42.yaml

python3 scripts/check_dataset.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml

python3 scripts/preflight_basisbfn.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --output-dir runs/basisbfn/artifacts/b4-g/seed_42/preflight \
  --device cuda

python3 scripts/profile_basisbfn.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --preflight runs/basisbfn/artifacts/b4-g/seed_42/preflight/preflight_b4-g.json \
  --output runs/basisbfn/artifacts/b4-g/seed_42/profile.json \
  --device cuda

python3 train.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --device cuda
```

训练入口核对配置、SSR 架构与权重路径、数据清单和对应 arm／seed 的
CUDA preflight／200-step profile。训练 GPU 必须与 profile 记录一致。
预检覆盖冻结状态、参数量、初始化、损失有效性与梯度可达性。
其它分支和种子使用相同流程，分别生成配置和运行目录。

## 推理与评测

训练后的固定端点为：

```text
runs/basisbfn/b4-g/seed_42/epoch_0006.pt
```

对真实图片进行推理：

```bash
python3 infer.py \
  --checkpoint runs/basisbfn/b4-g/seed_42/epoch_0006.pt \
  --input /path/to/degraded/images \
  --output results/basisbfn \
  --device cuda --deterministic
```

换机器后，checkpoint 中保存的 SSR 路径可能已失效。可额外传
`--config` 指定相同模型设置、当前 SSR 路径的配置；推理加载器只允许迁移
SSR 路径，拒绝修改模型语义。

按明确的图像对清单进行验证集评测：

```bash
python3 evaluate.py \
  --checkpoint runs/basisbfn/b4-g/seed_42/epoch_0006.pt \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda --test-y-channel --compare-structural \
  --json results/basisbfn_b4g_seed42.json
```

正式四子集的目录检查与评测入口分别为
[`check_allweather_test.py`](scripts/check_allweather_test.py) 和
[`evaluate_allweather.sh`](scripts/evaluate_allweather.sh)。

## 论文报告结果

以下数值为论文报告的四种子均值。原始逐图评测 JSON 与实验权重不随发布包
分发；这些数值属于论文结果，自动测试验证范围见复现文档。

| 模型 | 平均 PSNR | 平均 SSIM |
| --- | ---: | ---: |
| SSR | 34.15 | 0.9442 |
| BasisBFN B4-G | 34.50 | 0.9483 |

论文另报告贝叶斯融合贡献 `+0.10 dB`、门控贡献 `+0.08 dB`。
代码测试只验证实现行为，不证明这些质量增益。标题中的
`ICML2026（4433均分3.5）` 是项目提供的评审信息，不表示论文已录用。

发布内容、自动验证范围与精确复现要求见
[发布内容与复现要求](docs/REPRODUCIBILITY.md)。

## 代码导航

```text
sgbformer/models/coefficient_state.py  # 基底、系数 observer、BFN 目标与门控
sgbformer/models/receiver.py           # 单次条件 U-Net
sgbformer/models/bfn.py                # Gaussian sender、贝叶斯更新和离散损失
sgbformer/models/foundation.py         # 外部冻结 SSR 集成
sgbformer/models/model.py              # 完整模型与训练输出
sgbformer/losses.py                    # 训练损失组合
sgbformer/basisbfn.py                  # 独立的四组配置与启动检查
sgbformer/configs/basisbfn.yaml        # 随 Python 包分发的公共契约
sgbformer/data/                        # 图像对加载与采样
configs/experiments/basisbfn_*.yaml     # 主模型和参数匹配的消融配置
scripts/smoke_basisbfn.py              # 无权重的模块自检
scripts/preflight_basisbfn.py          # 训练前预检
scripts/profile_basisbfn.py            # 200-step CUDA profile
scripts/analyze_basisbfn.py            # 固定端点与消融矩阵分析
tests/                                # 回归测试
.github/workflows/ci.yml               # GitHub 自动验证
```

Python 导入命名空间保留 `sgbformer`，公开模型／损失名称为 `BasisBFN` 和
`BasisBFNLoss`；兼容别名用于加载已有 checkpoint。
