# 验证与复现范围

本页说明公开代码能验证什么，以及精确复现实验还需要哪些材料。

## 公开实现与自动验证

| 内容 | 对应实现或验证入口 |
| --- | --- |
| 图像依赖的 RGB 基底、系数 observer 和后验门控 | `sgbformer/models/coefficient_state.py` |
| Gaussian sender、贝叶斯更新和离散损失 | `sgbformer/models/bfn.py` |
| 冻结 SSR、单次 receiver 与训练损失集成 | `sgbformer/models/model.py`、`receiver.py`、`losses.py` |
| 624,217 个可训练参数及参数匹配的四分支 | `tests/test_basisbfn.py`、BasisBFN preflight |
| 初始化零校正、精度增长、门控输入和梯度 | 模块自检与回归测试 |
| BF16 下 ridge 投影的 FP32 求解 | `tests/test_basisbfn.py` |
| 四分支 × 四种子的配置和固定端点 | 配置生成器、协议验证器与回归测试 |
| 数据加载、checkpoint、推理和指标实现 | `tests/` 中对应回归测试 |
| CPU 回归、主要入口及 Python 包构建 | `.github/workflows/ci.yml` |

无权重的快速验证：

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/smoke_basisbfn.py --device cpu
python3 -m pytest -q
```

模块自检使用小型测试张量运行真实 readout，不下载权重或数据，不保存实验
checkpoint，不输出复原图像或 PSNR。回归测试中的替代 SSR 模型用于验证
接口和冻结行为，不能作为真实 SSR 推理结果。

## 完整 CUDA 训练／推理

README 提供完整运行入口。执行这些入口还需：

- 官方 SSR 源码与匹配的预训练 checkpoint；
- CUDA 环境及 SSR 所需的 `mamba-ssm`；
- 图像数据、正确的图像对清单和当前实验配置；
- 对应 arm／seed 的真实 preflight 与 200-step profile；
- 固定 epoch-6 EMA checkpoint，才能运行已有模型的真实图像推理。

CPU 单测和 GitHub Actions 不代替以上端到端验证，不生成 CUDA preflight
或 profile 产物。

## 实验结果的证据

README 中的 `34.50 / 0.9483`、贝叶斯融合 `+0.10 dB` 和门控 `+0.08 dB`
是论文稿报告值。当前公开仓库尚未附足以逐图验证这些结果的评测 JSON、
原始划分清单和完整运行资产。不能把这些数值解释为 CI 测量结果。

标题中的 ICML2026 评审分数来自项目提供的信息；仓库未附公开评审记录，
也未声明论文已经录用。

精确复现和对比结果时，应保留以下材料：

| 材料 | 用途 | 当前公开范围 |
| --- | --- | --- |
| 代码 commit、依赖与 CUDA 版本 | 固定实现和运行环境 | 代码公开；实际运行环境需记录 |
| SSR 代码版本和权重 SHA-256 | 固定冻结主干 | 集成与准备脚本公开；源码和权重外部获取 |
| 原始 `train_pairs.tsv`、`val_pairs.tsv` 及哈希 | 固定逐图划分 | 未公开 |
| 四分支、四种子的解析后配置 | 固定实验设置 | 配置模板和生成器公开 |
| preflight、profile 和训练日志 | 验证真实运行条件 | 生成脚本公开；原始结果未随仓库发布 |
| epoch-6 EMA checkpoint 的 SHA-256 | 固定评测对象 | 仓库不分发模型权重 |
| 原始逐图评测 JSON、指标与 bootstrap 输出 | 核查 PSNR、SSIM 和消融贡献 | 分析入口公开；论文原始评测产物未公开 |

如需补充公开实验依据，可以发布非权重材料，例如原始划分清单、配置、
checkpoint 哈希和逐图评测 JSON；不需要把预训练模型放入代码仓库。

## 用于项目展示

当前可以依据代码与测试介绍 Python／PyTorch 方法实现、冻结主干集成、
参数匹配的消融设计、预检与指纹校验、训练／评测工具及自动回归。
精确的质量提升数字应附对应实验依据，并明确其来自论文报告还是已复现测量。
