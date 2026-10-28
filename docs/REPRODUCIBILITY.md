# 发布内容与复现要求

BasisBFN v1.0.0 发布方法代码、四组消融配置、训练／推理／评测入口、
自动测试和 Python wheel。本页列出验证范围、外部运行依赖和论文复现材料。

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
| 四分支训练、断点恢复、EMA、图像推理与评测 | `tests/test_train_integration.py` |
| 数据加载、checkpoint 失败清理和指标实现 | `tests/` 中对应回归测试 |
| CPU 回归、主要入口及 Python 包构建 | `.github/workflows/ci.yml` |

无权重的快速验证：

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/smoke_basisbfn.py --device cpu
python3 -m pytest -q
```

模块自检使用小型测试张量运行真实 readout，不下载权重或数据，不保存实验
checkpoint，不输出复原图像或 PSNR。回归测试中的替代 SSR 模型用于验证
接口和冻结行为。四组集成测试使用替代 SSR 和合成图像，验证训练、精确
断点恢复、EMA、图像文件推理和逐图评测；该测试不测量真实 SSR 的复原质量。

## 完整 CUDA 训练／推理

README 提供完整运行入口。执行这些入口还需：

- 官方 SSR 源码与匹配的预训练 checkpoint；
- CUDA 环境及 SSR 所需的 `mamba-ssm`；
- 图像数据、正确的图像对清单和当前实验配置；
- 对应 arm／seed 的真实 preflight 与 200-step profile；
- 固定 epoch-6 EMA checkpoint，才能运行已有模型的真实图像推理。

CPU 单测和 GitHub Actions 不代替以上端到端验证，不生成 CUDA preflight
或 profile 运行报告。

## 实验结果的证据

README 中的 `34.50 / 0.9483`、贝叶斯融合 `+0.10 dB` 和门控 `+0.08 dB`
是论文报告值。原始逐图评测 JSON、划分清单和实验权重不随本版本分发；
论文质量指标与 CI 验证范围分别列示。

标题中的 ICML2026 评审分数为项目提供的信息，不代表录用状态；评审记录
不属于本版本分发内容。

精确复现和对比结果时，应保留以下材料：

| 材料 | 用途 | 本版本分发内容 |
| --- | --- | --- |
| 代码 commit、依赖与 CUDA 版本 | 固定实现和运行环境 | 代码公开；实际运行环境需记录 |
| SSR 代码版本和预训练权重 | 固定冻结主干 | 集成与准备脚本公开；源码和权重外部获取 |
| 原始 `train_pairs.tsv`、`val_pairs.tsv` | 固定逐图划分 | 不分发；提供确定性划分生成器 |
| 四分支、四种子的解析后配置 | 固定实验设置 | 配置模板和生成器公开 |
| preflight、profile 和训练日志 | 验证真实运行条件 | 提供生成脚本；运行报告由实际训练环境生成 |
| epoch-6 EMA checkpoint | 固定评测对象 | 仓库不分发模型权重 |
| 原始逐图评测 JSON、指标与 bootstrap 输出 | 核查 PSNR、SSIM 和消融贡献 | 提供评测与分析入口；原始结果不分发 |

复现实验应在独立运行目录保存清单、解析配置、环境信息、epoch-6 EMA
checkpoint 和逐图评测 JSON，以固定评测对象和统计口径。
