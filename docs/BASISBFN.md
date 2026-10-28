# BasisBFN 方法与实验协议

BasisBFN 的完整名称为 **Bayesian Residual Refinement with Image-Dependent
Bases for Adverse Weather Image Restoration**。主模型是 `B4-G`，在冻结的
SSR 上进行系数空间的贝叶斯残差精修。

## 模型与训练目标

SSR 产生基础复原图像和 48 通道特征。条件 U-Net 将 61 通道条件输入映射为
16 通道共享上下文 `h`。readout 一次构造四个图像依赖的 RGB 基底：

```text
basis = Conv2d(SiLU(GroupNorm(h)))       # [B, 4, 3, H, W]
mu_0 = 0                               # [B, 4, H, W]
rho_0 = 1
```

共享 observer 每步接收 `[h, mu_i, log(rho_i), time_i]`，预测有界系数观测
`y_i` 和自适应精度 `alpha_i`。四步更新使用以下两种规则：

```text
rho_next = rho + alpha
Bayesian:      mu_next = (rho * mu + alpha * y) / rho_next
Deterministic: mu_next = mu + (alpha / rho_next) * y
```

训练时，目标残差先除以 `residual_scale=0.2`，再通过停止梯度的 ridge
投影求出系数目标，ridge 为 `1e-3`。随机采样一个离散时间，Gaussian
sender 生成带噪系数状态，observer 预测干净系数目标。两种更新规则使用
同一训练目标。

当前实现的损失口径为：

```text
T = 4
discrete_loss = (T / 2) * alpha_step * mean((target_coeff - prediction)^2)
bfn_training_loss = discrete_loss / T
total_loss = MSE(restored, target) + 4e-5 * bfn_training_loss
```

`bfn.py` 的 `discrete_loss` 包含 `T` 采样补偿；`coefficient_state.py` 的
`bfn_training_loss` 显式除以 `T`。因此 `4e-5` 是配置对模型输出的
`bfn_loss` 张量的权重。该链路没有再乘 `residual_scale²`。复查公式时应
同时检查这两层，不能混用其它损失定义的权重。

最终校正与输出为：

```text
gate = sigmoid(Conv([h, mu_4, log(rho_4)]))
correction = gate * sum_k(basis_k * tanh(mu_4,k)) / sqrt(4)
restored = clamp(frozen_ssr_output + 0.2 * correction, 0, 1)
```

observer 输出投影零初始化；gate 输出投影也零初始化，初始概率为 `0.5`。
由于系数初始为零，四组配置最初都保持冻结 SSR 映射，第一次反向传播可以
到达 observer。一次优化后，梯度继续到达基底、U-Net 和启用的 gate。

## 四组受控消融

| 配置 | 系数更新 | 后验门控 |
| --- | --- | --- |
| `basisbfn_d4.yaml` | 确定性加法 | 关闭 |
| `basisbfn_d4g.yaml` | 确定性加法 | 开启 |
| `basisbfn_b4.yaml` | 贝叶斯融合 | 关闭 |
| `basisbfn_b4g.yaml` | 贝叶斯融合 | 开启 |

四组均实例化同样的 gate 参数，具有相同的参数名称和 **624,217** 个可训练
参数。关闭 gate 时只跳过应用，不删除其参数。每张图像调用一次 SSR、
一次 U-Net、一次基底生成和四次系数 observer；训练额外调用一次随机时间
observer。外层 RGB 更新参数 `model.bfn.update_mode` 保持 `deterministic`，
实际系数规则由 `coefficient_state_readout.update_mode` 决定。

配置由公共契约、共享训练设置和 arm 设置组成：

```text
sgbformer/configs/basisbfn.yaml
  → configs/experiments/_basisbfn_common.yaml
    → configs/experiments/basisbfn_{d4,d4g,b4,b4g}.yaml
      → 可选的 arm/seed 生成配置
```

公共契约随 wheel 分发，协议检查直接使用该契约。
`sgbformer` 导入命名空间与通用模型组件保留，便于加载已有模型状态。

## 预检、profile 与训练

完整命令见 [README](../README.md)。每个 arm 和模型种子
`42/43/44/45` 使用各自的配置、产物和运行目录，共用固定数据划分。

真实 CUDA preflight 检查 17,166 对训练数据、903 对验证数据、八张图的
初始化恒等、DCT 正交初始化、参数冻结、精度增长、门控范围与梯度可达性。
200-step CUDA profile 检查训练调用次数，并记录吞吐与显存，显存上限为
2 GiB。它不计算验证质量指标。

启动检查核对源配置路径与解析后配置、SSR 架构文件与权重路径、两份清单的
路径和数量、CUDA GPU 名称与 profile 批量设置。
改动任一核对项后应重新生成真实预检与 profile。CPU 单测中的合成报告
只验证拒绝逻辑，不是正式 CUDA 运行报告。

## 固定端点评测与分析

论文评测端点为 **epoch-6 EMA**。端点验证器拒绝其它 epoch、
`best_psnr.pt`、平均权重和 self-ensemble 结果。固定端点评测为确定性、
整图、Y 通道、crop border 0、无 self-ensemble。

先对同一验证清单生成冻结 SSR 基线：

```bash
python3 evaluate_foundation.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda --test-y-channel \
  --json results/f0_dev.json
```

再按 README 的 `evaluate.py` 命令分别生成四组 epoch-6 JSON。同一模型种子的
矩阵分析示例：

```bash
python3 scripts/analyze_basisbfn.py \
  --baseline results/f0_dev.json \
  --checkpoint D4=runs/basisbfn/d4/seed_42/epoch_0006.pt \
  --checkpoint D4-G=runs/basisbfn/d4-g/seed_42/epoch_0006.pt \
  --checkpoint B4=runs/basisbfn/b4/seed_42/epoch_0006.pt \
  --checkpoint B4-G=runs/basisbfn/b4-g/seed_42/epoch_0006.pt \
  --candidate D4=results/d4_seed42.json \
  --candidate D4-G=results/d4g_seed42.json \
  --candidate B4=results/b4_seed42.json \
  --candidate B4-G=results/b4g_seed42.json \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --output-dir results/matrix_seed42 \
  --bootstrap-samples 10000 --seed 42
```

分析器核对 checkpoint 路径、epoch、EMA 和模型调用协议，输出四组
对 F0 的差异、D/B 效应、gate 效应及交互。按天气类别分别 bootstrap，
Snow／Outdoor-Rain／RainDrop 权重为 `0.50/0.25/0.25`。图像 bootstrap
不包含模型训练种子方差。论文结果与公开证据范围见
[REPRODUCIBILITY](REPRODUCIBILITY.md)。
