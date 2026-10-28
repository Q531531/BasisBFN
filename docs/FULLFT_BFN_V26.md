# v26 主线：全解冻基座 + 保留 BFN（FULLFT_BFN_V26）

> 标注约定：【代码事实】给出文件与行号；【已测量】给出实验/文档来源；【假设】为未验证推断。
> 本文档对应新增配置 `configs/experiments/ssr_bfn_fullft_v26.yaml`、
> `ssr_bfn_fullft_v26_pilot.yaml`。
>
> **更新（pilot 跑完之后）**：v26 已被 `ssr_bfn_fullft_v26b.yaml` 取代，判定实验为
> `ssr_bfn_fullft_v26b_proxy.yaml`（6 epoch），分析工具为
> `scripts/analyze_dev_by_category.py`。原因、实测数字与完整命令序列见 **§6**。
> §1–§5 保留 pilot 之前的原始设计记录，未作改写。

## 1. 结论先行：为什么换主线

【已测量，已归档 adapter-only 实验记录】官方四集 foundation 评测为
**Macro PSNR 34.239638 / Macro LPIPS 0.062147**；同一口径下所有"改动解码器 /
加小 adapter"的实验都低于它：

| 实验 | Macro PSNR | Δ vs foundation |
|---|---:|---:|
| Foundation（外部 SSR） | 34.239638 | 0 |
| Decoder-only composite | 34.171370 | −0.068268 |
| Decoder + spectral composite | 34.171290 | −0.068349 |
| Decoder + spectral Charbonnier | 34.162996 | −0.076642 |
| Decoder + spectral MSE | 34.206020 | −0.033618 |

【已测量，已归档 adapter-only 对照】"冻结全部 SSR、只训 adapter"的对照
臂可训练参数仅 29,779；正在跑的 `frozen_main_db` D/B 也只放开
`decoder_level1/decoder_level2/output`
（【代码事实】`configs/experiments/_ssr_frozen_main_common.yaml:41-44`），
可训练参数约 0.95M【已测量，用户提供的当前运行状态】。

结论：**"冻结基座"这条自我约束本身就是上限**。SSR 预训练权重已经在同一训练集上
收敛，只在其外部挂一个几十万参数的旁路，既没有容量也没有梯度通路去修正基座内部
表征。v26 的设计意图是：**只保留 BFN 这一个论文层面的硬约束，其它约束全部放开**。

## 2. v26 与 frozen-main 的逐项对照

| 维度 | frozen_main_db（D/B，正在跑） | v26 主线 |
|---|---|---|
| `protocol.name` | `frozen_main_db`（触发全套锁） | `fullft_bfn_v26`（不触发任何锁） |
| 基座是否解冻 | `freeze: true`，仅 `decoder_level1/level2/output` 可训 | `freeze: false`，SSR 全部参数可训，`trainable_modules: []` |
| 基座 eval 模式 | `frozen_modules_eval: true` | `frozen_modules_eval: false`（全网络正常 train 模式） |
| 可训练参数 | ≈0.95M（含三个解码器模块）【已测量，用户提供】 | ≈11.3M：SSR 本体 **10,825,680**【已测量：`shasum -a 256 pretrained/ssr_net_g_best.pth` = `18897da4561959ca122bb2e43aa3e70bb585e5255406305da3f43365f19d7284`，载入其 `params` 键统计 558 个张量共 10,825,680 个参数】+ 新增 adapter/BFN 约 0.4~0.5M【估算：global 32→64、spatial 8→16、hidden 64→128 后由原 ≈0.198M 放大约 2~3 倍】。**精确值以 pilot banner 的 `trainable_params=` 为准** |
| BFN/adapter 容量 | global 32 / spatial 8 / hidden 64；latent flow 40,179 + dec3 55,745 + dec2 37,025 + gate 73,920 = **206,869** | global 64 / spatial 16 / hidden 128；latent flow 133,603 + dec3 145,729 + dec2 108,577 + gate 73,920 = **461,829** 【已测量：本机 CPU 实例化 `DegradationLatentBFN` / `LatentConditionedFiLMSpectralAdapter` / `PaperSpectralGateAdapter` 统计】 |
| BFN 步数与更新 | steps 4，D=deterministic / B=bayesian | steps 4，`update_mode: bayesian`（ctrl 臂为 steps 1 + deterministic） |
| adapter 插入点 | `encoder_level3` / `decoder_level3` / `decoder_level2` | 相同（不能再加 `decoder_level1`，见 §3.2） |
| 采样 | `sampling: {mode: shuffle}` | `category_balanced`，Snow 0.50 / Outdoor-Rain 0.25 / RainDrop 0.25（对齐官方四集 macro 口径） |
| 损失 | structural(MSE) 1.0 + bfn 0.01 + latent_category 0.01 | 同上再加 `frequency_weight: 0.05`（【代码事实】`sgbformer/losses.py:150-154` 对 `outputs["structural"]` 取 `frequency_amplitude_loss`） |
| 学习率 | 新模块 2e-4；三个解码器模块各 2e-5 | 新模块 2e-4；`backbone.model` 整体 1.5e-5（【代码事实】`train.py:110-117` 最长前缀匹配） |
| 预算/门禁 | `require_budget_lock: true` + profiler/preflight/manifest + epoch-4 tripwire | `require_budget_lock: false`，`protocol.tripwire: null`，无 manifest |
| epoch | 由 GPU-hour 预算推导（12–30） | 固定 30（单 `joint` stage） |
| 测试期增强 | 无 | `evaluate.py --self-ensemble`（×8 几何 TTA）+ `scripts/average_checkpoints.py`（权重平均），二者必须单独成行披露 |

## 3. 代码事实

### 3.1 梯度通路：为什么 `freeze: false` 就够了

1. 【代码事实】`sgbformer/models/foundation.py:301`：
   `self.model.requires_grad_(not freeze)` —— `freeze: false` 让 SSR 全部参数
   `requires_grad=True`。
2. 【代码事实】`foundation.py:304-308`：`foundation_trainable` 因此为 `True`，
   并且不会再调用 `self.model.eval()`。
3. 【代码事实】`foundation.py:475`：
   `context = nullcontext() if self.foundation_trainable else torch.no_grad()` ——
   这是关键分支。`cache_decoder_prefix()` 里的
   `patch_embed → encoder_level1 → down1_2 → encoder_level2 → down2_3 →
   encoder_level3 → decoder_level3`（`foundation.py:477-483`）在
   `foundation_trainable=True` 时**保留 autograd 图**，encoder 与 decoder_level3
   的基础权重真的能拿到梯度。
4. 【代码事实】latent 训练路径上没有任何 `no_grad`：
   `train.py:762` → `model.py:1254-1265 training_outputs` →
   `model.py:1014 latent_training_outputs` → `model.py:619 cache_decoder_prefix`
   → `model.py:710 decode_from_cached_decoder3`。
5. 【代码事实】`model.py:1071-1077`：latent 路径的输出里
   `restored == structural == foundation`，因此 `loss.structural_weight`
   直接监督"被调制后的完整前向"，梯度会同时流进 BFN、adapter 和 SSR 本体。

结论：**端到端联合训练不需要改任何源码，只需要 `freeze: false`。**

### 3.2 为什么 latent adapter 不能放到 `decoder_level1`

【代码事实】`foundation.py:451-462`（`supports_latent_cache`）要求
latent_film_spectral adapter 的模块名集合 ⊆ `{decoder_level3, decoder_level2}`
且必须含 `decoder_level3`；【代码事实】`model.py:249-253` 在不满足时抛
`latent_bfn requires decoder_level3 and decoder_level2 latent_film_spectral
adapters on a split-capable SSR`。加 `decoder_level1` 会在建模型时直接崩，放开它
必须改 `sgbformer/**`（禁止项）。

### 3.3 其他必须遵守的源码约束（已逐项核对）

- `core.bfn_steps` 必须等于 `latent_bfn.steps`：【代码事实】`model.py:344`
  把 `self.bfn_steps` 覆盖为 `latent_steps`，`model.py:363-368` 再做相等校验。
  因此 ctrl 臂把 `core.bfn_steps` 一并改成 1。
- `model.bfn.inference_mode: none`：【代码事实】`model.py:184-204`，否则会额外构造
  一个 latent 路径根本不用的 `ResidualReceiver`（其参数会进 optimizer 但永远没有
  梯度）。`model.py:345` 随后把 `refinement_mode` 恢复为 `iterative`，
  所以 `core.require_iterative_bfn: true` 仍然通过。
- stage 的 epochs 之和必须等于 `training.epochs`：【代码事实】
  `sgbformer/training.py:49-52`；latent 路径只允许 `joint` stage：
  【代码事实】`model.py:1254-1258`。
- 采样键名/取值格式：【代码事实】`train.py:384-405` 读
  `mode` / `category_patterns` / `expected_counts` / `target_fractions`，
  `sgbformer/data/samplers.py:15-46` 用 `re.fullmatch` 分类并与
  `expected_counts` 严格比对（**仅在该键存在时**），`samplers.py:69-80` 要求
  `target_fractions` 恰好覆盖全部观测类别。v26 **故意不写** `expected_counts`，
  因为训练流已改为 95% 划分，其逐类别数量会随 split manifest 重新生成而变化；
  实际命中的数量每次运行都会打到日志里（【代码事实】`train.py:633-641` 的
  `sampling_source_counts` / `sampling_epoch_targets`），以日志为准。
- `data.train.pairs_file: ${SGB_DATA_ROOT}/train_pairs.tsv`（**与 frozen-main
  不同，必须持出 dev**）：frozen-main 用 `pairs_file: null` 目录直扫 18069 对，
  而 dev（`val_pairs.tsv`，903 对）是这 18069 对里随机抽的 5%
  （【代码事实】`scripts/create_split_manifest.py`），也就是说 **frozen-main 的 dev
  是训练集子集**。可训练参数只有 0.95M 时这个泄漏还能忍；v26 有约 11.3M 可训练参数（10,825,680 + 461,829 = 11,287,509）、
  跑 30 epoch，dev 曲线会直接变成记忆度曲线，`checkpoint_selection` 与过拟合监控
  全部失效。因此 v26 训练在 95% 划分（约 17166 对）上，把 903 对 dev 留在训练流外。
  - 诚实口径的保留意见：SSR 初始化权重本身就是在全部 18069 对上预训练的，所以
    dev 在权重层面也不是干净的 held-out；它只能保证**本次微调**不再额外记忆这
    903 张，而这正是检测全解冻退化所需要的信号。
  - 前置条件：`${SGB_DATA_ROOT}/train_pairs.tsv` 必须存在，缺失时用
    `scripts/create_split_manifest.py` 重新生成，并核对
    `wc -l ${SGB_DATA_ROOT}/train_pairs.tsv` 约为 17166。
- RainDrop 过采样：95% 划分里 RainDrop 约 777 对，却要占 25% 的 epoch 配额，
  相当于每 epoch 重复约 5.5 次。`samplers.py:107-118` 每次补池都重洗，且抽取序号
  进入增强种子（`samplers.py:50-53`），重复样本会拿到不同 crop；但仍需在 dev 曲线上
  盯 RainDrop 的单独过拟合。
- `global_dim must exceed num_categories`：【代码事实】
  `sgbformer/models/bfn.py:316-317`，64 > 3 通过。

## 4. 为什么一行 `sgbformer/**` 都没改

【代码事实】`sgbformer/protocol.py:54-80`（`implementation_fingerprint`）对
`sgbformer/**/*.py`（递归）+ `train.py` + `scripts/profile_frozen_main.py` +
`scripts/preflight_frozen_main.py` + `scripts/freeze_frozen_main_budget.py` +
`scripts/decide_frozen_main_tripwire.py` 共 31 个文件做 SHA256；
【代码事实】`protocol.py:518-526`：当 `protocol.name == "frozen_main_db"` 且
resume 时，指纹不匹配就 `RuntimeError: Frozen-main implementation SHA256
mismatch`——正在跑的 D/B 会直接废掉。

本次改动落在指纹清单**之外**（已用 `implementation_fingerprint()` 实际核对：
清单 31 个文件，不含 `evaluate.py`，也不含 `scripts/average_checkpoints.py`）：

```text
configs/experiments/ssr_bfn_fullft_v26.yaml        （新增）
configs/experiments/ssr_bfn_fullft_v26_pilot.yaml  （新增）
scripts/average_checkpoints.py                     （新增）
evaluate.py                                        （修改：--self-ensemble）
docs/FULLFT_BFN_V26.md                             （新增）
tests/test_fullft_v26_config.py                    （新增）
```

pilot 复盘（§6）追加的文件同样全部落在指纹清单之外：

```text
configs/experiments/ssr_bfn_fullft_v26b.yaml        （新增）
configs/experiments/ssr_bfn_fullft_v26b_proxy.yaml  （新增）
configs/experiments/ssr_bfn_fullft_v26.yaml         （修改：仅顶部注释）
configs/experiments/ssr_bfn_fullft_v26_pilot.yaml   （修改：仅顶部注释）
scripts/analyze_dev_by_category.py                  （新增）
docs/FULLFT_BFN_V26.md                              （修改：本节与 §6）
tests/test_analyze_dev_by_category.py               （新增）
tests/test_fullft_v26_config.py                     （修改：v26b 契约断言）
```

`scripts/analyze_dev_by_category.py` **不在**指纹清单里（清单只含四个
`*frozen_main*` 脚本），改动后实测指纹仍为
`9ac5db2bc57cb458be066c1bf795f48a3a35342700813c66619911abafc935df`（31 个文件）。

v26 也不会触发任何 frozen 锁：【代码事实】`train.py:324-325`、`train.py:377`、
`train.py:460`、`train.py:470` 全部以 `protocol.name == "frozen_main_db"` 为条件；
`protocol.py:319-321` 在 `require_budget_lock: false` 时立刻返回 `None`，
不需要 manifest / profiler / preflight。

### 如果后续确实需要改 `sgbformer/**`

先把整棵代码树复制一份，让 locked D/B 的记录留在旧副本里，再在新副本上改：

```bash
cd ~/Documents/ky/SGBformer-Spectral-Gated-Bayesian-Flow-Network-Enhanced-Transformer-main
cp -a sgbformer sgbformer_locked_db     # 旧副本：只用于 D/B resume 与复盘
cd sgbformer                            # 新副本：在这里做 v26 及任何源码改动
```

D/B 的 resume 必须在 `sgbformer_locked_db` 里执行（其 `runs/` 与 manifest 一起
被复制，指纹保持一致）。

## 5. 执行命令序列（5090）

```bash
cd /path/to/bfn/sgbformer
conda activate diffusionad
export SGB_DATA_ROOT=/path/to/allweather
export SSR_REPO=/path/to/SSR
export SSR_CHECKPOINT=$SSR_REPO/Allweather/pretrained_models/net_g_best.pth
export ALLWEATHER_TEST_ROOT=/path/to/testallweather
```

### 5.1 Pilot（2 epoch，先看 OOM / 吞吐 / dev）

```bash
tmux new -s v26_pilot
python train.py --config configs/experiments/ssr_bfn_fullft_v26_pilot.yaml \
  --device cuda 2>&1 | tee runs/fullft_v26_pilot.log
```

看三件事：
1. 启动 banner 的 `trainable_params=`（【代码事实】`train.py:572`）与
   `latent_update_mode=bayesian`、`bfn_steps=4`（【代码事实】`train.py:613-631`）；
2. `sampling_source_counts`：训练流现在是 95% 划分（约 17166 对），三类计数以
   日志实际打印为准（自然分布下约 `snow ≈ 8551 / outdoor_rain ≈ 7838 /
   raindrop ≈ 777`，具体取决于 split manifest 的随机划分）；
   `sampling_epoch_targets` 应满足 snow ≈ 总数的 50%、outdoor_rain 与 raindrop
   各 ≈ 25%（【代码事实】`train.py:633-641` 打印，`samplers.py:89-102` 把余数分给
   小数部分最大的类别）。**把这两行原样抄进 run log 记录**，因为 v26 不再硬编码
   `expected_counts`；
3. 每 epoch 的 optimizer step 数 ≈ 训练对数 / (micro 4 × accum 4) ≈ 1073
   （frozen-main 是 1130，因为它把 903 对 dev 也算进了训练流），据此估算
   30 epoch 的 GPU 小时；epoch 1–2 的 `dev_y_psnr` 不应明显低于冻结 SSR 的 dev 基线。
   注意 v26 的 dev 已真正持出，**其绝对值会低于 frozen-main 的 dev 数字，这是预期的，
   不是退化**；只能与 v26 自己的逐 epoch 曲线纵向比较。

### 5.2 主线（30 epoch）

```bash
tmux new -s v26_main
python train.py --config configs/experiments/ssr_bfn_fullft_v26.yaml \
  --device cuda 2>&1 | tee runs/fullft_v26_main.log
# 中断后续训（同一 config）：
python train.py --config configs/experiments/ssr_bfn_fullft_v26.yaml \
  --device cuda --resume runs/fullft_v26/main/latest.pt
```

逐 epoch 盯 `runs/fullft_v26/main/metrics.jsonl` 的 `validation.psnr`。
`best_psnr.pt` 由 dev Y-PSNR 选出（【代码事实】`train.py:986-1035`）。

### 5.3 官方四集评测

旧 latent 评测入口已在仓库清理时移除，不能复用。直接用 `evaluate.py`，
写法参照 `scripts/evaluate_allweather.sh:42-53`：

```bash
CKPT=runs/fullft_v26/main/best_psnr.pt
OUT=results/fullft_v26_main
mkdir -p "$OUT"
python scripts/check_allweather_test.py --root "$ALLWEATHER_TEST_ROOT"

names=(snow100k_s snow100k_l outdoor_rain raindrop)
inputs=("$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-S/synthetic"
        "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-L/synthetic"
        "$ALLWEATHER_TEST_ROOT/Test1/input"
        "$ALLWEATHER_TEST_ROOT/raindrop/test_a/data")
targets=("$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-S/gt"
         "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-L/gt"
         "$ALLWEATHER_TEST_ROOT/Test1/gt"
         "$ALLWEATHER_TEST_ROOT/raindrop/test_a/gt")

for i in "${!names[@]}"; do
  python evaluate.py --checkpoint "$CKPT" \
    --input-dir "${inputs[$i]}" --target-dir "${targets[$i]}" \
    --device cuda --test-y-channel --crop-border 0 --lpips \
    --quiet --json "$OUT/${names[$i]}.json"
done
```

四集 macro 用 `scripts/evaluate_allweather.sh:55-79` 末尾同样的方式取平均。与
foundation（或与 frozen-main 结果）做配对显著性对比时用
`scripts/compare_allweather.py`（【代码事实】其 CLI 为
`--baseline DIR/JSON --candidate DIR/JSON --output OUT [--bootstrap-samples N] [--seed S]`）：

```bash
python scripts/compare_allweather.py \
  --baseline results/ssr_official --candidate "$OUT" \
  --output "$OUT/vs_foundation.json"
```

### 5.4 ×8 self-ensemble（单独成行报告）

```bash
for i in "${!names[@]}"; do
  python evaluate.py --checkpoint "$CKPT" \
    --input-dir "${inputs[$i]}" --target-dir "${targets[$i]}" \
    --device cuda --test-y-channel --crop-border 0 --lpips \
    --self-ensemble --quiet --json "$OUT/${names[$i]}_tta8.json"
done
```

JSON 的 `protocol.self_ensemble=true`、`protocol.self_ensemble_transforms=8`、
`protocol.self_ensemble_geometry="rot90x4_times_horizontal_flip_pixel_mean"`，
`runtime.restoration_trajectories` 也会 ×8，论文里据此单独成行。
`--self-ensemble` 与 `--stochastic`、`--uncertainty` 互斥（会直接报错），
与 `--tile` / `--batch-size` / `--lpips` / `--category-pattern` / `--json` 兼容。

### 5.5 权重平均（单独成行报告）

`train.py:1010-1024` 在 `(epoch+1) % save_interval == 0` 时保存
`epoch_XXXX.pt`，v26 的 `save_interval: 1` 意味着**每个 epoch 都有快照**，
不需要手工 `cp`：

```bash
python scripts/average_checkpoints.py \
  --inputs runs/fullft_v26/main/epoch_0026.pt \
           runs/fullft_v26/main/epoch_0027.pt \
           runs/fullft_v26/main/epoch_0028.pt \
           runs/fullft_v26/main/epoch_0029.pt \
           runs/fullft_v26/main/epoch_0030.pt \
  --weights ema --output runs/fullft_v26/main/avg_last5_ema.pt

python evaluate.py --checkpoint runs/fullft_v26/main/avg_last5_ema.pt ...
```

如果将来把 `save_interval` 调大或删了快照（每个快照含 optimizer 状态，30 个会占
不少磁盘【假设：未实测体积】），在训练过程中手工留快照，不要为此改 `train.py`：

```bash
mkdir -p runs/fullft_v26/main/snapshots
cp runs/fullft_v26/main/latest.pt \
   runs/fullft_v26/main/snapshots/epoch_$(printf '%02d' "$EPOCH").pt
```

`--weights ema`（默认）复现的是 `evaluate.py` 实际加载的视图：先取 `model`
再用 `ema.shadow` 覆盖（【代码事实】`sgbformer/checkpoint.py:146-148`，
EMA 结构见 `sgbformer/ema.py:48-57`）。输出文件丢掉
optimizer/EMA/RNG（不可 resume），可直接喂给 `evaluate.py --checkpoint`。

## 6. Pilot 实测结果与两处配置缺陷（v26 → v26b）

> 本节是 v26 pilot 跑完之后补写的**复盘**。结论：v26 有两处必须修的配置缺陷，
> 但 pilot 的 −0.084 dB **不能**作为"全解冻方法无效"的证据。修正版是
> `configs/experiments/ssr_bfn_fullft_v26b.yaml`（30 epoch 主线）与
> `ssr_bfn_fullft_v26b_proxy.yaml`（6 epoch 判定实验）。
> `ssr_bfn_fullft_v26.yaml` / `ssr_bfn_fullft_v26_pilot.yaml` **配置值一律不改**，
> 只在文件顶部加了"已被取代"注记，作为失败记录保留。

### 6.1 实测事实（全部【已测量】，来源：服务器 pilot 运行日志，用户提供）

| 项目 | 实测值 |
|---|---|
| GPU 可用总显存 | **23.41 GiB**（不是设计 v26 时假设的 32 GB） |
| `micro_batch_size: 4` | **首次前向即 OOM**（256×256） |
| `micro_batch_size: 2` | 稳定跑完整 epoch |
| global batch / 梯度累积 | 仍 16 / 由 4 变 **8**（【代码事实】`train.py:421-430`） |
| 可训练参数 | **11,287,509** |
| 每 epoch micro-batch 数 | **8583**（= 17166 / 2） |
| 每 epoch optimizer step 数 | **1073**（= 17166 / 16，与 mb 无关） |
| 每 epoch 墙钟 | **3482.9 s = 0.968 h** |
| 30 epoch 主线成本 | ≈ **29.0 GPU 小时**【估算：0.968 × 30】 |
| 6 epoch proxy 成本 | ≈ **5.81 GPU 小时**【估算：0.968 × 6】 |

### 6.2 dev 对照（903 对持出 dev，同环境同口径）【已测量】

| 模型 | PSNR | SSIM |
|---|---:|---:|
| foundation（外部 SSR） | 34.76885 | 0.94353 |
| pilot `best_psnr.pt`（best_epoch = 2） | 34.68485 | 0.94330 |

逐图配对 bootstrap（10000 samples, seed 42）：

| 指标 | Δ | CI95 | improved |
|---|---:|---|---:|
| PSNR | **−0.083995** | [−0.097263, −0.070903] | **292 / 903** |
| SSIM | −0.000225 | 不含 0 | — |

### 6.3 诊断 A：像素项只占目标函数的 46.1%

【代码事实】`sgbformer/losses.py:174-183` 求的是**加权和**：

```python
total = (structural_weight * structural + bfn_weight * bfn
         + final_weight * final + correlation_weight * correlation
         + frequency_weight * frequency + adapter_weight * adapter
         + uncertainty_weight * uncertainty
         + latent_category_weight * category)
```

而【代码事实】`losses.py:184-194` 返回的诊断 dict 里各分量是**未加权的原始值**，
所以日志里的 `struct= / bfn= / freq= / latent_cls=` 必须手工乘权重才能看出占比。

用 pilot epoch 2 末段日志（`loss=0.00201 struct=0.00093 bfn=0.04451
final=0.00000 corr=0.00000 freq=0.00944 unc=0.00000 latent_cls=0.01701
lr=1.500e-05`）与 v26 权重（struct 1.0 / bfn 0.01 / freq 0.05 / cls 0.01）复算：

| 分量 | 权重 × 原始值 | 加权贡献 | 占比 |
|---|---|---:|---:|
| structural | 1.00 × 0.00093 | 9.300e-4 | **46.1%** |
| frequency | 0.05 × 0.00944 | 4.720e-4 | 23.4% |
| bfn | 0.01 × 0.04451 | 4.451e-4 | 22.1% |
| latent_category | 0.01 × 0.01701 | 1.701e-4 | 8.4% |
| 合计 | — | **2.0172e-3** | 100% |

合计 2.0172e-3 与日志 `loss=0.00201` 相差 7.2e-6。【保留意见】日志按 5 位小数打印，
五个数各自的显示舍入合起来约 ±1.05e-5，所以这是"在显示精度内吻合"，不是
逐位精确复现——但足以证明分解正确（否则不会落在 1e-5 量级）。

结论：**53.9% 的梯度信号不是 PSNR 所度量的像素保真度**；`frequency_weight: 0.05`
不是温和正则，而是与像素项同量级的第二目标。这是 −0.084 dB 最可能的直接原因。

### 6.4 诊断 B：2-epoch pilot 全程跑在峰值 LR，一次都没退火

【代码事实】`sgbformer/training.py:87-99`（`cosine_warmup_factor`，**按 epoch 步进**）：

```python
if warmup_epochs > 0 and epoch < warmup_epochs:
    return max((epoch + 1) / warmup_epochs, 1e-8)
cosine_epochs = max(total_epochs - warmup_epochs, 1)
progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
return minimum_ratio + (1.0 - minimum_ratio) * cosine
```

v26 是 `warmup_epochs: 1`、`minimum_learning_rate_ratio: 0.1`。代入：

| 运行 | 末 epoch | progress | cosine | 退火因子 | backbone LR |
|---|---|---:|---:|---:|---:|
| pilot（total 2） | 0 | warmup 分支 | — | **1.000000** | 1.500e-5 |
| pilot（total 2） | 1 | (1−1)/1 = 0 | 1.000000 | **1.000000** | 1.500e-5 |
| proxy（total 6） | 5 | 4/5 = 0.8 | 0.0954915 | **0.185942** | **2.7891e-6** |
| 主线（total 30） | 29 | 28/29 = 0.9655172 | 0.0029311 | **0.102638** | **1.5396e-6** |

pilot 两个 epoch 的因子都恰好是 1.0，与日志 `lr=1.500e-05` 恒定完全一致。
这四个数字在 `tests/test_fullft_v26_config.py::test_proxy_anneals_while_the_pilot_never_did`
里直接对 `cosine_warmup_factor` 断言，不是手算。

**结论（明确写出）：pilot 的 −0.084 dB 不能证明 v26 退化。**它是"全部解冻 + 100%
峰值 LR + 零退火"的两 epoch 快照；用它判定 v26 在方法上无效。真实的 30 epoch 主线
末端 backbone LR 只有 1.54e-6，是 pilot 的 1/10。

### 6.5 诊断 C：训练采样口径与 dev 评测口径错配

- 训练侧：`sampling.mode: category_balanced`，macro 口径
  Snow 0.50 / Outdoor-Rain 0.25 / RainDrop 0.25（v26 yaml:241-250）。
- 评测侧：dev 903 对是 18069 对的 5% **随机**抽样（【代码事实】
  `scripts/create_split_manifest.py`），因此是**自然分布**，约
  Snow 450 / Outdoor-Rain 412 / RainDrop 41【估算：按自然占比换算，实际计数以
  `scripts/analyze_dev_by_category.py` 输出的 `classification.counts` 为准】。
- 换算：Outdoor-Rain 在训练里从约 45.7% 降权到 25%（**0.55×**），RainDrop 从约
  4.53% 升权到 25%（**约 5.5×**）。

因此**自然分布加权的 dev 算术平均天然会下降**，哪怕官方 macro 指标是上升的：
被降权的 Outdoor-Rain 在 dev 里占 45.6% 的权重，被升权的 RainDrop 只占 4.5%。
判定必须改用 **macro 加权 dev**（类别层面 0.5 / 0.25 / 0.25），而不是 dev 均值。
这正是 `scripts/analyze_dev_by_category.py` 的 `macro_weighted` 字段。

### 6.6 v26b：改了什么、为什么

`configs/experiments/ssr_bfn_fullft_v26b.yaml`（`_base_: ssr_bfn_fullft_v26.yaml`）
只覆盖两组值，**其余全部继承**：

```yaml
training:
  output_dir: runs/fullft_v26b/main
  micro_batch_size: 2
loss:
  frequency_weight: 0.01
  bfn_weight: 0.004
  latent_category_weight: 0.002
```

新权重在 pilot 的原始分量上的贡献：

| 分量 | 权重 × 原始值 | 加权贡献 |
|---|---|---:|
| frequency | 0.010 × 0.00944 | 9.440e-5 |
| bfn | 0.004 × 0.04451 | 1.7804e-4 |
| latent_category | 0.002 × 0.01701 | 3.402e-5 |
| 辅助项合计 | — | **3.0646e-4**（= struct 9.300e-4 的 32.95%） |
| 总计 | 9.300e-4 + 3.0646e-4 | 1.23646e-3 |

像素项占比 9.300e-4 / 1.23646e-3 = **75.2%**（原 46.1%）。
【估算】权重一改，原始分量本身会漂移，所以 75.2% 是"按 pilot 实测分量做的一阶重加权"，
不是对新平衡点的预测；但比例是唯一能直接控制的量，且
`tests/test_fullft_v26_config.py::test_v26b_pixel_share_is_frozen_between_74_and_77_percent`
把这段算术固化成了测试区间 (0.74, 0.77)，防止以后有人再随手调权重。

辅助项**故意不置零**：BFN 通路必须保持被监督（它是 v26 唯一保留的论文级硬约束），
`latent_category_weight > 0` 才会喂类别头（【代码事实】`losses.py:161-171`
在缺 logits/labels 时直接抛错）。

`micro_batch_size: 2` 的依据见 6.1：mb4 实测 OOM；global batch 仍 16，梯度累积 4→8，
**学习率与调度不需要改**（`training.py:87-99` 按 epoch 步进，与 micro batch 无关）。

**唯一改动的变量组 = 损失权重 + micro batch。** backbone LR 仍 1.5e-5、采样仍
balanced、latent BFN 仍 steps 4 bayesian、仍持出 903 dev —— 这样 6 epoch proxy 的
结果才能归因到这一组改动上。
`tests/test_fullft_v26_config.py::test_v26b_changes_only_loss_weights_and_micro_batch`
用集合差断言 `training` 只有 `output_dir` / `micro_batch_size` 不同、`loss` 只有三个
权重不同。

### 6.7 判定流程与完整命令序列

环境变量沿用服务器上已有的那套：

```bash
cd /path/to/bfn/sgbformer
conda activate diffusionad
export SGB_DATA_ROOT=/path/to/allweather
export SSR_REPO=/path/to/SSR
export SSR_CHECKPOINT=$SSR_REPO/Allweather/pretrained_models/net_g_best.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 第 0 步（**零 GPU 成本，现在就先跑**）：把已有的 pilot dev JSON 按类别拆开

这一步不需要 GPU、不需要再跑任何前向，只读两个已经存在的评测 JSON：

```bash
# 把这两个变量指向已有的 dev 评测产物（foundation 34.76885 / pilot 34.68485 那两份）
export DEV_FOUNDATION_JSON=results/fullft_v26_pilot/dev_foundation.json
export DEV_PILOT_JSON=results/fullft_v26_pilot/dev_pilot_best.json

python3 scripts/analyze_dev_by_category.py \
  --baseline "$DEV_FOUNDATION_JSON" \
  --candidate "$DEV_PILOT_JSON" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --output results/fullft_v26_pilot/dev_by_category.json \
  --bootstrap-samples 10000 --seed 42
```

要看的输出：

1. `natural_mean.psnr.delta` 应复现 **−0.083995**，脚本会把这条自校验写进
   `natural_mean.reproduces_reported_delta.matches`（默认容差 1e-4，可用
   `--reported-delta-psnr` / `--reported-delta-tolerance` 改，加
   `--require-reported-delta` 则不吻合时退出码 2）。**这一项为 false 就先别看别的**，
   说明喂错了 JSON。
2. `categories.{snow,outdoor_rain,raindrop}` 的 `n` / `delta` / CI95 / `improved`：
   退化到底落在哪一类。如果 balanced sampling 起了作用，RainDrop 的 Δ 应当比
   Outdoor-Rain 好。
3. `macro_weighted.psnr.delta` 及其 CI95：这才是与训练采样口径一致的判定量。
4. `warnings`：RainDrop 只有约 41 张，必然触发 `n < 60` 的高方差警告。

【重要】`--pairs-file` 强烈建议给上：【代码事实】`sgbformer/data/paired.py:124`
存进 JSON 的 `images[].name` 是**去掉扩展名**的相对 stem，而三条类别正则
（`.*\.jpe?g` / `im_\d+_s\d+_a0[456]\.png` / `\d+_rain\.png`）都以扩展名结尾。
不给 `--pairs-file` 时脚本会按 `.png → .jpg → .jpeg` 的顺序补回扩展名（对
All-Weather 命名是唯一可解的），但那种模式下 snow 的 `.*\.jpe?g` 变成 catch-all，
非法文件名无法被发现，脚本会给出对应 warning。给了 `--pairs-file` 就是精确切分。

#### 第 1 步：启动 6 epoch proxy（约 5.8 GPU 小时）

```bash
mkdir -p runs
nohup python3 -u train.py \
  --config configs/experiments/ssr_bfn_fullft_v26b_proxy.yaml \
  --device cuda > runs/fullft_v26b_proxy.log 2>&1 &
tail -f runs/fullft_v26b_proxy.log
```

开跑后先核对三件事：banner 的 `trainable_params=11287509`（【代码事实】
`train.py:572`）、`micro_batch_size=2` 对应的 accumulation 为 8、以及
`sampling_epoch_targets` 是 50/25/25（【代码事实】`train.py:633-641`）。

#### 第 2 步：proxy 跑完后，**逐个** `epoch_000N.pt` 离线评测 + macro 加权选择

为什么不能直接用 `best_psnr.pt`：【代码事实】`train.py:986-1035` 用
`validation["psnr"]`（903 对 dev 的**自然分布算术平均**）选 best，而 `train.py` 在
指纹清单内**不能改**。所以必须离线绕过它：

```bash
RUN=runs/fullft_v26b/proxy
OUT=results/fullft_v26b_proxy
mkdir -p "$OUT"

for ckpt in "$RUN"/epoch_00*.pt; do
  tag=$(basename "$ckpt" .pt)          # epoch_0001 ... epoch_0006
  python3 evaluate.py --checkpoint "$ckpt" \
    --input-dir "$SGB_DATA_ROOT/input" --target-dir "$SGB_DATA_ROOT/gt" \
    --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
    --device cuda --test-y-channel --crop-border 0 \
    --quiet --json "$OUT/dev_${tag}.json"

  python3 scripts/analyze_dev_by_category.py \
    --baseline "$DEV_FOUNDATION_JSON" \
    --candidate "$OUT/dev_${tag}.json" \
    --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
    --output "$OUT/by_category_${tag}.json" \
    --bootstrap-samples 10000 --seed 42 \
    --no-reported-delta-check
done

# 按 macro 加权 ΔPSNR 排序，挑最好的 epoch
python3 - <<'PY'
import glob, json
rows = []
for path in sorted(glob.glob("results/fullft_v26b_proxy/by_category_*.json")):
    report = json.load(open(path))
    macro = report["macro_weighted"]["psnr"]
    natural = report["natural_mean"]["psnr"]
    rows.append((macro["delta"], macro["paired_image_bootstrap_ci95"],
                 natural["delta"], path))
for delta, ci, natural, path in sorted(rows, reverse=True):
    print(f"{path}: macro dPSNR={delta:+.6f} CI95=[{ci[0]:+.6f}, {ci[1]:+.6f}] "
          f"(natural dPSNR={natural:+.6f})")
PY
```

注意：`--pairs-file $SGB_DATA_ROOT/val_pairs.tsv` 必须和训练时的 `data.val` 完全一致，
否则 `paired_image_comparison` 会以 `evaluation: image sets differ` 报错；
`--test-y-channel` / `--crop-border 0` 必须与 foundation 那份 dev JSON 相同，
否则会以 `incompatible protocol` 报错（【代码事实】
`sgbformer/comparison.py:43-90`）。这两个"宁可报错也不静默"的闸门是刻意保留的。

### 6.8 go / no-go 阈值建议

判定量固定为 **`macro_weighted.psnr.delta` 及其 CI95**（vs foundation 的同一份 dev
JSON），取 6 个 epoch 里 macro 最好的那个 checkpoint：

| 情形 | 判定 | 动作 |
|---|---|---|
| CI95 **下界 > 0** | **GO** | 直接跑 30 epoch v26b 主线（29 h） |
| CI95 含 0，且 Δ > 0 | **弱 GO** | 值得跑主线：6 epoch 只退火到 18.6%，30 epoch 会退到 10.3%，还有余量 |
| CI95 含 0，且 −0.03 ≤ Δ ≤ 0 | **HOLD** | 先按 6.3 的算术再降 `frequency_weight`（0.01 → 0.005）或把 `parameter_lrs.backbone.model` 降到 5e-6，再跑一次 proxy；不要直接上主线 |
| CI95 **上界 < 0**（且 Δ < −0.03） | **NO-GO** | 全解冻这条路在当前 LR / 权重下确实伤基座，回到冻结家族或改用更低 backbone LR 重设计 |

为什么阈值必须建立在 CI95 的上界/下界上，而不是点估计：

1. 点估计 Δ 的量级（0.0x dB）和 bootstrap 宽度（pilot 是 ±0.013 dB，macro 加权会更宽）
   同阶。只看点估计等于在噪声里做决策。
2. **下界 > 0** 才是"确有提升"的证据；**上界 < 0** 才是"确有退化"的证据；两者之间
   只能说"这份 dev 分不出来"，此时应当让更便宜的一侧（继续训练 / 微调权重）承担举证责任。
3. bootstrap 只对"这两个 checkpoint 在这 903 张上的差"做推断，**不包含训练随机种子方差**
   （【代码事实】`comparison.py:224-227` 的 `limitation` 字段原文如此）。因此不要把
   CI95 当成"换个种子重训也会落在这里"。

**保留意见（必须一起报告）**：dev 里 RainDrop 只有约 41 张，却在 macro 加权里占 25%
的权重，所以 macro 加权的 CI95 **必然比自然分布的 CI 宽**（脚本的分层重采样会如实体现
这一点，并额外给出 `n < 60` 的高方差 warning）。macro 加权修正的是**口径偏差**，
它变不出不存在的图像，不能降低小类的采样噪声。如果最终结论卡在 RainDrop 上，
正确做法是在官方 RainDrop test_a（58 张）或更大的验证集上复核，而不是加大 bootstrap 次数。

### 6.9 J8：短跑质量判定下限

**J8（冻结规则）**：少于 4 epoch 的运行只能验证工程链路、显存和吞吐，不得用于
质量 go/no-go；质量 proxy 必须包含明确的学习率退火，并离线比较逐 epoch 的
macro-weighted dev 指标。

这是一条保守实验规则，不是“少于 4 epoch 数学上都不会退火”的断言。按当前
`warmup_epochs: 1`、`minimum_learning_rate_ratio: 0.1`：

- 2-epoch pilot 的两个因子确实均为 1.0，完全没有退火；
- 3-epoch schedule 的最后一个因子已经是 0.55，但只有一个退火点，不作为质量判定；
- 当前冻结的 6-epoch proxy 有四个下降点，最终因子为 0.185942。

因此 v26 的 2-epoch 结果保留为 OOM、吞吐和目标函数诊断证据；v26b 的质量判定只使用
6-epoch proxy，不能根据短跑结果临时延长或缩短该 schedule。

### 6.10 v26c：单变量撤销 balanced sampling

v26b 六个 checkpoint 均未通过 macro GO；最佳 epoch 6 为
`macro ΔPSNR = -0.019267`、CI95 `[-0.035477, -0.003354]`。类别拆解进一步定位到：

| epoch | Snow Δ / CI95 | Outdoor-Rain Δ / CI95 | RainDrop Δ / CI95 |
|---|---|---|---|
| 5 | +0.043039 / [+0.031610, +0.053988] | −0.138830 / [−0.154207, −0.123465] | −0.032129 / [−0.095742, +0.028445] |
| 6 | +0.045867 / [+0.033804, +0.058204] | −0.147522 / [−0.163562, −0.131644] | −0.021282 / [−0.080522, +0.036743] |

balanced epoch 只抽 4292 / 7839 个 Outdoor-Rain，却把 763 个 RainDrop 扩成 4291
次抽取。预期的 RainDrop 增益未出现（CI 含 0），Outdoor-Rain 则稳定大幅退化。
因此下一项因果问题是：**撤销重采样能否保留 Snow 增益并修复 Outdoor-Rain？**

冻结配置：

```text
configs/experiments/ssr_bfn_fullft_v26c_nobalance_proxy.yaml
```

它继承 v26b proxy，只覆盖：

```yaml
training:
  output_dir: runs/fullft_v26c/nobalance_proxy
  sampling:
    mode: shuffle
```

`latent_category_weight` 保持 0.002。同步取消 sampler 与 category loss 会同时改两个变量，
无法归因。由于配置采用递归 merge，resolved config 中仍能看到继承的
`category_patterns` / `target_fractions`；它们在 `mode: shuffle` 分支中不会被读取，
属于无效元数据而非第二个行为变化。契约测试逐键断言实际只变化 `sampling.mode` 与
`output_dir`。

启动命令：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python3 -u train.py \
  --config configs/experiments/ssr_bfn_fullft_v26c_nobalance_proxy.yaml \
  --device cuda > runs/fullft_v26c_nobalance_proxy.log 2>&1 < /dev/null &
tail -f runs/fullft_v26c_nobalance_proxy.log
```

banner 应不再打印 `sampling_source_counts` / `sampling_epoch_targets`，其余参数、损失权重、
global batch、六 epoch schedule 必须与 v26b proxy 一致。完成后仍逐个评测六个
`epoch_000N.pt`，用 §6.7 相同的 `analyze_dev_by_category.py` 流程按
`macro_weighted.psnr.delta` 选择。只有 CI 下界 > 0 或弱 GO 才允许进入 30 epoch；
否则 balanced sampling 不是唯一瓶颈，下一项单变量才是降低 backbone LR。

### 6.11 v26d：单变量降低 backbone LR

v26c 与 v26b 的同 epoch、逐图配对结果：

| epoch | macro Δ（shuffle − balanced） | CI95 | natural Δ |
|---|---:|---|---:|
| 5 | +0.001010 | [−0.010259, +0.011473] | +0.005933 |
| 6 | −0.002325 | [−0.012156, +0.006477] | +0.004498 |

两个 macro CI 均覆盖 0，因此诚实结论是：**未见 sampling mode 对目标 macro 指标的
差异**。shuffle 在 epoch 5 的 natural mean 有小幅正差，但那不是主判定口径。
v26c 自身最佳 macro 仍为 `−0.020210`，CI95 `[−0.034048, −0.006382]`，没有通过
主线门禁。

下一项单变量回到 v26b balanced 配方，仅降低全解冻 SSR 的学习率：

```text
configs/experiments/ssr_bfn_fullft_v26d_lrlow_proxy.yaml
```

唯一行为覆盖：

```yaml
training:
  output_dir: runs/fullft_v26d/lrlow_proxy
  parameter_lrs:
    backbone.model: 0.000005
```

新模块 LR 仍为 `2e-4`；损失、采样、模型、seed、batch 和六 epoch schedule 全部不变。
末 epoch 的 backbone LR 为 `5e-6 × 0.185942 = 9.2971e-7`。

启动命令：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python3 -u train.py \
  --config configs/experiments/ssr_bfn_fullft_v26d_lrlow_proxy.yaml \
  --device cuda > runs/fullft_v26d_lrlow_proxy.log 2>&1 < /dev/null &
tail -f runs/fullft_v26d_lrlow_proxy.log
```

banner 的 optimizer LR 应为 `5.000e-06,2.000e-04`，其它字段与 v26b proxy 相同。
完成后沿用 §6.7 的逐 epoch 离线评测和 macro-weighted 选择。若最佳 macro CI 仍全负，
则当前“完整 SSR 全解冻”主线没有继续投入 30 epoch 的证据；不得再通过同时改多个变量
追逐结果。

## 7. OOM 应急

`micro_batch_size: 4 → 2`，其他都不动：

```yaml
training:
  micro_batch_size: 2
```

【代码事实】`train.py:421-430`：`accumulation = batch_size / (micro * world)`，
所以 4→2 只是把梯度累积从 4 步变成 8 步，**global batch 仍是 16**，
每 epoch 的 optimizer step 仍约为 1073，学习率与调度无需改动；
代价是 epoch 墙钟时间变长。若仍不够，再降 `data.workers` 或
`data.train.patch_size`（后者会改变训练分布，属于新实验，需重新披露）。

## 8. 风险与判定口径

1. **全解冻有退化风险**。SSR 已在同一训练集上收敛，1.5e-5 的基座学习率仍可能破坏
   预训练解；**必须逐 epoch 看 dev**（`validate_interval: 1`），dev 连续两个 epoch
   低于起点就停下来，先降 `parameter_lrs.backbone.model`（例如 1.5e-5 → 5e-6）。
2. **判定基准是 34.239638（foundation），不是 MODEM 的 34.18**。超过 MODEM 但没超过
   自己复现的 foundation，不构成"我们的方法有效"。
3. **self-ensemble 与权重平均必须单独成行披露**，不能混进主结果；两者都不改变模型，
   属于测试期/检查点后处理。
4. v26 没有 tripwire、没有预算锁、没有预注册，**它是探索性主线**；如果最终要写进论文，
   需要在论文中如实说明这一点（对比 D/B 的预注册协议）。
5. `--self-ensemble` 的成本是 8× 推理时间，`runtime` 字段会如实反映。
6. **本节第 1、2 条的"看 dev"口径已被 §6.5 修正**：`train.py` 打印和用于选 best 的
   `validation.psnr` 是 903 对 dev 的自然分布算术平均，与训练侧 50/25/25 的 macro
   采样口径不一致。逐 epoch 监控可以继续看它的**趋势**（是否崩），但**任何 go/no-go
   判定都必须改用 `scripts/analyze_dev_by_category.py` 的 `macro_weighted.psnr`
   及其 CI95**，阈值见 §6.8。

## 9. 预期增量（诚实口径，**不保证**）

| 来源 | 预期增量 | 置信度 | 依据 |
|---|---:|---|---|
| 全解冻 + 更大 BFN/adapter（206,869 → 461,829 新参数） | +0.05 ~ +0.25 dB | 中 | 【假设】SSR 已在同一训练集收敛，增量主要来自新增条件通路与基座微调的协同，而非数据不足 |
| category_balanced 采样（50/25/25 对齐 macro 口径） | +0.02 ~ +0.05 dB | 低–中 | 【假设】RainDrop 仅占训练集 4.53% 但占 macro 权重 25%，重采样理论上改善弱类；后续 v26c 的直接配对比较未见 macro 差异 |
| ×8 self-ensemble | +0.05 ~ +0.15 dB | 中 | 【假设】图像复原中 ×8 几何 TTA 的常见幅度，本仓库未实测 |
| 逐 epoch 权重平均（末 3–5 个快照） | +0.01 ~ +0.05 dB | 低 | 【假设】EMA 已在训练中生效，平均的边际收益会被压缩 |

以上四项**不可简单相加**，也**不保证**任一项为正。唯一已测量的锚点是
foundation 的 34.239638 / 0.062147【已测量，归档 adapter-only 评测】与冻结家族全部为
负的 −0.0336 ~ −0.0766 dB【已测量，同一归档记录】。
