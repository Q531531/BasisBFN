# 数据准备

All-Weather 训练合集包含 Outdoor-Rain、Snow100K 和 RainDrop，共
**18,069 对**图像。本页说明目录整理、图像配对、训练／验证划分及测试布局。

## 下载与解压

安装本项目后，额外安装下载工具 `gdown`：

```bash
python3 -m pip install gdown

DOWNLOAD_DIR=/path/to/downloads \
DATASET_DIR=/path/to/allweather_archive \
  bash scripts/download_all_weather.sh
```

脚本下载图像数据与文件列表并解压到 `DATASET_DIR`。它会查找
`allweather.txt`，但**不会自动生成本项目的训练／验证 TSV 清单**。
脚本输出的目录仍需按下节检查、整理，再设置 `SGB_DATA_ROOT`。

此脚本准备 All-Weather 训练数据，不能据此假定四个正式测试子集都已经
下载完毕。正式测试数据需单独确认并整理。

## 图像目录与配对规则

当前配置使用以下布局：

```text
allweather/
├── input/              # 退化图像
├── gt/                 # 干净目标图像
├── train_pairs.tsv
└── val_pairs.tsv
```

```bash
export SGB_DATA_ROOT=/path/to/allweather
```

每份 TSV 无表头，每行用一个制表符分隔两列：

```text
退化图像相对路径<TAB>目标图像相对路径
```

第一列相对于 `input/`，第二列相对于 `gt/`。两列允许文件名不同；
每对图像应尺寸相同、内容对应。应保留原始配对关系，避免仅因名字相似就配对。

[`create_split_manifest.py`](../scripts/create_split_manifest.py) 的简便生成路径
要求 `input/` 和 `gt/` 下的图像具有**相同相对路径**。若原始文件名不同，
需要按原始配对映射整理路径，或自行写出正确的两列 TSV；该脚本不会推断此类映射。

## 训练／验证划分与复现范围

当前 BasisBFN 配置的固定数量要求为：

| 划分 | 图像对数 |
| --- | ---: |
| 训练 | 17,166 |
| 验证 | 903 |
| 合计 | 18,069 |

在满足上述同路径配对条件后，可以生成一个确定性划分：

```bash
python3 scripts/create_split_manifest.py \
  --root "$SGB_DATA_ROOT" \
  --validation-ratio 0.05 \
  --seed 42
```

脚本先按相对路径排序，再使用 Python `random.Random(42)` 打乱；
验证数量为 `round(N × 0.05)`，并保证训练和验证各至少一对；输入不足两对
时拒绝划分。当 `N=18,069` 时，生成 903 对验证数据和 17,166 对训练数据。
已有清单默认不会被覆盖。

发布包不分发原实验的两份 TSV 清单。上述命令生成确定性划分；精确复现
论文的逐图划分需要原始清单。图像对数量一致不足以证明逐图划分一致。

划分种子与模型训练种子是两项不同设置：四个模型训练种子
`42/43/44/45` 应共用同一份已固定的数据划分。原始文件列表也不能直接
替代本项目的 `train_pairs.tsv`、`val_pairs.tsv`。

## 四个正式测试子集

`ALLWEATHER_TEST_ROOT` 的布局与仓库中
[`check_allweather_test.py`](../scripts/check_allweather_test.py) 保持一致：

```text
testallweather/
├── snow100k/
│   ├── Snow100K-S/
│   │   ├── synthetic/
│   │   └── gt/
│   └── Snow100K-L/
│       ├── synthetic/
│       └── gt/
├── Test1/
│   ├── input/
│   └── gt/
└── raindrop/
    └── test_a/
        ├── data/
        └── gt/
```

```bash
export ALLWEATHER_TEST_ROOT=/path/to/testallweather
python3 scripts/check_allweather_test.py --root "$ALLWEATHER_TEST_ROOT"
```

RainDrop Test-A 用于定量评测，应保持原有划分。
四个正式测试子集不参与训练／验证划分，也不用于选择 checkpoint。
当前协议使用固定 epoch-6 EMA checkpoint。

## 准备后的检查

生成某个 arm／seed 的配置后，先检查数据配对：

```bash
python3 scripts/check_dataset.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml
```

该命令是数据完整性检查；训练前仍需执行 README 中的 BasisBFN preflight
和 200-step CUDA profile。仓库不包含数据图像、原实验划分清单或预训练权重。
