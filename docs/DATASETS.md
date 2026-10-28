# 数据来源与准备

## 公开数据来源

BasisBFN 使用的 All-Weather 训练合集由
[TransWeather 官方仓库](https://github.com/jeya-maria-jose/TransWeather#datasets)
发布。其 README 明确说明：合集从 Outdoor-Rain、Snow100K 和 RainDrop
数据集中采样，包含 **18,069 对**图像，沿用 All-in-One 的混合天气训练设置。

| 数据资源 | 发布入口 | 本项目用途 |
| --- | --- | --- |
| All-Weather 训练合集 | [官方 Google Drive 数据包](https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view) | 准备训练／验证图像对 |
| 上游文件列表 | [官方 Google Drive 列表包](https://drive.google.com/file/d/1UsazX-P3sPcDGw3kxkyFWqUyNfhYN_AM/view) | 查看上游图像路径与文件组织 |
| Snow100K | [DesnowNet 作者项目页](https://sites.google.com/view/yunfuliu/desnownet) | 雪类来源；正式测试使用 Snow100K-S 和 Snow100K-L |
| RainDrop | [DeRaindrop 官方数据说明](https://github.com/rui1996/DeRaindrop#dataset) | 雨滴类来源；正式测试使用 Test-A |
| Outdoor-Rain | [TransWeather 的合集来源说明](https://github.com/jeya-maria-jose/TransWeather#datasets) | 户外雨类来源；正式测试使用 Test1 |

All-in-One 的基准设置可参考
[CVPR 2020 原论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf)。
以上链接指向上游发布资源；本仓库不包含数据集图像。

## 下载与解压

安装本项目后，额外安装下载工具 `gdown`：

```bash
python3 -m pip install gdown

DOWNLOAD_DIR=/path/to/downloads \
DATASET_DIR=/path/to/allweather_archive \
  bash scripts/download_all_weather.sh
```

脚本下载上述两个资源并解压到 `DATASET_DIR`。它会查找上游
`allweather.txt`，但**不会自动生成本项目的训练／验证 TSV 清单**。
脚本输出的上游目录仍需按下节检查、整理，再设置 `SGB_DATA_ROOT`。

此脚本下载的是上游 All-Weather 资源，不能据此假定四个正式测试子集都已经
下载完毕。正式测试数据应从相应上游资源单独确认并整理。

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
需要按上游映射整理路径，或自行写出正确的两列 TSV；该脚本不会推断此类映射。

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
验证数量为 `round(N × 0.05)`。当 `N=18,069` 时，生成 903 对验证数据和
17,166 对训练数据。已有清单默认不会被覆盖。

**原实验的两份 TSV 清单及其 SHA-256 校验值目前没有发布。**上述命令是
重新生成划分的方法，尚不能证明输出与原实验逐图一致。若要精确复现已有
实验，仍需取得原始清单，或用原始校验值核对重新生成的清单。

划分种子与模型训练种子是两项不同设置：四个模型训练种子
`42/43/44/45` 应共用同一份已固定的数据划分。原始上游文件列表也不能直接
替代本项目的 `train_pairs.tsv`、`val_pairs.tsv`。

可记录准备好的清单哈希：

```bash
shasum -a 256 "$SGB_DATA_ROOT/train_pairs.tsv" \
  "$SGB_DATA_ROOT/val_pairs.tsv"
```

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

RainDrop Test-A 是 DeRaindrop 官方定义的定量评测子集，应保持原有划分。
四个正式测试子集不参与训练／验证划分，也不用于选择 checkpoint。
当前协议使用固定 epoch-6 EMA checkpoint。

## 准备后的检查

生成某个 arm／seed 的配置后，先检查数据配对：

```bash
python3 scripts/check_dataset.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml
```

该命令是数据完整性检查；训练前仍需执行 README 中的 BasisBFN preflight
和 200-step CUDA profile。仓库不包含数据图像、原实验划分清单或预训练权重，
也没有随本次文档更新执行完整数据下载或正式测试评测。
