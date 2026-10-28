# 开发指南

## 本地验证

从仓库根目录执行：

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/smoke_basisbfn.py --device cpu
python3 -m pytest -q
```

测试和模块自检无需 SSR 权重、训练数据或 GPU。完整训练需要另外准备
[数据与划分](docs/DATASETS.md)以及外部 SSR。

## 提交变更

先运行与改动相关的验证，再运行全量回归，随后提交 Pull Request。
PR 应说明具体行为变化、运行的验证及其限制。GitHub Actions 会检查 CPU
自检、回归、主要命令入口和 wheel 构建。

修改实验配置或核心实现后，真实训练应重新生成匹配的 preflight／profile，
不能沿用旧指纹产物。

## 问题反馈

请提供代码 commit、Python／PyTorch／CUDA 版本、运行命令、错误信息和相关
配置。无需提供访问令牌、数据集图像或预训练权重。
