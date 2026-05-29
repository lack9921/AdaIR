# Elvis 5-Degradation Training — AdaIR 适配指南

## 概述

本适配在保持 AdaIR 原始代码完整性的前提下，新增了 `--elvis_mode` 训练模式，支持使用
**5 种退化类型**（Blur / Haze / Lowlight / Rain / Snow）训练 5 合 1 模型。

## 改动文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `options.py` | 新增 3 个参数 | Elvis 模式开关、数据路径、验证集大小 |
| `utils/dataset_utils.py` | 扩展 `AdaIRTrainDataset` | 统一加载 GT/LQ 配对 + train/val 切分 |
| `train.py` | 新增 `validation_step` | 每 epoch 计算 val_loss / PSNR / SSIM |

**AdaIR 网络模型（`net/model.py`）和原始 7 种退化训练逻辑完全未动。**

## 数据目录结构

解压后目录结构如下：

```
package/
├── AdaIR/             ← 代码（cd 到这里运行）
└── train/             ← 训练数据
    ├── Blur/…GT+LQ/
    ├── Haze/…GT+LQ/
    ├── Lowlight/…GT+LQ/
    ├── Rain/…GT+LQ/
    └── Snow/…GT+LQ/
```

每种退化 **GT = 还原真值，LQ = 低质量输入**。两种子目录内的文件名一一对应：

```
Blur/GT/0001.jpg ~ 4900.jpg
Blur/LQ/0001.jpg ~ 4900.jpg
```

按文件名数字排序后：
- 前 4850 张 -> 训练集
- 后 50 张 -> 验证集

## 启动训练

### 默认 Elvis 模式（5 种退化 + 验证）

```bash
cd AdaIR
python train.py --elvis_mode --epochs 150 --batch_size 8 --num_gpus 1
```

`--elvis_mode` 会自动：
1. 把 `de_type` 切换到 `['blur', 'haze', 'lowlight', 'rain', 'snow']`
2. 创建验证集 DataLoader
3. 每 epoch 记录 `val_loss` / `val_psnr` / `val_ssim`

### 可调参数

```bash
# 修改验证集大小（每类取后 N 张）
python train.py --elvis_mode --elvis_val_last_n 100 --num_gpus 1

# 指定数据目录（如果 train 不在 AdaIR 上级目录）
python train.py --elvis_mode --elvis_train_dir "/some/other/path/train" --num_gpus 1

# 恢复原始 AdaIR 7 种退化训练
python train.py
```

### 全部参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--elvis_mode` | `False` | 启用 Elvis 5 退化训练 |
| `--elvis_train_dir` | `../train` | 训练数据根目录（相对 AdaIR/） |
| `--elvis_val_last_n` | `50` | 每类最后 N 张做验证 |
| `--de_type` | 原 7 种 | Elvis 模式下自动覆盖 |
| `--epochs` | `150` | 训练轮数 |
| `--batch_size` | `8` | 每 GPU 批次大小 |
| `--patch_size` | `128` | 随机裁剪大小 |
| `--num_workers` | `16` | 数据加载线程数 |
| `--num_gpus` | `4` | GPU 数量 |
| `--ckpt_dir` | `AdaIR` | 检查点保存目录 |
| `--wblogger` | `AdaIR` | wandb 项目名（不传则用 TensorBoard） |

## 查看训练过程

### TensorBoard

训练日志默认写入 `logs/`，可以用 TensorBoard 查看：

```bash
tensorboard --logdir logs/
```

### 记录的指标

| 指标 | 来源 | 频率 |
|------|------|------|
| `train_loss` | `training_step` | 每步 |
| `val_loss` | `validation_step` | 每 epoch |
| `val_psnr` | `validation_step` | 每 epoch |
| `val_ssim` | `validation_step` | 每 epoch |

### 检查点

每 epoch 自动保存一次检查点到 `--ckpt_dir` 指定的目录，Lightning 的 `ModelCheckpoint` 默认保存全部 epoch。

## 恢复原始 AdaIR 训练

本适配对原始训练路径无影响：

```bash
# 照常训练原 7 种退化
python train.py --de_type denoise_15 denoise_25 denoise_50 derain dehaze deblur enhance --epochs 150 --batch_size 8 --num_gpus 1
```

`--elvis_mode` 不加时，数据集和训练逻辑与 AdaIR 原版完全一致。
