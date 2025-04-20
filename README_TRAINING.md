 # Mamba模型分布式训练指南

本文档介绍如何使用优化后的分布式训练脚本，采用混合精度训练加速Mamba模型训练。

## 功能特性

- **分布式训练**：支持多GPU训练，显著提高训练速度
- **混合精度训练**：使用`torch.cuda.amp`进行混合精度训练，减少显存占用并提高计算速度
- **优化的数据加载**：通过配置适当的`num_workers`提高数据加载效率
- **方便的监控**：使用screen会话后台运行，可随时查看训练进度

## 快速开始

### 1. 准备环境

确保已安装以下依赖：

```bash
pip install torch torchvision tqdm
```

### 2. 修改训练配置

根据您的需求修改`run_training.sh`中的参数：

- `CUDA_VISIBLE_DEVICES`: 设置要使用的GPU ID
- `NUM_GPUS`: 使用的GPU数量
- `BATCH_SIZE`: 每个GPU的批次大小 
- `EPOCHS`: 训练轮数
- `DATA_DIR`: 数据集路径

### 3. 启动训练

```bash
# 给脚本添加执行权限
chmod +x run_training.sh

# 运行脚本
./run_training.sh
```

脚本会在screen会话中启动训练，并显示会话名称。

### 4. 监控训练

```bash
# 查看所有screen会话
screen -ls

# 连接到指定会话
screen -r 会话名称

# 分离会话（继续在后台运行）
# 按 Ctrl+A, 然后按 D
```

## 训练参数调整

您可以通过修改`train_distributed.py`中的参数来进一步优化训练：

```bash
python train_distributed.py --help
```

主要参数：

- `--batch-size`: 每个GPU的批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--num-workers`: 数据加载的worker数量
- `--in-channels`: 输入通道数
- `--num-classes`: 类别数量

## 检查点和恢复训练

训练过程会自动保存以下检查点：

- `latest.pth`: 最新的检查点
- `best.pth`: 验证集上性能最好的检查点
- `epoch_X.pth`: 每隔几个epoch保存的检查点

要从检查点恢复训练：

```bash
torchrun --nproc_per_node=4 train_distributed.py --resume ./checkpoints/latest.pth
```

## 性能优化提示

1. **增加批次大小**：如果GPU显存允许，可以增加批次大小提高训练效率
2. **调整num_workers**：通常设置为CPU核心数的2-4倍效果最佳
3. **检查数据加载瓶颈**：使用`nvtop`或`nvidia-smi`监控GPU利用率，如低于80%可能存在数据加载瓶颈

## 常见问题

**Q: 如何调整学习率以适应不同的总批次大小？**

A: 当增加GPU数量时，总批次大小也会增加。通常可以使用线性缩放规则：`new_lr = original_lr * (new_batch_size / original_batch_size)`

**Q: 如何提高数据加载速度？**

A: 尝试以下方法：
- 增加`num_workers`
- 使用`pin_memory=True`
- 考虑预处理数据并缓存到磁盘
- 使用更快的存储介质（如SSD/NVMe）

**Q: 如何解决"CUDA out of memory"错误？**

A: 尝试以下方法：
- 减小批次大小
- 启用梯度累积（在代码中添加）
- 使用`torch.cuda.empty_cache()`清理未使用的缓存