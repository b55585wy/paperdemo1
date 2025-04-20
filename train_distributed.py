 
 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# 根据您的模型导入
try:
    from cnnmamba import VSSM
except ImportError:
    from models import MambaSS1D

def setup_logging(log_file=None):
    """设置日志格式，同时输出到文件和控制台"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )

def setup_distributed(backend='nccl'):
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logging.info("环境变量未设置，假设这是单GPU训练")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def get_model(args, device):
    """创建并返回模型"""
    # 根据您的模型类型选择正确的初始化
    try:
        model = VSSM(
            num_classes=args.num_classes,
            in_chans=args.in_channels,
            depths=args.depths
        )
    except:
        model = MambaSS1D(
            d_model=args.d_model,
            d_state=args.d_state,
            dropout=args.dropout
        )
    
    return model.to(device)

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, world_size):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not (dist.get_rank() == 0))
    
    for batch_idx, (inputs, targets) in enumerate(progress):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            if hasattr(model.module, 'training') and callable(getattr(model.module, 'training')):
                outputs, aux_outputs = model(inputs)
                loss = criterion(outputs, targets)
                # 处理辅助损失
                if aux_outputs is not None:
                    aux_loss = 0
                    for aux_out in aux_outputs:
                        aux_loss += criterion(aux_out, targets)
                    loss += 0.4 * aux_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        
        # 使用scaler缩放损失并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        progress.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # 汇总统计信息
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    epoch_time = time.time() - start_time
    
    # 只在主进程打印
    if dist.get_rank() == 0:
        logging.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {epoch_time:.2f}s")
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    if dist.get_rank() == 0:
        logging.info(f"Validation | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, acc, best_acc, args, is_best=False):
    """保存检查点"""
    if dist.get_rank() != 0:
        return
    
    state = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'best_acc': best_acc
    }
    
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存最近的检查点
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, latest_path)
    
    # 如果是最佳模型，也保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)
        logging.info(f"保存最佳模型，准确率: {acc:.2f}%")
    
    # 每隔几个epoch保存一次
    if (epoch + 1) % args.save_interval == 0:
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save(state, epoch_path)

def main():
    parser = argparse.ArgumentParser(description='PyTorch分布式训练')
    # 数据集参数
    parser.add_argument('--data-dir', default='./data', help='数据集路径')
    parser.add_argument('--num-classes', type=int, default=4, help='类别数量')
    parser.add_argument('--in-channels', type=int, default=1, help='输入通道数')
    
    # 模型参数
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 9, 2], help='各阶段的深度')
    parser.add_argument('--d-model', type=int, default=512, help='模型维度（用于MambaSS1D）')
    parser.add_argument('--d-state', type=int, default=16, help='状态空间维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='每个GPU的批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载的worker数量')
    
    # 检查点参数
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='检查点保存路径')
    parser.add_argument('--resume', default='', help='恢复训练的检查点路径')
    parser.add_argument('--save-interval', type=int, default=5, help='保存检查点的epoch间隔')
    
    # 日志参数
    parser.add_argument('--log-dir', default='./logs', help='日志保存路径')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file)
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    # 此处需要根据您的实际数据集替换
    train_dataset = None  # 替换为您的实际数据集
    val_dataset = None  # 替换为您的实际数据集
    
    # 如果您没有具体的数据集实现，可以先注释这部分，稍后添加
    if train_dataset is None or val_dataset is None:
        logging.warning("请实现您的数据集加载逻辑")
        # 以下为示例占位符
        from torch.utils.data import TensorDataset
        dummy_data = torch.randn(1000, args.in_channels, 224, 224)
        dummy_labels = torch.randint(0, args.num_classes, (1000,))
        train_dataset = TensorDataset(dummy_data, dummy_labels)
        val_dataset = TensorDataset(dummy_data[:100], dummy_labels[:100])
    
    # 配置分布式采样器
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = get_model(args, device)
    
    # 分布式包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    # 恢复训练（如果提供了检查点）
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            logging.info(f"恢复训练从epoch {start_epoch}，最佳准确率: {best_acc:.2f}%")
        else:
            logging.error(f"未找到检查点: {args.resume}")
    
    # 打印模型架构摘要
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"模型总参数: {total_params/1e6:.2f}M, 可训练参数: {trainable_params/1e6:.2f}M")
        logging.info(f"训练使用的批次大小: {args.batch_size * world_size} (每个GPU {args.batch_size})")
    
    # 训练循环
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, world_size
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 保存检查点
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(
            model, optimizer, epoch, val_acc, best_acc, args, is_best
        )
    
    # 训练完成
    total_time = time.time() - start_time
    if rank == 0:
        logging.info(f"训练完成，总时间: {total_time/3600:.2f}小时, 最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()