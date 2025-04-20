import os
import re
import glob
import torch
import numpy as np
import yaml
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict
from typing import List, Dict, Tuple

from load_files import load_npz_files
from preprocess import preprocess
from models import MambaSS1D, PositionalEncoding

# 定义睡眠分期分类器

class SleepStageClassifier(nn.Module):
    def __init__(self, time_steps=3000, hidden_dim=64, num_classes=5, num_blocks=2):
        super(SleepStageClassifier, self).__init__()
        
        # 时间维度特征提取（每个时间步的特征）
        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 位置编码增强时序信息
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=time_steps//2)
        
        # MambaSS1D 序列处理模块
        self.sequence_blocks = nn.ModuleList([
            MambaSS1D(
                d_model=hidden_dim,
                d_state=16,
                d_conv=3,
                expand=2.0,
                dropout=0.1
            ) for _ in range(num_blocks)
        ])
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x 输入形状: [B, N, W, 1]
        """
        # 合并N和B维度
        batch_size, N, W, C = x.shape
        x = x.view(batch_size*N, W, C)  # [B*N, W, 1]
        
        # 时间维度特征提取
        x = x.permute(0, 2, 1)  # [B*N, 1, W]
        x = self.time_feature_extractor(x)  # [B*N, hidden_dim, W/2]
        
        # 转换为序列处理格式并添加位置编码
        x = x.permute(0, 2, 1)  # [B*N, W/2, hidden_dim]
        x = self.pos_encoder(x)  # 添加位置编码
        
        # 通过MambaSS1D块处理序列
        for block in self.sequence_blocks:
            x = block(x)  # [B*N, W/2, hidden_dim]
        
        # 全局池化获取序列特征
        x = x.permute(0, 2, 1)  # [B*N, hidden_dim, W/2]
        x = self.global_pool(x)  # [B*N, hidden_dim, 1]
        x = x.squeeze(-1)  # [B*N, hidden_dim]
        
        # 分类
        x = self.classifier(x)  # [B*N, num_classes]
        
        # 恢复原始维度
        x = x.view(batch_size, N, -1)  # [B, N, num_classes]
        return x

# 自定义数据集
class SleepDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 自定义collate函数处理变长序列
def collate_fn(batch):
    data, labels = zip(*batch)
    # 找出每个序列的实际长度
    lengths = [x.shape[0] for x in data]
    max_len = max(lengths)
    
    # 创建批次张量并填充
    batch_size = len(data)
    time_steps = data[0].shape[1]
    channels = data[0].shape[2]
    
    # 填充样本到相同长度
    padded_data = torch.zeros(batch_size, max_len, time_steps, channels)
    for i, (x, length) in enumerate(zip(data, lengths)):
        padded_data[i, :length] = x
    
    # 处理不同格式的标签
    label_list = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            label_list.append(label.item())
        else:
            label_list.append(int(label))
    
    # 将标签转换为张量
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    return padded_data, label_tensor

# 按受试者ID分组数据文件
def group_by_subject(npz_files):
    subjects = defaultdict(list)
    for f in npz_files:
        # 提取受试者ID
        match = re.search(r'SC4(\d+)[12][EFG]0\.npz', os.path.basename(f))
        if match:
            subject_id = int(match.group(1))
            subjects[subject_id].append(f)
    return subjects

# K折交叉验证准备
def prepare_kfold_data(npz_files, k=5):
    subjects = group_by_subject(npz_files)
    subject_ids = list(subjects.keys())
    
    # 创建K折分割
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    
    for train_idx, test_idx in kf.split(subject_ids):
        train_subjects = [subject_ids[i] for i in train_idx]
        test_subjects = [subject_ids[i] for i in test_idx]
        
        train_files = []
        test_files = []
        
        for s in train_subjects:
            train_files.extend(subjects[s])
        for s in test_subjects:
            test_files.extend(subjects[s])
        
        folds.append((train_files, test_files))
    
    return folds

def get_class_weights(device):
    """
    获取各睡眠阶段的权重
    """
    # 基于典型睡眠分布设置权重
    weights = torch.tensor([
        1.0,  # W  - 清醒阶段
        1.8,  # N1 - 浅睡眠1（最稀有）
        1.0,  # N2 - 浅睡眠2（最常见）
        1.25,  # N3 - 深睡眠
        1.20   # REM - 快速眼动睡眠
    ], device=device)
    
    # 归一化权重
    weights = weights / weights.sum() * 5
    print(f"使用睡眠阶段权重: {weights}")
    
    return weights

# 训练函数
def train_model(model, train_loader, val_loader, device, params):
    # 获取类别权重
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    best_f1 = 0.0
    patience_counter = 0
    
    accumulation_steps = 4  # 每4步更新一次参数
    
    for epoch in range(params['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(batch_data)
                # 只取每个序列最后一个时间步的预测
                last_outputs = outputs[:, -1, :]
                loss = criterion(last_outputs, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                # 只取每个序列最后一个时间步的预测
                last_outputs = outputs[:, -1, :]
                loss = criterion(last_outputs, batch_labels)
                val_loss += loss.item()
                
                _, preds = torch.max(last_outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 打印结果
        print(f'Epoch {epoch+1}/{params["epochs"]}')
        print(f'训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}')
        print(f'验证准确率: {val_acc:.4f} | 验证F1分数: {val_f1:.4f}')
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print("混淆矩阵:")
        print(cm)
        
        # 学习率调整
        scheduler.step(val_f1)
        
        # 早停
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), params['model_save_path'])
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                print(f'早停：验证F1分数已经{params["patience"]}个周期没有提高')
                break
    
    return best_f1

# 主函数
def main():
    # 加载配置
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        config = yaml.full_load(f)
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = config['data']['data_dir']
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"找到 {len(npz_files)} 个数据文件")
    
    # 准备K折数据
    folds = prepare_kfold_data(npz_files, k=config['training']['k_folds'])
    
    # K折交叉验证
    fold_f1_scores = []
    
    for fold_idx, (train_files, test_files) in enumerate(folds):
        print(f"\n训练第 {fold_idx+1} 折...")
        print(f"训练文件数: {len(train_files)} | 测试文件数: {len(test_files)}")
        
        # 加载数据
        train_data, train_labels = load_npz_files(train_files)
        test_data, test_labels = load_npz_files(test_files)
        
        # 数据预处理
        train_samples, train_sample_labels = preprocess(
            train_data, train_labels, config['preprocess'], not_enhance=False
        )
        test_samples, test_sample_labels = preprocess(
            test_data, test_labels, config['preprocess'], not_enhance=True
        )
        
        # 创建数据加载器
        train_dataset = SleepDataset(train_samples, train_sample_labels)
        test_dataset = SleepDataset(test_samples, test_sample_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 创建模型
        model = SleepStageClassifier(
            time_steps=config['model']['time_steps'],
            hidden_dim=config['model']['hidden_dim'],
            num_classes=config['model']['num_classes'],
            num_blocks=config['model']['num_blocks']
        ).to(device)
        
        # 设置保存路径
        model_save_path = f"models/model_fold_{fold_idx+1}.pt"
        config['training']['model_save_path'] = model_save_path
        
        # 训练模型
        fold_f1 = train_model(
            model, train_loader, test_loader, device, config['training']
        )
        
        fold_f1_scores.append(fold_f1)
        print(f"第 {fold_idx+1} 折最佳F1分数: {fold_f1:.4f}")
    
    # 打印交叉验证结果
    print("\nK折交叉验证结果:")
    for i, f1 in enumerate(fold_f1_scores):
        print(f"第 {i+1} 折 F1: {f1:.4f}")
    print(f"平均F1分数: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")

if __name__ == "__main__":
    main()