import os
import re
import glob
import shutil
import logging
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import sys

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from preprocess import preprocess, normalization
from load_files import load_npz_files
from models import ConbimambaBlock
from dataset import SleepDataset
from custom_collate_fn import custom_collate_fn

def gpu_settings():
    """
    Configure GPU settings for PyTorch.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    return torch.device('cpu')


def get_parser() -> argparse.Namespace:
    """
    Parse command-line arguments and set up logging.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", '-g', default='1', help="Number of GPUs for training.")
    parser.add_argument("--modal", '-m', default='1', choices=['0', '1'],
                        help="Training modality: 0 for single modality, 1 for multi-modality.")
    parser.add_argument("--data_dir", '-d', default="../autodl-fs/prepared-cassette", help="Directory containing data.")
    parser.add_argument("--output_dir", '-o', default='./result', help="Directory to save results.")
    parser.add_argument("--valid", '-v', default='20', help="Number of folds for k-fold validation.")
    parser.add_argument("--from_fold", default='0', help="Starting fold for training.")
    parser.add_argument("--train_fold", default='5', help="Number of folds to train this time.")

    args = parser.parse_args()

    res_path = args.output_dir
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.makedirs(res_path)

    logging.basicConfig(filemode='a', filename=f'{res_path}/log.log', level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] in %(funcName)s - %(levelname)s: %(message)s')

    return args


def print_params(params: dict):
    """
    Print model's hyperparameters in a formatted way.
    """
    print("=" * 20, "[Hyperparameters]", "=" * 20)
    for key, val in params.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for k, v in val.items():
                print(f"\t{k}: {v}")
        else:
            print(f"{key}: {val}")
    print("=" * 60)


def train(args: argparse.Namespace, hyper_param_dict: dict) -> dict:
    """
    Train the Salient Sleep Net model using PyTorch.
    """
    print("\n=== 进入训练函数 ===")
    # Fetch arguments
    res_path = args.output_dir
    k_folds = int(args.valid)
    from_fold = int(args.from_fold)
    train_fold = int(args.train_fold)
    if from_fold + train_fold > k_folds:
        train_fold = k_folds - from_fold
    modal = int(args.modal)
    
    print(f"交叉验证: 共{k_folds}折, 从第{from_fold}折开始, 训练{train_fold}折")

    # Fetch GPU numbers
    gpu_num = int(args.gpus) if 1 <= int(args.gpus) <= 4 else 1
    device = gpu_settings()
    print(f"使用 {gpu_num} 个GPU, 设备: {device}")
    logging.info(f"Using {gpu_num} GPUs")

    # Load data
    print(f"从 {args.data_dir} 加载数据...")
    npz_names = glob.glob(os.path.join(args.data_dir, '*.npz'))
    if len(npz_names) == 0:
        logging.critical(f"No npz files found in {args.data_dir}")
        print(f"错误: 未找到数据文件, 请检查数据目录 {args.data_dir}")
        exit(-1)
    npz_names.sort()
    print(f"找到 {len(npz_names)} 个npz文件")

    # Replace the problematic section
    # create the loading files list for each subject: [id1:[name1,name2],id2:[name3,name4],id3:[name5]]
    print("按被试者ID分组数据文件...")
    npzs_list = []
    ids = 20 if len(npz_names) < 100 else 83  # 20 for sleepedf-39, 83 for sleepedf-153
    for id in range(ids):
        inner_list = []
        for name in npz_names:
            pattern = re.compile(f".*SC4{id:02}[12][EFG]0.npz")
            if re.match(pattern, name):
                inner_list.append(name)
        if inner_list:  # not empty
            npzs_list.append(inner_list)
    
    print(f"分组后共有 {len(npzs_list)} 个被试者组")
    if len(npzs_list) == 0:
        print("警告: 未找到任何符合命名规则的数据文件!")
        print("将直接使用所有npz文件")
        # 如果没有找到符合命名规则的文件，直接使用所有文件
        npzs_list = [[name] for name in npz_names]

    # Use list splitting instead of numpy array splitting
    def split_list(lst, n):
        if len(lst) < n:
            print(f"警告: 数据组数量({len(lst)})小于交叉验证折数({n})!")
            # 如果数据组数量小于交叉验证折数，调整折数
            n = len(lst)
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    npzs_list = split_list(npzs_list, k_folds)
    print(f"数据已分为 {len(npzs_list)} 折用于交叉验证")

    # Save as object array to handle variable-length lists
    save_dict = {'split': np.array(npzs_list, dtype=object)}
    np.savez(os.path.join(res_path, 'split.npz'), **save_dict)
    print(f"分割信息已保存到 {os.path.join(res_path, 'split.npz')}")

    sleep_epoch_len = hyper_param_dict['sleep_epoch_len']
    print(f"睡眠片段长度: {sleep_epoch_len}秒")

    # 导入必要的模块
    try:
        from loss import WeightedCrossEntropyLoss
        print("成功导入损失函数")
    except ImportError as e:
        print(f"导入损失函数失败: {e}")
        raise
    
    # Loss function
    print("初始化损失函数...")
    class_weights = hyper_param_dict.get('class_weights', [1.0, 1.5, 2.0, 1.5, 1.0])
    weighted_loss = WeightedCrossEntropyLoss(weight=class_weights).to(device)
    print(f"损失函数权重: {class_weights}")

    # Result lists
    acc_list, val_acc_list = [], []
    loss_list, val_loss_list = [], []
    
    # 导入tqdm用于进度条显示
    try:
        from tqdm import tqdm
        print("成功导入tqdm进度条")
    except ImportError:
        print("警告: 未安装tqdm, 将使用简单进度显示")
        # 创建简单的tqdm替代品
        class SimpleTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable
                self.total = len(iterable)
                self.n = 0
                self.desc = kwargs.get('desc', '')
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.n >= self.total:
                    raise StopIteration
                    
                item = self.iterable[self.n]
                self.n += 1
                
                if self.n % 10 == 0 or self.n == self.total:
                    print(f"{self.desc}: {self.n}/{self.total} ({self.n/self.total*100:.1f}%)")
                    
                return item
                
            def set_postfix(self, **kwargs):
                # 简单显示后缀
                msg = ', '.join(f"{k}={v:.4f}" for k, v in kwargs.items())
                print(f"{self.desc}: {msg}")
        
        tqdm = SimpleTqdm

    # Model initialization
    print("初始化模型...")
    # 从超参数中获取模型参数
    model_params = hyper_param_dict.get('model', {})
    if not model_params:
        print("警告: 超参数中没有模型参数，使用默认值")
        model_params = {
            'encoder_dim': 32,
            'num_attention_heads': 8,
            'feed_forward_expansion_factor': 2,
            'conv_expansion_factor': 2,
            'feed_forward_dropout_p': 0.1,
            'attention_dropout_p': 0.1,
            'conv_dropout_p': 0.1,
            'conv_kernel_size': 3,
            'half_step_residual': True
        }
    
    # 减小模型隐藏层维度，避免内存不足问题
    reduced_hidden_dim = 16  # 将隐藏层维度从默认32减小到16
    print(f"由于序列较长，减小模型隐藏层维度到 {reduced_hidden_dim} 以避免内存不足")
    
    # 初始化模型
    try:
        print("创建模型实例...")
        model = ImprovedSleepModel(
            input_dim=1,  # 只使用EEG通道
            hidden_dim=reduced_hidden_dim,  # 使用减小的隐藏层维度
            num_classes=hyper_param_dict.get('num_classes', 5)
        ).to(device)
        print("模型创建成功")
    except Exception as e:
        print(f"模型创建失败: {e}")
        raise
    
    if gpu_num > 1:
        try:
            model = nn.DataParallel(model)
            print(f"模型已设置为DataParallel模式使用{gpu_num}个GPU")
        except Exception as e:
            print(f"设置DataParallel失败: {e}")
            print("使用单GPU继续")

    # Optimizer and scheduler
    print("初始化优化器和学习率调度器...")
    optimizer = Adam(model.parameters(), lr=hyper_param_dict.get('train', {}).get('learning_rate', 0.001))
    print(f"优化器: Adam, 学习率: {hyper_param_dict.get('train', {}).get('learning_rate', 0.001)}")
    
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=hyper_param_dict.get('patience', 5),
        verbose=True
    )
    print(f"学习率调度器: ReduceLROnPlateau, 耐心值: {hyper_param_dict.get('patience', 5)}")

    # Save initial weights
    torch.save(model.state_dict(), 'weights.pth')
    print("初始模型权重已保存")

    # K-fold training and validation
    for fold in range(from_fold, from_fold + train_fold):
        print(f"\n========== 开始第 {fold + 1}/{k_folds} 折训练 ==========")
        logging.info(f"Starting validation: {fold + 1}/{k_folds}")

    
        # 获取验证集和训练集
        print(f"划分训练集和验证集...")
        if fold >= len(npzs_list):
            print(f"错误: 指定的fold {fold} 超出了可用的折数 {len(npzs_list)}")
            break
            
        valid_npzs = list(itertools.chain.from_iterable(npzs_list[fold]))
        print(f"验证集: {len(valid_npzs)} 文件")
        
        train_npzs = list(set(npz_names) - set(valid_npzs))
        print(f"训练集: {len(train_npzs)} 文件")

        # 修改数据加载部分
        logging.info("Loading data...")
        print("加载训练数据...")
        train_eeg, train_labels = load_npz_files(train_npzs)
        print(f"加载验证数据...")
        val_eeg, val_labels = load_npz_files(valid_npzs)
        
        # 预处理数据
        print("预处理训练数据...")
        train_data, train_labels = improved_preprocess(train_eeg, train_labels, hyper_param_dict.get('preprocess', {}), True)
        print("预处理验证数据...")
        val_data, val_labels = improved_preprocess(val_eeg, val_labels, hyper_param_dict.get('preprocess', {}), True)
        logging.info("Preprocessing done")
        print("预处理完成")

        # 输出数据形状信息
        print(f"训练数据: {len(train_data)} 样本")
        print(f"验证数据: {len(val_data)} 样本")
        
        # 创建数据加载器
        print("创建数据集和数据加载器...")
        train_dataset = SleepDataset(train_data, train_labels)
        val_dataset = SleepDataset(val_data, val_labels)
        
        # 修改批次大小以避免内存不足
        small_batch_size = 2  # 使用小批次进行训练，而不是使用超参数中的batch_size
        print(f"使用批次大小 {small_batch_size} 以避免内存不足")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=small_batch_size,  # 使用小批次
            shuffle=True,
            num_workers=0,  # 避免多进程导致的内存增加
            pin_memory=True,  # 使用PIN_MEMORY加速GPU传输
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=small_batch_size,  # 使用小批次 
            num_workers=0,  # 避免多进程导致的内存增加
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"训练数据加载器: {len(train_loader)} 批次")
        print(f"验证数据加载器: {len(val_loader)} 批次")
        
        # 验证数据形状
        print("\n========= 数据形状验证 =========")
        print(f"训练样本数量: {len(train_data)}")
        
        # 检查数据和标签形状
        mismatch_count = 0
        for i in range(min(3, len(train_data))):  # 只检查前3个样本
            data_shape = train_data[i].shape
            label_shape = train_labels[i].shape if hasattr(train_labels[i], 'shape') else (len(train_labels[i]),)
            
            print(f"样本 {i}: 数据形状 {data_shape}, 标签形状 {label_shape}")
        print("================================\n")

        # 训练循环
        print(f"开始训练: {hyper_param_dict.get('train', {}).get('epochs', 50)} 轮...")
        epochs = hyper_param_dict.get('train', {}).get('epochs', 50)
        best_val_acc = 0
        best_model_path = os.path.join(res_path, f"fold_{fold + 1}_best_model.pth")
        
        for epoch in range(epochs):
            print(f"\n----- Epoch {epoch+1}/{epochs} -----")
            
            # 训练一个epoch
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, weighted_loss, device)
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            
            # 验证
            val_loss, val_acc = validate(model, val_loader, weighted_loss, device)
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"最佳模型已保存: {best_val_acc:.2f}% 准确率")
            
            # 记录损失和准确率
            loss_list.append(train_loss)
            acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
        
        print(f"第 {fold + 1} 折训练完成，最佳验证准确率: {best_val_acc:.2f}%")
        
        # 清理缓存，重新加载初始权重准备下一轮训练
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load('weights.pth'))

       

    # 清除临时权重文件
    os.remove('weights.pth')
    
    # 返回训练历史
    return {
        'acc': acc_list,
        'val_acc': val_acc_list,
        'loss': loss_list,
        'val_loss': val_loss_list
    }

class ImprovedSleepModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=5):
        super(ImprovedSleepModel, self).__init__()
        
        # 局部特征提取
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=50, stride=25)
        
        # 序列建模 - 使用可处理变长序列的组件
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=2
        )
        
        # 输出层
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, mask=None):
        # 处理列表输入
        if isinstance(x, list):
            outputs = []
            for i, single_x in enumerate(x):
                # 获取单个样本的掩码
                single_mask = mask[i] if mask is not None else None
                
                # 处理单个样本
                single_output = self._forward_single(single_x, single_mask)
                outputs.append(single_output)
            
            return outputs
        else:
            # 处理批量输入
            return self._forward_single(x, mask)
    
    def _forward_single(self, x, mask=None):
        """处理单个样本或批次的前向传播"""
        # x: [batch_size, seq_len, channels] 或 [seq_len, channels]
        
        # 确保输入是3D张量
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, seq_len, channels]
        
        batch_size, seq_len, channels = x.shape
        
        # 转换为卷积输入格式
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 局部特征提取
        conv_out = F.relu(self.conv(x))
        
        # 转换回序列格式
        conv_out = conv_out.permute(0, 2, 1)  # [batch_size, smaller_seq_len, hidden_dim]
        
        # 应用Transformer (支持掩码)
        if mask is not None:
            # 确保掩码是2D张量
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)  # [1, seq_len]
                
            # 调整掩码大小以匹配卷积后的长度
            conv_seq_len = conv_out.shape[1]
            mask = F.interpolate(mask.float().unsqueeze(1), size=conv_seq_len).squeeze(1).bool()
            
            # 创建填充掩码 (True表示填充位置)
            padding_mask = ~mask
            
            # 序列建模
            transformer_out = self.transformer(conv_out.transpose(0, 1), src_key_padding_mask=padding_mask).transpose(0, 1)
        else:
            transformer_out = self.transformer(conv_out.transpose(0, 1)).transpose(0, 1)
        
        # 分类
        output = self.classifier(transformer_out)
        
        return output

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels, masks) in enumerate(train_loader):
        # 移至设备
        if isinstance(inputs, list):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
            
        if isinstance(labels, list):
            labels = [l.to(device) for l in labels]
        else:
            labels = labels.to(device)
            
        if isinstance(masks, list):
            masks = [m.to(device) for m in masks]
        else:
            masks = masks.to(device)
        
        # 前向传播
        outputs = model(inputs, masks)
        
        # 计算损失
        if isinstance(outputs, list) and isinstance(labels, list):
            # 处理列表格式的输出和标签
            batch_loss = 0
            for output, label in zip(outputs, labels):
                batch_loss += criterion(output.view(-1, output.size(-1)), label.view(-1))
            loss = batch_loss / len(outputs)
        else:
            # 处理张量格式的输出和标签
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        if isinstance(outputs, list) and isinstance(labels, list):
            batch_correct = 0
            batch_total = 0
            for output, label in zip(outputs, labels):
                _, predicted = torch.max(output.data, -1)
                
                # 只计算有效数据点
                valid_mask = (label != -100)
                if valid_mask.sum() > 0:
                    batch_correct += (predicted[valid_mask] == label[valid_mask]).sum().item()
                    batch_total += valid_mask.sum().item()
            
            correct += batch_correct
            total += batch_total
        else:
            _, predicted = torch.max(outputs.data, -1)
            
            # 只计算有效数据点
            valid_mask = (labels != -100)
            correct += (predicted[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
        total_loss += loss.item()
        
    return total_loss/len(train_loader), 100.*correct/total if total > 0 else 0

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, masks in val_loader:
            # 移动数据到设备
            if isinstance(inputs, list):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
                
            if isinstance(labels, list):
                labels = [l.to(device) for l in labels]
            else:
                labels = labels.to(device)
                
            if isinstance(masks, list):
                masks = [m.to(device) for m in masks]
            else:
                masks = masks.to(device)
            
            # 前向传播
            outputs = model(inputs, masks)
            
            # 损失计算
            if isinstance(outputs, list) and isinstance(labels, list):
                # 处理列表格式的输出和标签
                batch_loss = 0
                for output, label in zip(outputs, labels):
                    batch_loss += criterion(output.view(-1, output.size(-1)), label.view(-1))
                loss = batch_loss / len(outputs)
            else:
                # 处理张量格式的输出和标签
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # 统计
            total_loss += loss.item()
            
            # 准确率计算 - 与训练相同的处理方式
            if isinstance(outputs, list):
                batch_correct = 0
                batch_total = 0
                for output, label in zip(outputs, labels):
                    _, predicted = torch.max(output, -1)  # 在类别维度上取最大值
                    
                    # 忽略填充标签
                    valid_mask = (label != -100)
                    if valid_mask.sum() > 0:
                        predicted_valid = predicted[valid_mask]
                        labels_valid = label[valid_mask]
                        batch_total += labels_valid.numel()
                        batch_correct += (predicted_valid == labels_valid).sum().item()
                
                correct += batch_correct
                total += batch_total
            else:
                _, predicted = torch.max(outputs, 2)
                
                # 忽略填充标签
                valid_mask = (labels != -100)
                if valid_mask.sum() > 0:
                    predicted_valid = predicted[valid_mask]
                    labels_valid = labels[valid_mask]
                    total += labels_valid.numel()
                    correct += (predicted_valid == labels_valid).sum().item()
    
    return total_loss/len(val_loader), 100.*correct/total if total > 0 else 0

def improved_preprocess(data, labels, param, is_train):
    """改进的预处理函数，保留完整睡眠序列"""
    # 标准化
    normalized_data = [normalization(d) for d in data]
    processed_data = []
    processed_labels = []
    
    for i, (d, l) in enumerate(zip(normalized_data, labels)):
        print(f"处理文件 {i}: 数据形状 {d.shape}, 标签形状 {l.shape}")
        
        # 直接使用完整序列，不切分
        processed_data.append(d)
        processed_labels.append(l)
        
        # 数据增强 - 仅在训练集使用
        if is_train and param.get('use_augmentation', False):
            # 添加噪声增强
            noise_level = param.get('noise_level', 0.05)
            noise = torch.randn_like(d) * noise_level
            noisy_data = d + noise
            processed_data.append(noisy_data)
            processed_labels.append(l)
            
            # 时间扭曲增强
            # ...其他增强方法
    
    return processed_data, processed_labels

def improved_collate_fn(batch):
    """处理变长序列的整合函数"""
    # 分离数据和标签
    data, labels = zip(*batch)
    
    # 找出当前批次中的最大序列长度
    max_len = max(d.shape[0] for d in data)
    
    # 创建填充和掩码
    padded_data = []
    padded_labels = []
    masks = []
    
    for d, l in zip(data, labels):
        # 当前序列长度
        curr_len = d.shape[0]
        
        # 创建掩码 (1表示有效, 0表示填充)
        mask = torch.ones(curr_len, dtype=torch.bool)
        if curr_len < max_len:
            # 添加0填充
            padding = torch.zeros(max_len - curr_len, d.shape[1], d.shape[2])
            padded_d = torch.cat([d, padding], dim=0)
            
            # 标签填充 (-100表示忽略索引)
            padding_label = torch.full((max_len - curr_len,), -100, dtype=l.dtype)
            padded_l = torch.cat([l, padding_label], dim=0)
            
            # 扩展掩码
            mask_padding = torch.zeros(max_len - curr_len, dtype=torch.bool)
            mask = torch.cat([mask, mask_padding], dim=0)
        else:
            padded_d = d
            padded_l = l
        
        padded_data.append(padded_d)
        padded_labels.append(padded_l)
        masks.append(mask)
    
    # 堆叠为批次
    padded_data = torch.stack(padded_data)
    padded_labels = torch.stack(padded_labels)
    masks = torch.stack(masks)
    
    return padded_data, padded_labels, masks

if __name__ == "__main__":
    print("=== 程序开始执行 ===")
    try:
        # Set up GPU settings
        device = gpu_settings()
        print(f"使用设备: {device}")

        # Parse arguments
        args = get_parser()
        print(f"命令行参数: {args}")
        print(f"数据目录: {args.data_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"GPU数量: {args.gpus}")
        print(f"模态: {args.modal}")
        print(f"交叉验证折数: {args.valid}")
        print(f"起始折: {args.from_fold}")
        print(f"训练折数: {args.train_fold}")

        # Load hyperparameters from YAML file
        print("加载超参数...")
        try:
            with open("hyperparameters.yaml", encoding='utf-8') as f:
                hyper_params = yaml.safe_load(f)
            print("超参数加载成功")
        except Exception as e:
            print(f"加载超参数失败: {e}")
            print("尝试创建默认超参数")
            # 创建默认超参数
            hyper_params = {
                'sleep_epoch_len': 30,
                'num_classes': 5,
                'class_weights': [1.0, 1.5, 2.0, 1.5, 1.0],
                'preprocess': {
                    'big_group_size': 200,
                    'sequence_epochs': 32,
                    'enhance_window_stride': 5
                },
                'train': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 50
                },
                'model': {
                    'encoder_dim': 32,
                    'num_attention_heads': 8,
                    'feed_forward_expansion_factor': 2,
                    'conv_expansion_factor': 2,
                    'feed_forward_dropout_p': 0.1,
                    'attention_dropout_p': 0.1,
                    'conv_dropout_p': 0.1,
                    'conv_kernel_size': 3,
                    'half_step_residual': True
                },
                'patience': 5
            }
        
        print_params(hyper_params)

        # 确保关键参数存在
        model_params = hyper_params.get('model', {})
        for param in ['num_attention_heads', 'feed_forward_expansion_factor', 
                     'conv_expansion_factor', 'feed_forward_dropout_p', 
                     'attention_dropout_p', 'conv_dropout_p', 
                     'conv_kernel_size', 'half_step_residual']:
            if param not in model_params:
                print(f"警告: 模型参数 '{param}' 缺失，使用默认值")
                if param == 'num_attention_heads':
                    model_params[param] = 8
                elif param == 'feed_forward_expansion_factor':
                    model_params[param] = 2
                elif param == 'conv_expansion_factor':
                    model_params[param] = 2
                elif param == 'feed_forward_dropout_p':
                    model_params[param] = 0.1
                elif param == 'attention_dropout_p':
                    model_params[param] = 0.1
                elif param == 'conv_dropout_p':
                    model_params[param] = 0.1
                elif param == 'conv_kernel_size':
                    model_params[param] = 3
                elif param == 'half_step_residual':
                    model_params[param] = True

        # 检查数据目录
        print(f"检查数据目录: {args.data_dir}")
        if not os.path.exists(args.data_dir):
            print(f"错误: 数据目录 {args.data_dir} 不存在!")
            exit(1)
            
        npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
        print(f"找到 {len(npz_files)} 个npz文件")
        if len(npz_files) == 0:
            print(f"警告: 在 {args.data_dir} 中未找到npz文件!")

        # Train the model and get training history
        print("开始训练模型...")
        train_history = train(args, hyper_params)
        print("训练完成!")

        print("=== 程序执行完毕 ===")
    except Exception as e:
        import traceback
        print(f"程序执行异常: {e}")
        traceback.print_exc()