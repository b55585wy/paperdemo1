import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from load_files import load_npz_files


def normalization(data: torch.Tensor) -> torch.Tensor:
    """
    对输入数据进行标准化处理
    
    Args:
        data: 输入数据，维度为 [None, W, C]，其中None是样本数量，W是时间宽度，C是通道数
        
    
    """
    # 在时间维度上计算均值和标准差
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    return (data - mean) / (std + 1e-8)

def preprocess(data: List[torch.Tensor], 
               labels: List[torch.Tensor], 
               param: Dict, 
               not_enhance: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    数据预处理
    
    Args:
        data: PSG数据列表
        labels: 标签列表
        param: 预处理参数
        not_enhance: 是否不使用数据增强
        
    Returns:
        处理后的数据和标签
    """
    # 标准化
    normalized_data = [normalization(torch.tensor(d)) for d in data]
    
    # 创建样本（每个样本是sequence_epochs个连续的30秒片段）
    samples = []
    sample_labels = []
    
     
    
    for i, (d, l) in enumerate(zip(normalized_data, labels)):
        # 打印原始数据和标签的形状
        print(f"文件 {i}: 数据形状 {d.shape}, 标签形状 {l.shape}")
        
        # 正确的维度访问方式
        time_steps = d.shape[0]  # 片段数N
        
        # 对每个病人/每晚的数据单独进行滑动窗口处理
        sequence_epochs = param['sequence_epochs']  # 获取序列长度参数
        stride = param['enhance_window_stride'] if not not_enhance else sequence_epochs
        
        # 滑动窗口创建样本
        cnt = 0
        samples_per_file = 0
        # d的维度是[睡眠不定长（根据每个人每晚睡眠的时长来确定）片段数N, 片段固定时长, 通道]
        # l的维度是[睡眠不定长片段数N, 1]
        # 数据增强实际上是相当于重复采样扩大睡眠不定长片段N
        while (cnt + sequence_epochs) <= time_steps:
            # 提取样本和对应标签
            sample_data = d[cnt:cnt+sequence_epochs, :, :]  # 在第一维(N)上截取
            sample_label = l[cnt:cnt+sequence_epochs]
            
       
        
             
            
            samples.append(sample_data)
            sample_labels.append(sample_label)
            samples_per_file += 1
            cnt += stride
        
    
    
    
    
    
    # 返回样本列表和标签列表
    return samples, sample_labels

if __name__ == "__main__":
    import glob
    import os
    import yaml
    
    # 测试代码
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)
        
    data_dir = "/Users/wangyi/Desktop/data/sleepedf/prepared/"
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    data, labels = load_npz_files(npz_files)
    processed_data, processed_labels = preprocess(data, labels, hyper_params['preprocess'])
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed labels shape: {processed_labels.shape}")