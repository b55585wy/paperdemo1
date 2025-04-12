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
    normalized_data = [normalization(d) for d in data]
    
    # 创建样本（每个样本是sequence_epochs个连续的30秒片段）
    samples = []
    sample_labels = []
    
    for i, (d, l) in enumerate(zip(normalized_data, labels)):
        l = l.squeeze(-1)  # 确保标签是[N]维度
        time_steps = d.shape[0]
        sequence_epochs = param['sequence_epochs']
        stride = param['enhance_window_stride'] if not not_enhance else sequence_epochs
        
        cnt = 0
        while (cnt + sequence_epochs) <= time_steps:
            sample_data = d[cnt:cnt+sequence_epochs]
            sample_label = l[cnt + sequence_epochs - 1].item()  # 关键修改
            
            samples.append(sample_data)
            sample_labels.append(sample_label)
            cnt += stride
    
    # 验证标签格式
    assert all(isinstance(lbl, int) for lbl in sample_labels), "标签必须为整数"
    print(f"示例标签类型: {type(sample_labels[0])}, 值: {sample_labels[0]}")
    unique_labels = set(sample_labels)
    print(f"唯一标签值: {unique_labels}")
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