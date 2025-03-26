import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def normalization(data: torch.Tensor) -> torch.Tensor:
    """
    对数据进行标准化
    
    Args:
        data: 输入数据 [C, None, W, 1]
        
    Returns:
        标准化后的数据
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
    def data_big_group(d: torch.Tensor) -> torch.Tensor:
        """将数据分成大组以防止数据泄露"""
        chunks = []
        beg = 0
        while (beg + param['big_group_size']) <= d.shape[1]:
            y = d[:, beg:beg + param['big_group_size'], :, :]
            chunks.append(y.unsqueeze(1))
            beg += param['big_group_size']
        return torch.cat(chunks, dim=1) if chunks else torch.tensor([])

    def label_big_group(l: torch.Tensor, param: dict) -> torch.Tensor:
        """将标签分成大组以防止数据泄露"""
        chunks = []
        beg = 0
        while (beg + param['big_group_size']) <= len(l):
            y = l[beg:beg + param['big_group_size']]
            y = y.unsqueeze(0)  # 等同于 np.newaxis
            chunks.append(y)
            beg += param['big_group_size']
        
        # 如果chunks不为空，沿着dim=0拼接所有张量；否则返回形状为(0, 1)的空张量
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 1), dtype=torch.long)

    def data_window_slice(d: torch.Tensor) -> torch.Tensor:
        """数据增强的滑动窗口处理"""
        stride = param['sequence_epochs'] if not_enhance else param['enhance_window_stride']
        chunks = []
        
        for modal in d:  # 遍历每个模态
            modal_chunks = []
            for group in modal:  # 遍历每个大组
                group_chunks = []
                cnt = 0
                while (cnt + param['sequence_epochs']) <= len(group):
                    window = group[cnt:cnt + param['sequence_epochs']]
                    group_chunks.append(window.unsqueeze(0))
                    cnt += stride
                if group_chunks:
                    modal_chunks.append(torch.cat(group_chunks, dim=0))
            if modal_chunks:
                chunks.append(torch.cat(modal_chunks, dim=0).unsqueeze(0))
        
        return torch.cat(chunks, dim=0) if chunks else torch.tensor([])

    def labels_window_slice(l:torch.Tensor, param:dict, not_enhance:bool=False)->torch.Tensor:
        """将标签数据按照滑动窗口进行增强处理"""
        stride = param['sequence_epochs'] if not_enhance else param['enhance_window_stride']
        return_labels = None
        
        for cnt1, group in enumerate(l):
            # 添加维度检查
            if group.dim() == 1:
                group = group.unsqueeze(1)  # 转换为二维张量
                
            flat_labels = None
            cnt2 = 0
            
            while (cnt2 + param['sequence_epochs']) <= group.shape[0]:
                y = group[cnt2:cnt2 + param['sequence_epochs']]
                y = y.unsqueeze(0)
                
                flat_labels = y if flat_labels is None else torch.cat([flat_labels, y], dim=0)
                cnt2 += stride
            
            if flat_labels is not None:
                return_labels = flat_labels if return_labels is None else torch.cat([return_labels, flat_labels], dim=0)
        
        return return_labels if return_labels is not None else torch.zeros((0, param['sequence_epochs'], 1), dtype=torch.long)

    # 使用线程池处理数据
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 标准化
        data = [torch.tensor(d) for d in data]
        normalized_data = list(executor.map(normalization, data))
        
        # 分组
        grouped_data = list(executor.map(data_big_group, normalized_data))
        grouped_labels = list(executor.map(label_big_group, labels))
        
        # 滑动窗口处理
        enhanced_data = list(executor.map(data_window_slice, grouped_data))
        enhanced_labels = list(executor.map(labels_window_slice, grouped_labels))

    # 合并所有处理后的数据
    final_data = torch.cat([d for d in enhanced_data if d.numel()], dim=1)
    final_labels = torch.cat([l for l in enhanced_labels if l.numel()], dim=0)

    # 确保标签是long类型
    return final_data, final_labels.long().unsqueeze(2).unsqueeze(3)

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