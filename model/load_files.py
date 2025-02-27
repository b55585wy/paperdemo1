import numpy as np
import torch
from typing import Tuple, List

def load_npz_file(npz_file: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    加载单个npz文件的数据
    
    Args:
        npz_file: npz文件路径
        
    Returns:
        data: PSG数据
        labels: 睡眠分期标签
        sampling_rate: 采样率
    """
    with np.load(npz_file) as f:
        data = torch.from_numpy(f["x"]).float()
        labels = torch.from_numpy(f["y"]).long()
        sampling_rate = f["fs"].item()
    return data, labels, sampling_rate

def load_npz_files(npz_files: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    加载多个npz文件的数据
    
    Args:
        npz_files: npz文件路径列表
        
    Returns:
        data_list: PSG数据列表
        labels_list: 标签列表
    """
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        print(f"Loading {npz_f} ...")
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # 调整数据维度 [None, W, C] -> [C, None, W, 1]
        tmp_data = tmp_data.permute(2, 0, 1).unsqueeze(-1)
        
        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"Loaded {len(data_list)} files.")
    return data_list, labels_list