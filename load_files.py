import glob
import itertools
import re

import numpy as np
from typing import List, Tuple, Dict
import torch
import os
import torch.nn as nn
from models import ConbimambaBlock


def load_npz_file(npz_file: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load data from npz files.

    :param npz_file: a str of npz filename

    :return: a tuple of PSG data, labels and sampling rate of the npz file
    """
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_files(npz_files: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    加载数据并分离EEG和EOG通道

    :param npz_files: npz文件名列表
    :return: (eeg_data_list, eog_data_list, labels_list)
    """
    eeg_data_list = []
    eog_data_list = []
    labels_list = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")
 # d的维度是[睡眠不定长（根据每个人每晚睡眠的时长来确定）片段数N, 片段固定时长, 通道]
        # l的维度是[睡眠不定长片段数N, 1]
        # 数据增强实际上是相当于重复采样扩大睡眠不定长片段N
        # 转换为torch张量
        tmp_data = torch.tensor(tmp_data, dtype=torch.float32)  # Tensor维度是 (None,W(能看做宽度吗？这个时间步长),C)
        tmp_labels = torch.tensor(tmp_labels, dtype=torch.int32)  # Tensor (None,1)

     
        tmp_data = torch.squeeze(tmp_data)  # [None, W, C]
        #检查维度是否为3，否则就报错

        # 分离EEG和EOG通道
        eeg_data = tmp_data[:, :, 0:1]  # 取第一个通道作为EEG   # [None, W, 1]
        # eog_data = tmp_data[:, :, 1:2]  # 取第二个通道作为EOG   # [None, W, 1]


        eeg_data_list.append(eeg_data)
        # eog_data_list.append(eog_data)
        labels_list.append(tmp_labels)

    print(f"Loaded {len(eeg_data_list)} files totally.")
    # 先只用脑电信号
    # return eeg_data_list, eog_data_list, labels_list
    return eeg_data_list, labels_list





def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':
    # path = "./testdata"
    # k_folds = 5
    # from_fold = 0
    # train_fold = 5
    # npzs_list = []
    # npz_names = glob.glob(os.path.join(path, '*.npz'))
    # ids = 20 if len(npz_names) < 100 else 83  # 20 for sleepedf-39, 83 for sleepedf-153
    # for id in range(ids):
    #     inner_list = []
    #     for name in npz_names:
    #         pattern = re.compile(f".*SC4{id:02}[12][EFG]0.npz")
    #         if re.match(pattern, name):
    #             inner_list.append(name)
    #     if inner_list:  # not empty
    #         npzs_list.append(inner_list)
    #
    # npzs_list = split_list(npzs_list, k_folds)
    # for fold in range(from_fold, from_fold + train_fold):
    #
    #     valid_npzs = list(itertools.chain.from_iterable(npzs_list[fold]))
    #     train_npzs = list(set(npz_names) - set(valid_npzs))
    #
    #     # 修改数据加载部分
    #     train_eeg, train_eog, train_labels = load_npz_files(train_npzs)
    #     val_eeg, val_eog, val_labels = load_npz_files(valid_npzs)
    path = ["./testdata/SC4022E0.npz"]

    # 修改数据加载部分
    # train_eeg, train_eog, train_labels = load_npz_files(path)
    # val_eeg, val_eog, val_labels = load_npz_files(path)
    train_eeg, train_labels = load_npz_files(path)
    val_eeg, val_labels = load_npz_files(path)



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
        # 确保标签维度正确
        l = l.squeeze(-1) if l.dim() > 1 else l  # 确保标签是[N]而不是[N,1]
        
        # 正确的维度访问方式
        time_steps = d.shape[0]  # 片段数N
        
        # 对每个病人/每晚的数据单独进行滑动窗口处理
        sequence_epochs = param['sequence_epochs']  # 获取序列长度参数
        stride = param['enhance_window_stride'] if not not_enhance else sequence_epochs
        
        # 滑动窗口创建样本
        cnt = 0
        samples_per_file = 0
        
        while (cnt + sequence_epochs) <= time_steps:
            # 提取样本和对应标签
            sample_data = d[cnt:cnt+sequence_epochs]  # [sequence_epochs, W, C]
            sample_label = l[cnt+sequence_epochs-1]  # 取最后一个时间步的标签作为序列标签
            
            # 添加到样本列表
            samples.append(sample_data)
            sample_labels.append(sample_label)
            samples_per_file += 1
            cnt += stride
            
        print(f"文件 {i}: 生成了 {samples_per_file} 个样本")
    
    # 将样本转换为张量批次
    # 使用自定义collate函数处理可变长度序列
    return samples, sample_labels
