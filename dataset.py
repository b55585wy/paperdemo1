import torch
from torch.utils.data import Dataset

class SleepDataset(Dataset):
    """
    睡眠数据集类，用于处理由preprocess函数返回的样本列表
    """
    def __init__(self, data_samples, label_samples):
        """
        初始化睡眠数据集
        
        Args:
            data_samples: 样本列表，每个样本形状为[channels, sequence_length, features]
            label_samples: 标签列表，每个标签形状为[sequence_length]
        """
        self.data_samples = data_samples
        self.label_samples = label_samples
        
        # 确保数据和标签数量一致
        assert len(data_samples) == len(label_samples), "样本数量与标签数量不匹配"
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本和标签"""
        # 确保转换为张量
        data = self.data_samples[idx] if isinstance(self.data_samples[idx], torch.Tensor) else torch.tensor(self.data_samples[idx])
        label = self.label_samples[idx] if isinstance(self.label_samples[idx], torch.Tensor) else torch.tensor(self.label_samples[idx])
        
        return data, label