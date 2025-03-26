# 创建调试脚本
from preprocess import normalization, preprocess
import torch
import numpy as np

if __name__ == "__main__":
    # 1. 准备测试数据
    test_data = [
        torch.randn(1, 1000, 1, 1)  # [C, T, W, 1] 格式
    ]
    test_labels = [
        np.random.randint(0, 5, 1000)  # 随机生成标签
    ]
    
    # 2. 定义参数（与hyperparameters.yaml一致）
    param = {
        'big_group_size': 300,
        'sequence_epochs': 30,
        'enhance_window_stride': 15
    }
    
    # 3. 执行到目标步骤
    with ThreadPoolExecutor(max_workers=1) as executor:  # 单线程调试
        # 执行前置步骤
        data = [torch.tensor(d) for d in test_data]
        normalized_data = list(executor.map(normalization, data))
        
        # 分组处理
        grouped_labels = list(executor.map(preprocess.label_big_group, test_labels))
        print(f"Grouped labels shape: {[g.shape for g in grouped_labels]}")
        
        # 进入调试模式
        import pdb; pdb.set_trace()
        enhanced_labels = list(executor.map(preprocess.labels_window_slice, grouped_labels))