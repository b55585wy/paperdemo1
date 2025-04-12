import torch


def custom_collate_fn(batch):
    """
    自定义的collate函数，用于处理不规则长度的序列数据
    
    Args:
        batch: 包含(data, label)元组的列表
        
    Returns:
        batch_data, batch_labels, masks: 批次数据、标签和掩码
    """
    # 分离数据和标签
    data = []
    labels = []
    masks = []
    
    print("批次大小:", len(batch))
    
    # 处理数据和标签
    for i, (sample_data, sample_label) in enumerate(batch):
        if isinstance(sample_data, torch.Tensor):
            # 记录详细的原始形状
            data_shape = sample_data.shape
            label_shape = sample_label.shape if hasattr(sample_label, 'shape') else (len(sample_label),)
            print(f"样本 {i} 原始形状: 数据={data_shape}, 标签={label_shape}")
            
            # 检查特殊情况：[片段数, 时间步长, 特征数]
            if len(data_shape) == 3 and data_shape[0] > 100:
                # 特殊情况：多个片段数据
                num_segments = data_shape[0]
                print(f"  处理多片段数据: 片段数={num_segments}")
                
                # 每个片段应该有自己的标签，检查标签是否匹配
                if not isinstance(sample_label, torch.Tensor) or sample_label.shape[0] != num_segments:
                    print(f"  警告：标签数量 ({label_shape}) 与片段数量 ({num_segments}) 不匹配!")
                    # 如果标签数量不匹配，创建伪标签（使用0）
                    segment_labels = torch.zeros(num_segments, dtype=torch.long)
                    print(f"  创建伪标签: 形状={segment_labels.shape}")
                else:
                    # 标签数量匹配片段数量
                    segment_labels = sample_label
                    print(f"  标签匹配片段数量: 片段数={num_segments}, 标签数={segment_labels.shape[0]}")
                
                # 限制处理的片段数量以避免内存问题
                max_segments = min(64, num_segments)
                
                # 每个片段形成一个独立样本
                for seg_idx in range(max_segments):
                    # 提取单个片段的数据 [时间步长, 特征数]
                    segment_data = sample_data[seg_idx].unsqueeze(0)  # [1, 时间步长, 特征数]
                    # 使用对应的标签
                    segment_label = segment_labels[seg_idx].unsqueeze(0)  # 单个标签 [1]
                    
                    print(f"  片段 {seg_idx} 数据形状: {segment_data.shape}, 标签形状: {segment_label.shape if hasattr(segment_label, 'shape') else '标量'}")
                    
                    # 添加到批次
                    data.append(segment_data)
                    labels.append(segment_label)
                    # 创建掩码 - 全1表示所有数据点都有效
                    mask = torch.ones(segment_data.shape[1], dtype=torch.bool)
                    masks.append(mask)
            else:
                # 标准情况，直接添加
                data.append(sample_data)
                labels.append(sample_label)
                # 创建掩码 - 全1表示所有数据点都有效
                mask = torch.ones(sample_data.shape[0], dtype=torch.bool)
                masks.append(mask)
        else:
            print(f"样本 {i} 的数据不是Tensor, 类型: {type(sample_data)}")
    
    # 检查是否所有样本都无效
    if len(data) == 0:
        print("错误: 处理后的批次为空，所有样本都被跳过")
        # 创建一个伪批次，避免程序崩溃
        dummy_data = torch.zeros((1, 1, 1), dtype=torch.float32)
        dummy_label = torch.zeros((1), dtype=torch.long)
        dummy_mask = torch.ones((1), dtype=torch.bool)
        return [dummy_data], [dummy_label], [dummy_mask]
    
    print(f"最终批次: {len(data)} 个数据, {len(labels)} 个标签, {len(masks)} 个掩码")
    
    # 确保数据和标签数量一致
    assert len(data) == len(labels) == len(masks), "数据、标签和掩码数量不一致"
    
    # 不使用stack，直接返回列表
    return data, labels, masks 