import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失函数，用于处理不平衡的睡眠阶段分类问题
    """
    def __init__(self, weight=None, reduction='mean'):
        """
        初始化加权交叉熵损失函数
        
        Args:
            weight: 各类别的权重
            reduction: 损失聚合方式，可选'none', 'mean', 'sum'
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight) if weight is not None else None
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算损失值
        
        Args:
            inputs: 模型输出，形状为 [batch_size, num_classes, ...] 或列表
            targets: 目标标签，形状为 [batch_size, ...] 或列表
            
        Returns:
            loss: 损失值
        """
        # 检查inputs是2D还是3D
        if len(inputs.shape) == 2:
            # 已经是2D: [batch_size*seq_len, num_classes]
            return F.cross_entropy(
                inputs,
                targets,
                weight=self.weight,
                reduction=self.reduction,
                ignore_index=-100  # 忽略填充标签
            )
        elif len(inputs.shape) == 3:
            # 3D: [batch_size, seq_len, num_classes]
            batch_size, seq_len, num_classes = inputs.shape
            
            # 处理列表输入
            if isinstance(inputs, list) and isinstance(targets, list):
                total_loss = 0
                batch_size = len(inputs)
                
                # 打印调试信息
                print(f"损失函数处理列表输入，batch_size={batch_size}")
                
                for i in range(batch_size):
                    input_tensor = inputs[i]
                    target_tensor = targets[i]
                    
                    # 打印形状信息进行调试
                    print(f"输入[{i}]形状: {input_tensor.shape}, 标签[{i}]形状: {target_tensor.shape if hasattr(target_tensor, 'shape') else 'unknown'}")
                    
                    # 检查标签是否为空
                    if target_tensor.numel() == 0:
                        print(f"警告: 样本 {i} 的标签为空！跳过该样本")
                        continue
                    
                    # 确保权重与设备一致
                    if self.weight is not None:
                        self.weight = self.weight.to(input_tensor.device)
                    
                    # 确保标签和输入维度匹配
                    if input_tensor.dim() == 3 and target_tensor.dim() == 2:
                        # 输入: [seq_len, output_len, num_classes]
                        # 标签: [seq_len, output_len]
                        pass  # 正常情况
                    elif input_tensor.dim() == 3 and target_tensor.dim() == 1:
                        # 尝试调整标签维度
                        seq_len = input_tensor.size(0)
                        if target_tensor.size(0) == seq_len:
                            target_tensor = target_tensor.unsqueeze(1)  # [seq_len] -> [seq_len, 1]
                        else:
                            print(f"错误: 输入序列长度 {seq_len} 与标签长度 {target_tensor.size(0)} 不匹配")
                    
                    try:
                        # 检查输入是三维张量，且形状为[seq_len, seq_len, num_classes]
                        if input_tensor.dim() == 3:
                            # 获取形状信息
                            dim0, dim1, dim2 = input_tensor.shape
                            label_dim = target_tensor.shape[0]
                            
                            # 形状特殊处理
                            if dim0 == dim1 and dim0 == label_dim:
                                # 输入: [seq_len, seq_len, num_classes], 标签: [seq_len]
                                # 选择对角线元素
                                diag_indices = torch.arange(dim0, device=input_tensor.device)
                                input_tensor = input_tensor[diag_indices, diag_indices]
                                print(f"选择对角线元素后的形状: {input_tensor.shape}")
                            elif dim1 == label_dim:
                                # 输入: [other_dim, seq_len, num_classes], 标签: [seq_len]
                                # 取第一行，确保维度匹配
                                input_tensor = input_tensor[0, :, :]
                                print(f"提取第一行后的形状: {input_tensor.shape}")
                            else:
                                # 默认处理：只保留与标签对应的时间维度
                                print(f"默认处理：将 {input_tensor.shape} 转换为与标签 {target_tensor.shape} 匹配的形状")
                                
                                # 获取转置后的维度
                                transposed_shape = input_tensor.transpose(0, 1).shape
                                print(f"转置后的形状: {transposed_shape}")
                                
                                # 选择与标签长度相匹配的维度处理
                                if transposed_shape[0] == label_dim:
                                    # 如果转置后第一维与标签匹配，使用转置后的数据
                                    input_tensor = input_tensor.transpose(0, 1)
                                    # 取第一个样本
                                    input_tensor = input_tensor[:, 0, :]
                                    print(f"转置并提取后的形状: {input_tensor.shape}")
                                else:
                                    # 截断到标签长度
                                    input_tensor = input_tensor[:label_dim, 0, :]
                                    print(f"截断到标签长度后的形状: {input_tensor.shape}")
                        
                        # 检查2D输入和1D标签的情况
                        if input_tensor.dim() == 2 and target_tensor.dim() == 1:
                            input_dim0, input_dim1 = input_tensor.shape
                            target_dim0 = target_tensor.shape[0]
                            
                            # 如果输入和标签批次大小不匹配
                            if input_dim0 != target_dim0:
                                print(f"批次大小不匹配: 输入{input_tensor.shape}, 标签{target_tensor.shape}")
                                
                                # 方案1: 如果标签只有1个元素，复制到和输入一样的长度
                                if target_dim0 == 1:
                                    target_tensor = target_tensor.repeat(input_dim0)
                                    print(f"标签扩展至: {target_tensor.shape}")
                                # 方案2: 取输入的前target_dim0个样本
                                elif input_dim0 > target_dim0:
                                    input_tensor = input_tensor[:target_dim0]
                                    print(f"输入截断至: {input_tensor.shape}")
                                # 方案3: 取标签的前input_dim0个样本
                                else:  # input_dim0 < target_dim0
                                    target_tensor = target_tensor[:input_dim0]
                                    print(f"标签截断至: {target_tensor.shape}")
                        
                        # 确保标签类型是长整型(Long)
                        target_tensor = target_tensor.long()
                        
                        # 计算损失，忽略值为-100的填充标签
                        loss = F.cross_entropy(
                            input_tensor.reshape(-1, input_tensor.size(-1)),
                            target_tensor.reshape(-1),
                            weight=self.weight,
                            reduction=self.reduction,
                            ignore_index=-100  # 忽略填充用的-100标签
                        )
                        
                        total_loss += loss
                    except Exception as e:
                        print(f"计算损失时出错: {e}")
                        print(f"输入形状: {input_tensor.shape}, 展平后: {input_tensor.view(-1, input_tensor.size(-1)).shape}")
                        print(f"标签形状: {target_tensor.shape}, 展平后: {target_tensor.view(-1).shape}")
                        raise e
                
                # 检查是否所有样本都被跳过
                if batch_size == 0:
                    print("错误: 所有样本的标签都为空！")
                    return torch.tensor(0.0, requires_grad=True, device=inputs[0].device if inputs else "cpu")
                
                # 返回平均损失
                return total_loss / batch_size
            else:
                # 原始处理逻辑
                batch_size, seq_len, num_classes = inputs.shape
                
                # 确保权重与设备一致
                if self.weight is not None:
                    self.weight = self.weight.to(inputs.device)
                
                # 计算交叉熵损失
                inputs_flat = inputs.view(-1, num_classes)
                targets_flat = targets.view(-1)
                
                return F.cross_entropy(
                    inputs_flat,
                    targets_flat,
                    weight=self.weight,
                    reduction=self.reduction
                )
        else:
            raise ValueError("输入张量的维度必须是2或3")