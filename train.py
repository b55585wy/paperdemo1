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
import yaml
import sys

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from preprocess import preprocess
from load_files import load_npz_files
from evaluation import draw_training_plot
from models import SingleSalientModel, TwoStreamSalientModel
from loss_function import WeightedCrossEntropyLoss


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
    # Fetch arguments
    res_path = args.output_dir
    k_folds = int(args.valid)
    from_fold = int(args.from_fold)
    train_fold = int(args.train_fold)
    if from_fold + train_fold > k_folds:
        train_fold = k_folds - from_fold
    modal = int(args.modal)

    # Fetch GPU numbers
    gpu_num = int(args.gpus) if 1 <= int(args.gpus) <= 4 else 1
    device = gpu_settings()
    logging.info(f"Using {gpu_num} GPUs")

    # Load data
    npz_names = glob.glob(os.path.join(args.data_dir, '*.npz'))
    if len(npz_names) == 0:
        logging.critical(f"No npz files found in {args.data_dir}")
        print("ERROR: Failed to load data")
        exit(-1)
    npz_names.sort()

    # Replace the problematic section
    # create the loading files list for each subject: [id1:[name1,name2],id2:[name3,name4],id3:[name5]]
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

    # Use list splitting instead of numpy array splitting
    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    npzs_list = split_list(npzs_list, k_folds)

    # Save as object array to handle variable-length lists
    save_dict = {'split': np.array(npzs_list, dtype=object)}
    np.savez(os.path.join(res_path, 'split.npz'), **save_dict)

    sleep_epoch_len = hyper_param_dict['sleep_epoch_len']

    # Loss function
    weighted_loss = WeightedCrossEntropyLoss(weight=hyper_param_dict['class_weights']).to(device)
    print(f"Loss weights: {hyper_param_dict['class_weights']}")

    # Result lists
    acc_list, val_acc_list = [], []
    loss_list, val_loss_list = [], []

    # Model initialization
    if modal == 0:
        model = SingleSalientModel(**hyper_param_dict).to(device)
    else:

        hyper_param_dict = hyper_params  # 假设这里是传递参数的地方
        print(hyper_param_dict)  # 添加调试信息
        model = TwoStreamSalientModel(**hyper_param_dict).to(device)

    if gpu_num > 1:
        model = nn.DataParallel(model)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=hyper_param_dict['train']['learning_rate'])
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5,
    verbose=True
    )

    # Save initial weights
    torch.save(model.state_dict(), 'weights.pth')

    # K-fold training and validation
    for fold in range(from_fold, from_fold + train_fold):
        logging.info(f"Starting validation: {fold + 1}/{k_folds}")
        print(f"{k_folds}-fold validation, turn: {fold + 1}")

        # Fix this line - remove .tolist() since npzs_list[fold] is already a list
        valid_npzs = list(itertools.chain.from_iterable(npzs_list[fold]))
        train_npzs = list(set(npz_names) - set(valid_npzs))

        logging.info("Loading data...")
        train_data_list, train_labels_list = load_npz_files(train_npzs)
        val_data_list, val_labels_list = load_npz_files(valid_npzs)
        logging.info("Data loaded")

        # Preprocess data
        logging.info("Preprocessing data...")
        train_data, train_labels = preprocess(train_data_list, train_labels_list, hyper_param_dict['preprocess'], True)
        val_data, val_labels = preprocess(val_data_list, val_labels_list, hyper_param_dict['preprocess'], True)
        logging.info("Preprocessing done")

        # Convert to PyTorch tensors
        train_data = [torch.FloatTensor(data).to(device) for data in train_data]
        val_data = [torch.FloatTensor(data).to(device) for data in val_data]
        train_labels = torch.LongTensor(train_labels).to(device)
        val_labels = torch.LongTensor(val_labels).to(device)

        # 在创建 TensorDataset 之前，记录维度信息到日志
        logging.info("Detailed shape information:")
        logging.info(f"train_data[0] (EEG) shape: {train_data[0].shape}")
        logging.info(f"train_data[1] (EOG) shape: {train_data[1].shape}")
        logging.info(f"train_labels shape: {train_labels.shape}")
        
        # 检查第一个维度（样本数）
        eeg_samples = train_data[0].shape[0]
        eog_samples = train_data[1].shape[0]
        label_samples = train_labels.shape[0]
        
        logging.info("Sample counts:")
        logging.info(f"EEG samples: {eeg_samples}")
        logging.info(f"EOG samples: {eog_samples}")
        logging.info(f"Label samples: {label_samples}")
        
        if not (eeg_samples == eog_samples == label_samples):
            logging.error("Sample dimension mismatch detected!")
            logging.error(f"EEG vs EOG diff: {abs(eeg_samples - eog_samples)}")
            logging.error(f"EEG vs Labels diff: {abs(eeg_samples - label_samples)}")
            logging.error(f"EOG vs Labels diff: {abs(eog_samples - label_samples)}")
            raise ValueError("Sample dimensions don't match")
        
        # Create datasets and dataloaders
        if modal == 0:
            train_dataset = TensorDataset(train_data[0], train_labels)
            val_dataset = TensorDataset(val_data[0], val_labels)
        else:
            train_dataset = TensorDataset(train_data[0], train_data[1], train_labels)
            val_dataset = TensorDataset(val_data[0], val_data[1], val_labels)

        train_loader = DataLoader(train_dataset, batch_size=hyper_param_dict['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyper_param_dict['train']['batch_size'])

        # Training loop
        best_val_acc = 0
        for epoch in range(hyper_param_dict['train']['epochs']):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
    
            for batch in train_loader:
                if modal == 0:
                    inputs, labels = batch
                    outputs = model(inputs)
                else:
                    eeg, eog, labels = batch
                    outputs = model((eeg, eog))
    
                # 在训练循环中添加
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad norm: {param.grad.norm()}")
                loss = weighted_loss(outputs, labels)
    
                # 添加调试信息
                print(f"Loss value: {loss.item()}")
                
                optimizer.zero_grad()
                loss.backward()
                
                # 检查梯度
                total_grad_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm()
                        total_grad_norm += grad_norm
                        if grad_norm > 0:
                            print(f"{name} grad norm: {grad_norm}")
                
                print(f"Total gradient norm: {total_grad_norm}")
                
                optimizer.step()
    
                train_loss += loss.item()
                
                # 修改这里的准确率计算方式
                _, predicted = outputs.max(2)  # 在类别维度上取最大值
                predicted = predicted.reshape(-1)  # 使用 reshape 而不是 view
                labels_flat = labels.reshape(-1)  # 使用 reshape 而不是 view
                train_total += labels_flat.size(0)
                train_correct += (predicted == labels_flat).sum().item()
    
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
    
            with torch.no_grad():
                for batch in val_loader:
                    if modal == 0:
                        inputs, labels = batch
                        outputs = model(inputs)
                    else:
                        eeg, eog, labels = batch
                        outputs = model((eeg, eog))
    
                    loss = weighted_loss(outputs, labels)
                    val_loss += loss.item()
                    
                    # 修改这里的准确率计算方式
                    _, predicted = outputs.max(2)  # 在类别维度上取最大值
                    predicted = predicted.view(-1)  # 展平预测结果
                    labels_flat = labels.view(-1)  # 展平标签
                    val_total += labels_flat.size(0)
                    val_correct += (predicted == labels_flat).sum().item()
    
                    # Correct reshaping for accuracy calculation - handle all dimensions
                    # 删除这个重复的准确率计算块，或者保留这个而删除上面的
                    # if len(outputs.shape) > 3:  # If shape is [batch, seq, classes, 1, 1]
                    #     batch_size, seq_len, num_classes = outputs.shape[0], outputs.shape[1], outputs.shape[2]
                    #     outputs_reshaped = outputs.view(batch_size * seq_len, num_classes)
                    #     labels_reshaped = labels.view(-1)  # Flatten all dimensions
                    # else:
                    #     batch_size, seq_len, num_classes = outputs.shape[0], outputs.shape[1], outputs.shape[2]
                    #     outputs_reshaped = outputs.view(batch_size * seq_len, num_classes)
                    #     labels_reshaped = labels.view(batch_size * seq_len)
                    
                    # _, predicted = outputs_reshaped.max(1)
                    # val_total += labels_reshaped.size(0)
                    # val_correct += predicted.eq(labels_reshaped).sum().item()
    
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
    
            scheduler.step(val_acc)  # Fixed missing parameter
    
            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(res_path, f"fold_{fold + 1}_best_model.pth"))
    
            # Log training and validation metrics
            print(f'Epoch: {epoch + 1}/{hyper_param_dict["train"]["epochs"]} | '
                  f'Train Loss: {train_loss / len(train_loader):.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss / len(val_loader):.4f} | '
                  f'Val Acc: {val_acc:.2f}%')

        # Clear GPU cache and reset model weights for the next fold
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load('weights.pth'))

        # Append results to lists
        acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        loss_list.append(train_loss / len(train_loader))
        val_loss_list.append(val_loss / len(val_loader))

    # Clear temporary weights file
    os.remove('weights.pth')

    # Return training history
    return {
        'acc': acc_list,
        'val_acc': val_acc_list,
        'loss': loss_list,
        'val_loss': val_loss_list
    }


def calculate_accuracy(outputs, targets):
    """计算分类准确率"""
    # 处理多维输出
    if len(outputs.shape) > 3:  # 5D tensor [batch, seq, classes, 1, 1]
        batch_size, seq_len, num_classes = outputs.shape[0], outputs.shape[1], outputs.shape[2]
        outputs_reshaped = outputs.view(batch_size * seq_len, num_classes)
        targets_reshaped = targets.view(-1)  # Flatten all dimensions
        _, predicted = outputs_reshaped.max(1)
        correct = predicted.eq(targets_reshaped).sum().item()
        total = targets_reshaped.size(0)
    elif len(outputs.shape) > 2:  # 3D tensor [batch, seq, classes]
        batch_size, seq_len = outputs.shape[0], outputs.shape[1]
        outputs_reshaped = outputs.view(batch_size * seq_len, -1)
        targets_reshaped = targets.view(batch_size * seq_len)
        _, predicted = outputs_reshaped.max(1)
        correct = predicted.eq(targets_reshaped).sum().item()
        total = targets_reshaped.size(0)
    else:
        # 获取预测的类别（最大概率的索引）
        _, predicted = torch.max(outputs.data, 1)  # 在类别维度上取最大值
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
    
    # 返回准确率
    return correct / total * 100.0


if __name__ == "__main__":
    print("Starting program...")
    # Set up GPU settings
    gpu_settings()

    # Parse arguments
    args = get_parser()

    # Load hyperparameters from YAML file
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.safe_load(f)
    print_params(hyper_params)

    # Train the model and get training history
    train_history = train(args, hyper_params)

    # Draw training plots
    draw_training_plot(train_history, int(args.from_fold) + 1, int(args.train_fold), args.output_dir)