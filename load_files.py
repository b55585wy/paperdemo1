import numpy as np
from typing import List, Tuple
import torch


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
    Load data and labels for training and validation

    :param npz_files: a list of str for npz file names

    :return: the lists of data and labels
    """
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # Convert np.ndarray to torch.Tensor
        tmp_data = torch.tensor(tmp_data, dtype=torch.float32)
        tmp_labels = torch.tensor(tmp_labels, dtype=torch.int32)

        # 维度处理
        tmp_data = torch.squeeze(tmp_data)  # [None, W, C]
        tmp_data = tmp_data[:, :, :, None, None]  # [None, W, C, 1, 1]
        
        # 分离并连接通道
        tmp_data = torch.cat([
            tmp_data[:, :, i, :, :].unsqueeze(0) for i in range(3)
        ], dim=0)  # [C, None, W, 1, 1]

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"Loaded {len(data_list)} files totally.")

    return data_list, labels_list