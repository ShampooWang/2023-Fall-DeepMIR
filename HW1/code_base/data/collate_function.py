from typing import Tuple
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_general(batch: Tuple):
    """
    Args:
        batch (Tuple): output of dataset's __getitem__
    Returns:
        return_dict (dict): 
    """
    keysInBatch = list(batch[0].keys())
    return_dict = {k: [] for k in keysInBatch}

    for row in batch:
        for key in keysInBatch:
            assert key in row.keys(), f"key: {key}, keys: {row.keys()}"
            return_dict[key].append(row[key])

    for key in return_dict:
        if key != "wav":
            return_dict[key] = torch.LongTensor(return_dict[key])

    return return_dict