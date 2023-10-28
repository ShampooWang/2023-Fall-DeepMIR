import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def get_keypadding_mask(max_length: int, data_lens: torch.Tensor) -> torch.Tensor:
    """Create keypadding mask for attention layers

    Args:
        max_length (int): the max sequence length of the batch
        audio_len (torch.Tensor): the lens for each data in the batch, shape = (bsz,)

    Returns:
        torch.Tensor: key_padding_mask, bool Tensor, True for padding
    """
    bsz = data_lens.shape[0]
    key_padding_mask = torch.ones([bsz, max_length])
    for mask, len in zip(key_padding_mask, data_lens):
        mask[:len] = 0.0
    key_padding_mask = key_padding_mask.type_as(data_lens).bool()

    return key_padding_mask

def plot_confusion_matrix(confusion_matrix, labels, output_path):
    confusion_matrix = confusion_matrix.cpu().numpy()
    for i in range(confusion_matrix.shape[0]):
        if confusion_matrix[i].sum(-1) != 0:
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum(-1) 
    df_cm = pd.DataFrame(confusion_matrix, index = labels,
                  columns = labels)
    plt.figure(figsize = (20, 20))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(output_path)
