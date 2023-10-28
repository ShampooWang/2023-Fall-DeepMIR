import numpy as np
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import json
import logging

logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    def __init__(self, 
                 dataset_root: str, 
                 split: str, 
                 input_sr = 44100,
                 output_sr=24000,
                 normalize_waveform=False,
                 origin_song=False,
                 **kwargs):

        self.dataset_root = dataset_root
        self.split = split
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.origin_song = origin_song
        
        if input_sr != output_sr:
            logger.info(f"Resample the input audio from {self.input_sr} -> {self.output_sr}")
        self.normalize_waveform = normalize_waveform

        if origin_song:
            json_path = os.path.join(dataset_root, f"{split}_origin.json")
        else:
            json_path = os.path.join(dataset_root, f"{split}.json")

        logger.info(f"Using original song: {origin_song}")

        with open(json_path, "r") as f:
            self.data = json.load(f)
            logger.info(f"Loading {len(self.data)} of singer data")
        
    def __getitem__(self, index):
        batch_dict = {}
        for k, v in self.data[index].items():
            if k == "path":
                if self.input_sr != self.output_sr:
                    data = torch.FloatTensor(librosa.load(v, sr=self.output_sr)[0])
                else:
                    data = torch.FloatTensor(librosa.load(v)[0])
                if data.ndim == 2:
                    data = data.mean(-1)
                assert data.ndim == 1, data.ndim
                if self.normalize_waveform:
                    data = F.layer_norm(data, data.shape)
                batch_dict["wav"] = data
            else: # song_id, singer_id
                batch_dict[k] = v
        return batch_dict
    
    def __len__(self):
        return len(self.data)