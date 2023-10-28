import os
import random
import numpy as np
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import librosa
import pyworld as pw
from collections import defaultdict
from typing import Tuple


def traverse_dir(
        root_dir,
        extension,
        amount=None,
        is_sort=False,):

    file_list = [ str(f) for f in Path(root_dir).glob(f'**/*.{extension}') ]

    if is_sort:
        file_list.sort()

    if amount is not None and amount >= 0:
        return file_list[:amount]
    else:
        return file_list

def collate_fn(batch: Tuple) -> dict:
    keysInBatch = batch[0].keys()
    return_dict = defaultdict(list)

    for instance in batch:
        for key in keysInBatch:
            assert key in instance.keys(), f"{instance.keys()}"
            return_dict[key].append(instance[key])

    for key in return_dict.keys():
        if isinstance(return_dict[key][0], torch.Tensor):
            return_dict[key] = pad_sequence(return_dict[key], batch_first=True)

    return return_dict



def get_data_loaders(args, whole_audio=True):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        whole_audio=whole_audio)
    
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        whole_audio=True)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        sample_rate,
        whole_audio=False,
        hop_size=256,
        num_mels=80,
        n_fft=1024,
        win_size=1024,
        fmin=0,
        fmax=8000,
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate

        self.hop_size = hop_size
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax

        self.path_root = path_root
        self.paths = traverse_dir(
            path_root,
            extension='wav',
            is_sort=True,
        )
        self.whole_audio = whole_audio

    def __getitem__(self, file_idx):
        path_audio = self.paths[file_idx]

        # check duration. if too short, then skip
        duration = librosa.get_duration(
            path=path_audio, 
            sr=self.sample_rate)
            
        if duration < (self.waveform_sec + 0.1):
            return self.__getitem__(file_idx+1)
        
        # get item
        return self.get_data(path_audio, duration)

    def get_data(self, path_audio, duration):
        name = path_audio.split("/")[-1].split(".")[0]

        # load audio
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)

        audio, sr = librosa.load(
                path_audio, 
                sr=self.sample_rate, 
                offset=idx_from,
                duration=waveform_sec)
        
        # clip audio into N seconds
        frame_resolution = (self.hop_size / self.sample_rate)
        frame_rate_inv = 1/frame_resolution
        audio = audio[...,:audio.shape[-1]//self.hop_size*self.hop_size]

        def mel_spectrogram(y, center=False):
            melTorch = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, \
                hop_length=self.hop_size, win_length=self.win_size, f_min=self.fmin, f_max=self.fmax, pad=int((self.n_fft-self.hop_size)/2), center=center)      
            spec = melTorch(torch.FloatTensor(y)) # n_mel x n_frames

            return spec.permute(1, 0) # n_frames x n_mel
        
        audio_mel = mel_spectrogram(audio).float()
        
        # extract f0
        f0, _ = pw.dio(
            audio.astype('double'), 
            self.sample_rate, 
            f0_floor=65.0, 
            f0_ceil=1047.0, 
            channels_in_octave=2, 
            frame_period=(1000*frame_resolution))
        f0 = f0.astype('float')[:audio_mel.size(0)]
        f0_hz = torch.from_numpy(f0).float().unsqueeze(-1)
        f0_hz[f0_hz<80]*= 0

        # out 
        audio = torch.from_numpy(audio).float()
        assert sr == self.sample_rate

        return dict(audio=audio, f0=f0_hz, mel=audio_mel, name=name)

    def __len__(self):
        return len(self.paths)