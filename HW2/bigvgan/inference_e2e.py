# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, spectral_normalize_torch
from models import BigVGAN as bigVGANGenerator
from pathlib import Path
import numpy as np
from tqdm import tqdm

h = None
device = None
torch.backends.cudnn.benchmark = False

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = bigVGANGenerator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = [ str(f) for f in Path(a.test_dir).glob('**/*.npy') ]

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(tqdm(filelist)):
            # load the ground truth audio and resample if necessary
            # wav, sr = librosa.load(os.path.join(a.input_wavs_dir, filname), h.sampling_rate, mono=True)
            # wav = torch.FloatTensor(wav).to(device)
            # compute mel spectrogram from the ground truth audio
            # x = get_mel(wav.unsqueeze(0))
            
            x = torch.from_numpy(np.load(filname)).to(device).unsqueeze(0)
            x = spectral_normalize_torch(x)

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, f"{filname.split('/')[-1].split('.')[0]}.wav")
            write(output_file, h.sampling_rate, audio)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/test')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', default="../exp/bigvgan_pretrained_spec_norm/g_ep40", required=True)

    a = parser.parse_args()

    assert "g_ep" in a.checkpoint_file, a.checkpoint_file

    config_file = os.path.join(a.checkpoint_file.split("g_ep")[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()



