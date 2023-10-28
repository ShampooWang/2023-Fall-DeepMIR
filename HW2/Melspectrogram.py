import numpy as np
import torch
import torchaudio
import librosa
import os

num_mels=80
n_fft=1024
hop_size=256
win_size=1024
sampling_rate=22050
fmin=0
fmax=8000

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    device = y.device
    melTorch = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, \
           hop_length=hop_size, win_length=win_size, f_min=fmin, f_max=fmax, pad=int((n_fft-hop_size)/2), center=center).to(device)      
    spec = melTorch(y)
    return spec

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio

def load_audio(audio_path, sr=None, mono=True):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path)
    audio = to_mono(audio) if mono else audio
    
    if sr and org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio

if __name__ == '__main__':
    load_audio_path = '/path/to/your/audio/dir'
    save_npy_path = '/path/you/want/to/save/mel_npy'
    if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)
    audio_list = os.listdir(load_audio_path)
    audio_list.sort()
    for audio in audio_list:
        y = load_audio(os.path.join(load_audio_path, audio), sr=sampling_rate)
        mel_tensor = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        mel = mel_tensor.squeeze().cpu().numpy()
        file_name = os.path.join(save_npy_path, audio[:-4]+'.npy')
        np.save(file_name, mel)
        mel = np.load(file_name) # check the .npy is readable

    # plot the last melspectrogram
    # ref: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # don't forget to do dB conversion
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sampling_rate,
                             fmax=fmax, ax=ax, hop_length=hop_size, n_fft=n_fft)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')