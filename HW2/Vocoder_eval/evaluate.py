import argparse
import auraloss
import json
import librosa
import numpy as np
import os
import torch
import torchaudio as ta
import crepe

from .fad import FrechetAudioDistance
from tqdm import tqdm

SR_TARGET = 22050
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    audio, sampling_rate = librosa.core.load(full_path, sr=SR_TARGET)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    return audio

def equalize_audio_length(audio1, audio2):
    len1 = audio1.size(1)
    len2 = audio2.size(1)

    num_zeros_to_add = abs(len1 - len2)
    if not num_zeros_to_add:
        return audio1, audio2
    
    zeros_to_add = torch.zeros(1, num_zeros_to_add)
    if len1 > len2:
        audio2 = torch.cat((audio2, zeros_to_add), dim=1)
    else:
        audio1 = torch.cat((audio1, zeros_to_add), dim=1)
    
    return audio1, audio2

    
def Log2f0_mean(frequency_true, frequency_pred):
    total_error = 0
    total = len(frequency_pred)
    for f_true, f_pred in zip(frequency_true, frequency_pred):
        total_error += abs(np.log2(f_true) - np.log2(f_pred)) * 12
        
    return total_error / total
    
def evaluate(y_path_list, y_g_hat_path_list):
    """Perform objective evaluation"""
    gpu = 0 if torch.cuda.is_available() else None    
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    print(f'Using {device}')
    torch.cuda.empty_cache()

    mrstft_tot = 0.0
    f0_mean_tot = 0

    resampler_16k = ta.transforms.Resample(SR_TARGET, 16000).to(device)

    # Modules for evaluation metrics
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)

    _M_STFT = True
    _F0 = True
    _FAD = True
    
    with torch.no_grad():
        for y_path, y_g_hat_path in tqdm(zip(y_path_list, y_g_hat_path_list)):
            if _M_STFT or _F0:
                y = load_wav(y_path)
                y_g_hat = load_wav(y_g_hat_path)
                y, y_g_hat = equalize_audio_length(y, y_g_hat)
                y = y.to(device)
                y_g_hat = y_g_hat.to(device)        
            
            if _M_STFT:
                # MRSTFT calculation
                mrstft_tot += loss_mrstft(y_g_hat.unsqueeze(0), y.unsqueeze(0)).item()
                
            if _F0:
                y_16k = (resampler_16k(y)[0] * MAX_WAV_VALUE).short().cpu().numpy()
                y_g_hat_16k = (resampler_16k(y_g_hat)[0] * MAX_WAV_VALUE).short().cpu().numpy()                
                _, frequency_true, confidence, _ = crepe.predict(y_16k, 16000, viterbi=True, verbose=0, model_capacity='medium')
                _, frequency_pred, _, _ = crepe.predict(y_g_hat_16k, 16000, viterbi=True, verbose=0, model_capacity='medium')
                filtered_data = [(true, pred) for true, pred, conf in zip(frequency_true, frequency_pred, confidence) if conf > 0.6 and 50<true<2100 ]
                filtered_true, filtered_pred = zip(*filtered_data)
                f0_mean_tot += Log2f0_mean(filtered_true, filtered_pred)
                
    RETURN = {}    
    
    if _M_STFT:
        RETURN['M-STFT'] = mrstft_tot / len(y_path_list)        

    if _F0:
        RETURN['log2f0_mean'] = f0_mean_tot / len(y_path_list)
        
    if _FAD:
        frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        frechet.model = frechet.model.to(torch.device('cpu' if gpu is None else f'cuda:{0}'))
        fad_score = frechet.score(y_g_hat_path_list, y_path_list, limit_num=None, recalculate=True)
        RETURN['FAD'] = fad_score['frechet_audio_distance']

    return RETURN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    a = parser.parse_args()

    gt_dir = '/path/to/m4singer_valid'
    synth_dir = '/path/to/your/vocoder_output_dir'+a.model

    results = evaluate(gt_dir, synth_dir)
    print(results)

    with open(f'score_{a.model}.txt', 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
