import os
import glob
import numpy as np
import parselmouth
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


# --- hyperparameters --- #
# for pitch detection
pitch_time_step = 0.005
pitch_floor = 60
max_candidates = 15
very_accurate = False
silence_thresh = 0.03
voicing_thresh = 0.8 # higher is fewer unvoiced intervals
octave_cost = 0.01
oct_jump_cost = 0.35
vuv_cost = 0.14
pitch_ceiling = 600.0
max_period = 0.02

# for summing
fade_length = 100 # fade length in samples. This avoids discontinuities

def apodize(values, minidx, maxidx, length):
    values[minidx-length:minidx] *= np.linspace(1.0,0.0,length)
    values[minidx:maxidx] = 0.0
    values[maxidx:maxidx+length] *= np.linspace(0.0,1.0,length)


def debuzz_waves(target_dir):
    path_part_dir = os.path.join(target_dir, "part")
    path_pred_list = [ str(f) for f in Path(os.path.join(target_dir, "pred")).glob('**/*.wav') ]

    print(f'Start debuzzing wave files')
    for path_pred in tqdm(path_pred_list):
        bn = os.path.basename(path_pred)
        path_harmonic = os.path.join(path_part_dir, bn).replace('.wav', '-harmonic.wav')
        path_noise    = os.path.join(path_part_dir, bn).replace('.wav', '-noise.wav')
        
        # load wave
        wave_pred = parselmouth.Sound(path_pred)
        wave_harmonic = parselmouth.Sound(path_harmonic) if os.path.exists(path_harmonic) else None
        wave_noise = parselmouth.Sound(path_noise) if os.path.exists(path_noise) else None
        
        if wave_harmonic is not None and wave_noise is not None: 
            # detect UV (unvoiced) intervals
            pitch = wave_pred.to_pitch_ac(
                pitch_time_step, 
                pitch_floor,
                max_candidates,
                very_accurate,
                silence_thresh,
                voicing_thresh,
                octave_cost,
                oct_jump_cost,
                vuv_cost,
                pitch_ceiling)
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values==0] = np.nan
            UV_Indices = np.argwhere(np.isnan(pitch_values)).flatten()
            
            # apply mask on harmonic signal during UV intervals
            step = int(pitch_time_step/2 * wave_harmonic.sampling_frequency) + 1
            for index in UV_Indices:
                h_index = (np.abs(wave_harmonic.xs() - pitch.xs()[index])).argmin() # upsample f0 to sample level
                apodize(wave_harmonic.values[0], h_index-step, h_index+step, length=fade_length)

            # the first and last 0.25 seconds don't have pitch detection, so mute these
            trim = int(wave_harmonic.sampling_frequency * 0.25)+1
            wave_harmonic.values[0][:trim] = 0
            wave_harmonic.values[0][-trim:] = 0
            
            # combine harmonic and noise signals
            wave_final = wave_harmonic.values + wave_noise.values
            
            # save
            sf.write(path_pred, np.squeeze(wave_final), int(wave_harmonic.sampling_frequency))