import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import subprocess
import soundfile as sf


def source_separate_and_remove_silence_and_segment(target_dir: str, output_dir: str): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_path_list = [ str(f) for f in Path(target_dir).glob("**/*.wav") ] + [ str(f) for f in Path(target_dir).glob("**/*.mp3") ]

    for audio_path in tqdm(audio_path_list):
        song_name = audio_path.split("/")[-1].split(".")[0]

        # source separate
        subprocess.run(["demucs", audio_path, "-n", "htdemucs", "--two-stems", "vocals", "-o", output_dir]) # output_dir/htdemucs/song_name/[vocals.wav, no_vocals.wav]

        # remove_silence
        vocal_wav_path = os.path.join(output_dir, "htdemucs", song_name, "vocals.wav")
        rm_sil_output_path = os.path.join(output_dir, f"{song_name}.wav")
        subprocess.run(["unsilence", vocal_wav_path, rm_sil_output_path, "-ao", "-y", "-ss", "36", "-dci", "-t", "16"]) # output_dir/song_name.wav

        # segment song into 5 sec. of snippets
        segment_output_subdir = os.path.join(output_dir, song_name) # output_dir/song_name/[seg0.wav, seg1.wav, ...]
        if not os.path.exists(segment_output_subdir):
            os.makedirs(segment_output_subdir)
        data, sr = sf.read(rm_sil_output_path)
        for i in range(0, len(data) // (5 * sr)):
            start_idx, end_idx = i * sr * 5, (i+1) * sr * 5
            segmented_data = data[start_idx:end_idx]
            if segmented_data.shape[0] > 0:
                sf.write(os.path.join(segment_output_subdir, f"seg{i}.wav"), segmented_data, sr)
        subprocess.run(["rm", "-rf", rm_sil_output_path])    

    subprocess.run(["rm", "-rf", os.path.join(output_dir, "htdemucs")])