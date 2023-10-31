import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf
from functools import partial
from vocoder_eval.evaluate import evaluate

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_data_root', default="/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/valid", type=str)
    parser.add_argument('--output_dir', default="/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/gl_pred", type=str)

    return parser.parse_args(argv)

def griffinLim(wav_path, output_dir):
    wav_name = wav_path.split("/")[-1].split(".")[0]
    y, sr = librosa.load(wav_path)
    S = np.abs(librosa.stft(y))
    y_inv = librosa.griffinlim(S, random_state=322)
    output_path = os.path.join(output_dir, f"{wav_name}.wav")
    sf.write(output_path, y_inv, sr)

    return [wav_path, output_path]


def main(argv):
    args = parseArgs(argv)
    valid_data_path = [ str(f) for f in Path(args.valid_data_root).glob('**/*.wav') ]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with Pool(32) as pool:
        gl_func = partial(griffinLim, output_dir=args.output_dir)
        path_list = list(tqdm(pool.imap(gl_func, valid_data_path), total=len(valid_data_path)))

    y_path_list = [ p[0] for p in path_list ]
    y_g_hat_path_list = [ p[1] for p in path_list ]
    RESULT = evaluate(y_path_list, y_g_hat_path_list)

    print(RESULT)

if __name__ == "__main__":
    main(sys.argv[1:])