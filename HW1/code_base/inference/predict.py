import os
import sys
import csv
import argparse
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model import SingerClassifier
from ..data import AudioDataset, collate_general
from typing import List, Dict


def predict_singer_csv(dataset_root, ckpt, json_prefix, batch_size, output_csv):
    dataset = AudioDataset(
        dataset_root=dataset_root,
        split=json_prefix,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_general,
        num_workers=4,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SingerClassifier.load_from_checkpoint(ckpt).to(device)
    model.eval()
    result_dict = defaultdict(list)

    for data in tqdm(dataloader):
        logits = model(data)
        for song_id, logit in zip(data["song_id"], logits):
            result_dict[song_id.item()] += (torch.topk(logit, 3, sorted=True)[1]).tolist()
            del song_id, logit
        del data, logits

    sorted_result = dict(sorted(result_dict.items()))
    with open(output_csv, "w") as f:
        writer = csv.writer(f)
        for song_id, pred_id in sorted_result.items():
            pred_id = torch.LongTensor(pred_id)
            maj_pred = torch.topk(pred_id.bincount(), 3, sorted=True)[1]
            row = [song_id] + [ model.singer_id2name[p_id.item()] for p_id in maj_pred ]
            writer.writerow(row)