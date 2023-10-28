import logging

import os
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..base import OrderedNamespace
from .base_model import BaseLightningModel
from ..module import SSLAudioEncoder
from ..util import plot_confusion_matrix

logger = logging.getLogger(__name__)


class SingerClassifier(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.config = config
        self.audio_encoder = SSLAudioEncoder(**config.audio_encoder)
        self.mlp = nn.Sequential(
            nn.Linear(self.audio_encoder.out_dim, int(self.audio_encoder.out_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.audio_encoder.out_dim/2), 20)
        )
        self.critertion = nn.CrossEntropyLoss()
        self.singer_id2name = {}
        with open(os.path.join(config.data.dataset_root, "validation.json")) as f:
            data = json.load(f)
            for d in data:
                singer_name = d["path"].split("/")[-4]
                if d["singer_id"] not in self.singer_id2name:
                    self.singer_id2name[d["singer_id"]] = singer_name

    def forward(self, batch: dict):
        wav = batch["wav"]
        audio_features = self.audio_encoder(wav).mean(1)
        logits = self.mlp(audio_features)

        return logits
    
    def compute_loss(self, logits: torch.tensor, target: torch.tensor) -> float:
        return self.critertion(logits, target)

    def compute_acc(self, logits: torch.tensor, target: torch.tensor) -> dict:
        B = logits.shape[0]
        top1_acc = top3_acc = 0
        for predict, label in zip(logits, target):
            top1_acc += (label == torch.topk(predict, 1)[1])
            top3_acc += (label in torch.topk(predict, 3)[1])

        return {"top1_acc": top1_acc / B, "top3_acc": top3_acc / B}

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)        
        
        assert "singer_id" in batch, f"{batch.keys()}"
        target = batch["singer_id"]
        loss = self.compute_loss(logits, target)
        acc_result = self.compute_acc(logits, target)
        log_metrics = {"loss": loss, **acc_result}
        log_metrics = {f"train_{k}": log_metrics[k] for k in log_metrics}

        self.log_dict(
            log_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        return {
            "loss": loss,
            "song_id": batch["song_id"],
            "singer_id": batch["singer_id"],
            "logits": logits,
        }

    def train_epoch_end(self, outputs):
        song_to_singer = defaultdict(int)
        song_top1_pred = defaultdict(list)
        song_top3_pred = defaultdict(list)
        confusion_matrix = torch.zeros([20, 20], device=self.device)
        
        for output in outputs:
            for song, singer, logit in zip(output["song_id"], output["singer_id"], output["logits"]):
                song_to_singer[song.item()] = singer
                song_top1_pred[song.item()] += torch.topk(logit, 1)[1].tolist()
                song_top3_pred[song.item()] += torch.topk(logit, 3)[1].tolist()

        log_metrics = {"top1_acc": 0.0, "top3_acc": 0.0}
        labels = prediction = []
        for song, pred in song_top1_pred.items():
            label = song_to_singer[song]
            maj_pred = max(set(pred), key=pred.count)
            if label == maj_pred: 
                log_metrics["top1_acc"] += 1 / len(song_top1_pred)
            confusion_matrix[label][maj_pred] += 1
        

        output_img_dir = os.path.join(self.config.trainer.default_root_dir, "images", "training")
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
        plot_confusion_matrix(confusion_matrix, [ n for n in dict(sorted(self.singer_id2name.items())).values() ], os.path.join(output_img_dir, f"confusion_mtx_ep{self.current_epoch}"))

        for song, pred in song_top3_pred.items():
            label = song_to_singer[song]
            pred = torch.LongTensor(pred)
            maj_pred = torch.topk(pred.bincount(), 3)[1]
            if label in maj_pred.to(label.device): 
                log_metrics["top3_acc"] += 1 / len(song_top3_pred)

        log_metrics = {f"train_{k}": log_metrics[k] for k in log_metrics}
        self.log_dict(
            log_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
      

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)        
        
        assert "singer_id" in batch, f"{batch.keys()}"
        target = batch["singer_id"]
        loss = self.compute_loss(logits, target)
        acc_result = self.compute_acc(logits, target)
        log_metrics = {"loss": loss, **acc_result}
        log_metrics = {f"val_{k}": log_metrics[k] for k in log_metrics}

        self.log_dict(
            log_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        return {
            "loss": loss,
            "song_id": batch["song_id"],
            "singer_id": batch["singer_id"],
            "logits": logits,
        }

    def validation_epoch_end(self, outputs):
        song_to_singer = defaultdict(int)
        song_top1_pred = defaultdict(list)
        song_top3_pred = defaultdict(list)
        confusion_matrix = torch.zeros([20, 20], device=self.device)
        
        for output in outputs:
            for song, singer, logit in zip(output["song_id"], output["singer_id"], output["logits"]):
                song_to_singer[song.item()] = singer
                song_top1_pred[song.item()] += torch.topk(logit, 1)[1].tolist()
                song_top3_pred[song.item()] += torch.topk(logit, 3)[1].tolist()

        log_metrics = {"top1_acc": 0.0, "top3_acc": 0.0}
        labels = prediction = []
        for song, pred in song_top1_pred.items():
            label = song_to_singer[song]
            maj_pred = max(set(pred), key=pred.count)
            if label == maj_pred: 
                log_metrics["top1_acc"] += 1 / len(song_top1_pred)
            confusion_matrix[label][maj_pred] += 1
        

        output_img_dir = os.path.join(self.config.trainer.default_root_dir, "images", "validation")
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
        plot_confusion_matrix(confusion_matrix, [ n for n in dict(sorted(self.singer_id2name.items())).values() ], os.path.join(output_img_dir, f"confusion_mtx_ep{self.current_epoch}"))

        for song, pred in song_top3_pred.items():
            label = song_to_singer[song]
            pred = torch.LongTensor(pred)
            maj_pred = torch.topk(pred.bincount(), 3)[1]
            if label in maj_pred.to(label.device): 
                log_metrics["top3_acc"] += 1 / len(song_top3_pred)

        log_metrics = {f"val_{k}": log_metrics[k] for k in log_metrics}
        self.log_dict(
            log_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
      
    def configure_optimizers(self):
        params = []
        params += self.audio_encoder.getTrainableParams()
        params += list(self.mlp.parameters())
        optimizer = getattr(torch.optim, self.config.optimizer.name)(
            params,
            **self.config.optimizer.args,
        )

        return optimizer