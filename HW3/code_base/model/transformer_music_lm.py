import logging

import os
import json
from collections import defaultdict
from typing import Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..base import OrderedNamespace
from .base_model import BaseLightningModel
from ..module import TransformerModel
from ..optim.scheduler import get_scheduler
from ..data.utils import write_midi
import pickle
from tqdm import tqdm
from midi2audio import FluidSynth


logger = logging.getLogger(__name__)


class TransformerMusicLM(BaseLightningModel):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)
        if hasattr(self.config, "scheduler"):
            self.config.scheduler.max_step = self.config.trainer.max_steps
            self.config.scheduler.warmup = int(self.config.scheduler.warmup * self.config.scheduler.max_step)
            logger.info(f"max_steps {self.config.scheduler.max_step}, warmup steps {self.config.scheduler.warmup}")

        model_name = config.transformer_model.get("model_name", None)
        model_confg = config.transformer_model.get("config", None)
        self.transformer_model = TransformerModel(model_name=model_name, config=model_confg)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: np.array):
        x = batch[:, 0, :]
        output_logits = self.transformer_model(x).logits

        return output_logits

    def generate(self, 
        generation_num: int = 1, 
        n_target_bar: int = 32, 
        output_dir: str = None, 
        generation_config: dict = None):

        if generation_config is None:
            generation_config = self.config.get("generation", None)

        if output_dir is None:
            output_dir = os.path.join(self.config.trainer.default_root_dir, "generated_examples")

        if not os.path.exists(os.path.join(output_dir, "midi")):
            os.makedirs(os.path.join(output_dir, "midi"))
        
        if not os.path.exists(os.path.join(output_dir, "audio")):
            os.makedirs(os.path.join(output_dir, "audio"))

        dict_path = self.config.data.dict_path
        event2word, word2event = pickle.load(open(dict_path, 'rb'))

        print('Start generating')
        fs = FluidSynth()
        self.transformer_model.eval()
        with torch.no_grad():
            for n_idx in tqdm(range(generation_num)):
                words = []
                ws = [event2word['Bar_None']]
                if 'chord' in dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

                # generate
                temp_x = torch.LongTensor(words[0]).unsqueeze(0)
                bar_num = 1
                while bar_num <= (n_target_bar + 1):
                    if temp_x.shape[1] >= self.config.generation.max_length:
                        truncate_start = temp_x.shape[1] - self.config.generation.max_length
                        temp_x = temp_x[:, truncate_start + 1:]

                    generate_toks = self.transformer_model.generate(temp_x.to(self.device), generation_config)
                    if generate_toks.nelement() == 0:
                        temp_x = torch.LongTensor(words[0]).unsqueeze(0)
                    else:    
                        temp_x = generate_toks
                        word = generate_toks.squeeze(0).cpu().detach()
                        is_bar_arr = (word == event2word['Bar_None'])
                        if bar_num + is_bar_arr.sum() < (n_target_bar + 1):
                            words.append(word.numpy())
                            bar_num += is_bar_arr.sum()
                        else:
                            for i, is_bar in enumerate(is_bar_arr):
                                if is_bar:
                                    bar_num += 1
                                    if bar_num == (n_target_bar + 1):
                                        words.append(word[:i].numpy())
                                        break
                

                words = np.concatenate(words, axis=0)
                words = words[(words < 249).nonzero()] # remove pad, bos, eos

                write_midi(
                    words=words,
                    word2event=word2event,
                    output_path=os.path.join(output_dir, "midi", f"{n_idx}.mid"),
                    prompt_path=None
                )
                fs.midi_to_audio(os.path.join(output_dir, "midi", f"{n_idx}.mid"), os.path.join(output_dir, "audio", f"{n_idx}.wav"))

    def training_step(self, batch, batch_idx):
        y = batch[:, 1, :]
        output_logits = self.forward(batch)
        loss = loss = self.criterion(output_logits.permute(0,2,1), y)
        log_metrics = {"loss": loss}

        self.log_dict(
            log_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return log_metrics

    def on_train_epoch_end(self):
        self.generate()
      
    def configure_optimizers(self):
        params = self.transformer_model.getTrainableParams()
        optimizer = getattr(torch.optim, self.config.optimizer.name)(
            params,
            **self.config.optimizer.args,
        )
        scheduler = {
            "scheduler": get_scheduler(optimizer=optimizer, **self.config.scheduler),
            "interval": "step"
        }

        return [optimizer], [scheduler]