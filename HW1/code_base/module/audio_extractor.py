from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Union, List, Tuple
import logging
from ..util import freeze_model, get_keypadding_mask
from .weighted_sum import WeightedSumLayer

logger = logging.getLogger(__name__)

class SSLAudioEncoder(nn.Module):
    def __init__(self, model_type: str, input_sr=44100, trainable=False, feat_select_idx: str = "weighted_sum") -> None:
        super().__init__()

        self.model_type = model_type
        self.input_sr = input_sr
        self.trainable = trainable
        self.feat_select_idx = feat_select_idx

        if "MERT" in model_type:
            self.encoder = AutoModel.from_pretrained(f"m-a-p/{self.model_type}", trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"m-a-p/{self.model_type}",trust_remote_code=True)
        elif "wavlm" in model_type:
            self.encoder = AutoModel.from_pretrained(f"microsoft/{self.model_type}", trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"microsoft/{self.model_type}",trust_remote_code=True)

        logger.info(f"Using {self.model_type} ssl model as audio encoder")
        with torch.no_grad():
            wav = torch.randn(self.processor.sampling_rate, dtype=torch.float, device="cpu")
            inputs = self.processor(wav, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
            outputs = self.encoder(**inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(0)
            self.n_hidden_states = all_layer_hidden_states.shape[0]
            self.out_dim = all_layer_hidden_states.shape[-1]

        assert self.feat_select_idx in ["weighted_sum", "last"], f"valid options: {['weighted_sum', 'last']}"
        if self.feat_select_idx == "weighted_sum":
           self.trainable = False
           self.weighted_sum_layer = WeightedSumLayer(self.n_hidden_states)
           logger.info(f"Using weighted sum layer for {self.n_hidden_states} hidden states")
        else:
            self.trainable = True

        if not self.trainable:
            freeze_model(self.encoder)
            self.encoder.eval()

    def getTrainableParams(self) -> list:
        if self.trainable:
            return list(self.encoder.parameters())
        else:
            return list(self.weighted_sum_layer.parameters())
        
    def preprocess_wav(self, wav: list) -> dict:
        wav_len = [ len(w) for w in wav ]
        wav = torch.stack(wav, dim=0)
        return {"input_values": wav, "attention_mask": ~get_keypadding_mask(max(wav_len), torch.LongTensor(wav_len))}

    def forward(self, wav: list) -> torch.tensor:
        inputs = self.preprocess_wav(wav)
        for k in inputs:
            inputs[k] = inputs[k].to(self.encoder.device)
        outputs = self.encoder(**inputs, output_hidden_states=True)

        if hasattr(self, "weighted_sum_layer"):
            return self.weighted_sum_layer(outputs.hidden_states)
        else:
            return outputs.last_hidden_state