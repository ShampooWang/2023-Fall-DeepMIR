import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoModelForCausalLM
from transformers import (
    MusicgenForCausalLM, 
    MusicgenForConditionalGeneration, 
    MusicgenDecoderConfig, 
    GenerationConfig, 
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)
import logging

logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    def __init__(self, model_type="gpt2", config=None, **kwargs) -> None:
        super().__init__()

        if model_type == "gpt2":
            self.config = GPT2Config(**config)
            self.model = GPT2LMHeadModel(self.config)
        elif model_type == "llama":
            self.config = LlamaConfig(**config)
            self.model = LlamaForCausalLM(self.config)
        else:
            raise NotImplementedError(model_type)
        
    def getTrainableParams(self) -> List[torch.Tensor]:
        """
        Returns:
            List[torch.Tensor]: model parameters
        """
        return list(self.model.parameters())

    def generate(self, x: torch.LongTensor, generation_config = None) -> Dict[str, torch.Tensor]:

        x = x.long()
        if hasattr(self.config, "num_codebooks"):
            x = x.unsqueeze(1)
        
        if generation_config is not None:
            generation_config = GenerationConfig(**generation_config)
            return self.model.generate(inputs=x, generation_config=generation_config)
        else:
            return self.model.generate(inputs=x)


    def forward(self, x: torch.LongTensor, y: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.LongTensor): midi

        Returns:
            Dict[torch.Tensor]: transformers.modeling_outputs.Seq2SeqLMOutput
                loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss.
                logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        """

        if hasattr(self.config, "num_codebooks"):
            x = x.unsqueeze(1)

        return self.model(input_ids=x)

