from ..model import (
    TransformerMusicLM
)
from .base_task import BaseTask, TrainGeneral

class TrainTransformerMusicLM(TrainGeneral):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(TransformerMusicLM)