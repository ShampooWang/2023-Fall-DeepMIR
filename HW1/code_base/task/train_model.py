from ..model import (
    BaseLightningModel,
    SingerClassifier
)
from .base_task import BaseTask, TrainGeneral

class Train4Test(TrainGeneral):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(BaseLightningModel)

class TrainSingerClassifier(TrainGeneral):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(SingerClassifier)