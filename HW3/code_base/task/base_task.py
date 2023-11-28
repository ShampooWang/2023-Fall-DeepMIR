import abc
import argparse
import os

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split

from ..base import OrderedNamespace
from ..data import (
    NewsDataset,
    collate_general
)
from ..util import add_general_arguments, set_logging, set_pl_logger
from ..util import compute_grad_norm


class BaseTask:
    def __init__(self):
        self.args = None
        self.config = None

    @abc.abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError


class TrainGeneral(BaseTask):
    def __init__(self):
        super().__init__()

    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_general_arguments(parser)
        return parser

    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        args = parser.parse_args()

        if not torch.cuda.is_available():
            args.device = "cpu"
            args.gpus = 0

        self.args = args
        set_logging(args)

        return args

    def run(self, model_cls, custom_trainer_callbacks=[]):
        assert self.args is not None
        seed_everything(self.args.seed)
        
        ########################################
        ## Add model and model configurations ##
        ########################################
        if self.args.resume_ckpt != "":
            model = model_cls.load_from_checkpoint(self.args.ckpt)
            if self.args.save_path != "":
                model.config.save_path = self.args.save_path
            config = model.config
        else:
            self.args.ckpt = None
            config = yaml.load(open(self.args.config, "r"), Loader=yaml.FullLoader)
            config = OrderedNamespace([self.args, config])
            model = model_cls(config)
            
        if config.save_path != "":
            config.trainer.default_root_dir = config.save_path
            
        config.trainer.logger = set_pl_logger(config)
        config.gpus = self.args.gpus
        self.config = config
        
        #######################################################
        ## Add datasets for training, validation and testing ##
        #######################################################
        if self.args.train:
            train_set = NewsDataset(
                **config.data,
            )
            train_loader = DataLoader(
                train_set,
                batch_size=config.data.train_batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
            )
            
        if self.args.test:
            test_set = NewsDataset(
                **config.data,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=config.data.dev_batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
            )
            
        #################################
        ## Set up trainer for training ##
        #################################
        model_checkpoint_train_loss = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{loss:.4f}",
            monitor="loss",
            save_top_k=3,
            mode="min",
            every_n_epochs=1,
            save_last=True,
        ) #Save the model periodically by monitoring a quantity e.g. validation loss

        trainer = Trainer(
                callbacks=[
                    TQDMProgressBar(),
                    model_checkpoint_train_loss,
                    *custom_trainer_callbacks,
                ],
                enable_progress_bar=True,
                devices=config.gpus,
                resume_from_checkpoint=None
                if self.args.resume_ckpt == ""
                else self.args.resume_ckpt,
                **config.trainer,
            )
        
        ###############################
        ## Start training or testing ##
        ###############################
        if self.args.train:
            trainer.fit(model, train_loader, ckpt_path=config.ckpt)
        if self.args.test:
            trainer.validate(model, test_loader, ckpt_path=config.ckpt, verbose=True)
