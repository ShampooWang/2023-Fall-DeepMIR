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
    AudioDataset,
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
            model = model_cls.load_from_checkpoint(self.args.resume_ckpt)
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
            train_set = AudioDataset(
                split="train",
                **config.data,
            )
            train_loader = DataLoader(
                train_set,
                batch_size=config.data.train_batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_general,
            )
            
        if self.args.train or self.args.eval:
            dev_set = AudioDataset(
                split="validation",
                **config.data,
            )
            dev_loader = DataLoader(
                dev_set,
                batch_size=config.data.dev_batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_general,
            )
            
        if self.args.test:
            test_set = AudioDataset(
                split="test",
                **config.data,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=config.data.dev_batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_general,
            )
            
        #################################
        ## Set up trainer for training ##
        #################################
        model_checkpoint_val_top1_acc = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_top1_acc:.4f}",
            monitor="val_top1_acc",
            save_top_k=1,
            mode="max",
            every_n_epochs=1,
            save_last=True,
        ) #Save the model periodically by monitoring a quantity e.g. validation loss


        early_stop_callback = EarlyStopping(monitor="val_top1_acc", min_delta=0.00, patience=10, verbose=False, mode="max")

        trainer = Trainer(
                callbacks=[
                    TQDMProgressBar(),
                    model_checkpoint_val_top1_acc,
                    early_stop_callback,
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
            trainer.fit(model, train_loader, dev_loader, ckpt_path=config.ckpt)
        if self.args.eval:
            trainer.validate(model, dev_loader, ckpt_path=config.ckpt, verbose=True)
        if self.args.test:
            trainer.validate(model, test_loader, ckpt_path=config.ckpt, verbose=True)
