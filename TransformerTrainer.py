import torch
from torch import Tensor, nn
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from nltk.tokenize import WordPunctTokenizer

import MyTransformer as mtr
from typing import List, Union
import random


class BaseDataModule(LightningDataModule):
    def __init__(self, batch_size, device, data_path, seed, max_len=None):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.data_path = data_path
        # self.seed = seed

        self.tokenizer = WordPunctTokenizer()

        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

        # seed = torch.get_rng_state()
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.set_deterministic(True)

        tokenize = lambda x: self.tokenizer.tokenize(x.lower())

        self.src_field = Field(
            tokenize=tokenize,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
            # fix_length=max_len,
            batch_first=True,
        )
        self.trg_field = Field(
            tokenize=tokenize,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
            # fix_length=max_len,
            batch_first=True,
        )

        self.src_pad_idx = None
        self.trg_pad_idx = None

        self.src_eos_idx = None
        self.trg_eos_idx = None

        self.src_bos_idx = None
        self.trg_bos_idx = None

        self.src_vocab_len = None
        self.trg_vocab_len = None

    def prepare_data(self):

        dataset = torchtext.data.TabularDataset(
            path=self.data_path,
            format="tsv",
            fields=[("src", self.src_field), ("trg", self.trg_field)],
        )
        train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
        train_data = train_data
        valid_data = valid_data
        test_data = test_data
        self.src_field.build_vocab(train_data, min_freq=3)
        self.trg_field.build_vocab(train_data, min_freq=3)

        self.src_pad_idx = self.src_field.vocab.stoi[self.src_field.pad_token]
        self.trg_pad_idx = self.trg_field.vocab.stoi[self.trg_field.pad_token]

        self.src_eos_idx = self.src_field.vocab.stoi[self.src_field.eos_token]
        self.trg_eos_idx = self.trg_field.vocab.stoi[self.trg_field.eos_token]

        self.src_bos_idx = self.src_field.vocab.stoi[self.src_field.init_token]
        self.trg_bos_idx = self.trg_field.vocab.stoi[self.trg_field.init_token]

        self.src_vocab_len = len(self.src_field.vocab)
        self.trg_vocab_len = len(self.trg_field.vocab)

        self.train_iter, self.valid_iter, _ = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            device=self.device,
            sort_key=lambda x: len(x.src),
        )
        self.test_iter = test_data

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.valid_iter

    def test_dataloader(self):
        return self.test_iter


class MyTranslator(LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int,
        n_enc_layers: int,
        n_dec_layers: int,
        n_enc_heads: int,
        n_dec_heads: int,
        enc_dropout: float,
        dec_dropout: float,
        src_pad_idx: int,
        trg_pad_idx: int,
        lr: float = 1e-4,
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.lr = lr
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.model = mtr.Transformer(
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            d_model=d_model,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            n_enc_heads=n_enc_heads,
            n_dec_heads=n_dec_heads,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
        )

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src_batch: Tensor, trg_batch: Tensor, src_mask: Tensor, trg_mask: Tensor
    ) -> Tensor:
        return self.model(src_batch, trg_batch, src_mask, trg_mask)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=100
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        trg_input = trg[:, :-1]
        targets = trg[:, 1:].contiguous().view(-1)

        # src_mask, trg_mask = mtr.create_masks(
        #     src, trg_input, self.src_pad_idx, self.trg_pad_idx
        # )

        src_mask = self.model.make_src_mask(src, self.src_pad_idx)
        trg_mask = self.model.make_trg_mask(trg_input, self.trg_pad_idx)

        preds = self(src, trg_input, src_mask, trg_mask)

        loss = self.criterion(preds.view(-1, preds.size(-1)), targets)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        trg_input = trg[:, :-1]
        targets = trg[:, 1:].contiguous().view(-1)

        src_mask = self.model.make_src_mask(src, self.src_pad_idx)
        trg_mask = self.model.make_trg_mask(trg_input, self.trg_pad_idx)

        preds = self(src, trg_input, src_mask, trg_mask)

        loss = self.criterion(preds.view(-1, preds.size(-1)), targets)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"valid_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def prepare_data(self):
        pass
