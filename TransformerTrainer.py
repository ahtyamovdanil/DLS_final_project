import torch
from torch import Tensor, nn
from transformers import get_linear_schedule_with_warmup

from pytorch_lightning.core.lightning import LightningModule
import MyTransformer as mtr


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
