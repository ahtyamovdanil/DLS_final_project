import torch
from torch import Tensor, nn
from pytorch_lightning.core.lightning import LightningModule
from Pruner import Pruner


class MyPruningTrainer(LightningModule):
    def __init__(
        self,
        model,
        src_pad_idx: int,
        trg_pad_idx: int,
        lr: float = 1e-5,
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.lr = lr
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.model = model
        self.pruner = Pruner(self.model)

    def forward(
        self, src_batch: Tensor, trg_batch: Tensor, src_mask: Tensor, trg_mask: Tensor
    ) -> Tensor:
        return self.model(src_batch, trg_batch, src_mask, trg_mask)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        trg_input = trg[:, :-1]
        targets = trg[:, 1:].contiguous().view(-1)

        src_mask = self.model.make_src_mask(src, self.src_pad_idx)
        trg_mask = self.model.make_trg_mask(trg_input, self.trg_pad_idx)

        pruning_loss = self.pruner()
        self.pruner.prune()

        if batch_idx % 10 == 0:
            with open("./data/probs.txt", "a") as file:
                file.writelines(
                    " ".join(str(item) for item in self.pruner.get_probs()) + "\n"
                )

        preds = self(src, trg_input, src_mask, trg_mask)

        model_loss = self.criterion(preds.view(-1, preds.size(-1)), targets)
        total_loss = model_loss + 2 * pruning_loss

        tensorboard_logs = {
            "total_train_loss": total_loss,
            "model_loss": model_loss,
            "pruning_loss": pruning_loss,
        }

        return {"loss": total_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        trg_input = trg[:, :-1]
        targets = trg[:, 1:].contiguous().view(-1)

        src_mask = self.model.make_src_mask(src, self.src_pad_idx)
        trg_mask = self.model.make_trg_mask(trg_input, self.trg_pad_idx)

        preds = self(src, trg_input, src_mask, trg_mask)

        model_loss = self.criterion(preds.view(-1, preds.size(-1)), targets)
        pruning_loss = self.pruner.get_total_sparsity_rate()
        total_loss = model_loss + 2 * pruning_loss

        return {
            "model_val_loss": model_loss,
            "pruning_val_sparsity": pruning_loss,
            "total_val_loss": total_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_model_loss = torch.stack([x["model_val_loss"] for x in outputs]).mean()
        avg_pruning_loss = self.pruner.get_total_sparsity_rate()
        avg_total_loss = torch.stack([x["total_val_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "total_val_loss": avg_total_loss,
            "model_val_loss": avg_model_loss,
            "pruning_val_sparsity": avg_pruning_loss,
        }
        return {"model_loss": avg_model_loss, "log": tensorboard_logs}

    def prepare_data(self):
        pass
