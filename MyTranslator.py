import torch
from torch import nn
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from nltk.tokenize import WordPunctTokenizer

from MyTransformer import Encoder, Decoder, Seq2Seq
from typing import List


class BaseDataModule(LightningDataModule):
    def __init__(self, batch_size, device, data_path, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.data_path = data_path
        self.seed = seed

        self.tokenizer = WordPunctTokenizer()

        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

        tokenize = lambda x: self.tokenizer.tokenize(x.lower())

        self.src_field = Field(
            tokenize=tokenize,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
            batch_first=True,
        )
        self.trg_field = Field(
            tokenize=tokenize,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True,
            batch_first=True,
        )
        self.trg_text_field = Field(
            init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True,
        )

    def prepare_data(self):

        dataset = torchtext.data.TabularDataset(
            path=self.data_path,
            format="tsv",
            fields=[("trg", self.trg_field), ("src", self.src_field),],
        )
        train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
        train_data = train_data
        valid_data = valid_data
        test_data = test_data
        self.src_field.build_vocab(train_data, min_freq=3)
        self.trg_field.build_vocab(train_data, min_freq=3)

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
        n_layers,
        n_heads,
        max_length,
        enc_hid_dim,
        enc_pf_dim,
        enc_dropout,
        dec_hid_dim,
        dec_pf_dim,
        dec_dropout,
        src_field,
        trg_field,
        lr,
        device,
    ):

        super().__init__()

        enc = Encoder(
            input_dim=len(src_field.vocab),
            hid_dim=enc_hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=enc_pf_dim,
            dropout=enc_dropout,
            device=device,
            max_length=max_length,
        )
        dec = Decoder(
            output_dim=len(trg_field.vocab),
            hid_dim=dec_hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=dec_pf_dim,
            dropout=dec_dropout,
            device=device,
            max_length=max_length,
        )
        self.src_field = src_field
        self.trg_field = trg_field
        self.lr = lr
        src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
        trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.seq2seq = Seq2Seq(
            encoder=enc,
            decoder=dec,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            device=device,
        ).to(device)

        def init_weights(model):
            for _, param in model.named_parameters():
                nn.init.uniform_(param, -0.08, 0.08)

        self.seq2seq.apply(init_weights)

    def forward(self, src_batch, trg_batch):
        return self.seq2seq(src_batch, trg_batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        source = batch.src
        target = batch.trg
        output, _ = self(source, target[:, :-1])

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target[:, 1:].contiguous().view(-1)

        loss = self.criterion(output, target)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        source = batch.src
        target = batch.trg

        output, _ = self(source, target[:, :-1])
        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target[:, 1:].contiguous().view(-1)

        # output = [(trg sent len - 1) * batch size, output dim]
        loss = self.criterion(output, target)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"valid_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def translate(self, sentence: List[str], max_len=15) -> List[List[str]]:
        """translate sentences of given dataset

        Args:
            sentence (List[str]): tokenized sentence to translate
            max_len (int, optional): max number of words in sentence. Defaults to 15.

        Returns:
            List[List[str]]: List of sentences (Lists) with translated words (str)
        """

        src_tokens = [self.src_field.init_token] + sentence + [self.src_field.eos_token]
        src_indexes = [self.src_field.vocab.stoi[token] for token in src_tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)

        src_mask = self.seq2seq.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = self.seq2seq.encoder(src_tensor, src_mask)

        trg_indexes = [self.trg_field.vocab.stoi[self.trg_field.init_token]]

        for _ in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
            # trg_tensor = trg_indexes.unsqueeze(0)
            trg_mask = self.seq2seq.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, _ = self.seq2seq.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask
                )

            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.trg_field.vocab.stoi[self.trg_field.eos_token]:
                break

        trg_tokens = [self.trg_field.vocab.itos[i] for i in trg_indexes][1:-1]
        trg_tokens = [item for item in trg_tokens if item != self.trg_field.unk_token]
        # print(trg_tokens)

        return trg_tokens

    def test_step(self, batch, batch_idx):

        source = vars(batch)["src"]
        target = vars(batch)["trg"]

        preds = self.translate(source, max_len=50)

        # loss = self.criterion(out, target.long())
        # conf_matrix = self.get_confusion_matrix(out, target)
        # f1_score2 = f1_skl(pred.view(-1).cpu().numpy(), target.view(-1).cpu().numpy())
        return {"preds": preds, "target": target}
        # return{'test_loss': loss, 'pred': out.argmax(1).view(-1), 'target': target}

    def test_epoch_end(self, outputs):
        preds = [x["preds"] for x in outputs]
        target = [[x["target"]] for x in outputs]
        # preds = [sent for x in outputs for sent in x["preds"]]
        # target = [sent for x in outputs for sent in x["target"]]
        score = bleu_score(preds, target)
        # conf_matrix = torch.stack([x['test_conf_matrix'] for x in outputs]).sum(0)
        # f1_score = self.get_f1_score(conf_matrix)
        # tp, tn, fn, fp = conf_matrix.view(-1)
        return {"bleu_test": f"{score*100:.2f}"}

    def prepare_data(self):
        pass
