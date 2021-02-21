import torchtext
from torchtext.data import Field, BucketIterator
from pytorch_lightning import LightningDataModule
from nltk.tokenize import WordPunctTokenizer


class BaseDataModule(LightningDataModule):
    def __init__(self, batch_size, device, data_path, seed, max_len=None):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.data_path = data_path
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
