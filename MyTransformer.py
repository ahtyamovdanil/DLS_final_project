from torch import Tensor
import torch.nn.functional as F
from torch import nn
import torch
from typing import Union, List, Tuple
import math
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(d_model):
                if i % 2:
                    pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                else:
                    pe[pos, i] = math.cos(pos / (10000**((2 * i) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class PosAttentionLayer(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                d_k: int,
                mask: Tensor = None) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"`hid_dim` must be divisible by `n_heads`, but found d_model={d_model}, n_heads={n_heads}"
            )
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attention = PosAttentionLayer(dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # transpose to get dimensions bs * n_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc_out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = (self.alpha * (x - x.mean(dim=-1, keepdim=True)) /
                (x.std(dim=-1, keepdim=True) + self.eps) + self.bias)
        return norm


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = Norm(d_model)

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = Norm(d_model)

    def forward(self, trg: Tensor, e_outputs: Tensor, src_mask: Tensor,
                trg_mask: Tensor) -> Tensor:
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
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
            dec_dropout: float
        # device: Union[torch.device, str]
    ) -> None:
        super().__init__()
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_enc_heads = n_enc_heads
        self.n_dec_heads = n_dec_heads
        self.encoder = Encoder(src_vocab_size, d_model, n_enc_layers, n_enc_heads,
                               enc_dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, n_dec_layers, n_dec_heads,
                               dec_dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)
        # self.device = device

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                trg_mask: Tensor) -> Tensor:
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

    def make_src_mask(self, src, src_pad_idx):
        # src = [batch size, src len]
        src_mask = (src != src_pad_idx).unsqueeze(-2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg: Tensor, trg_pad_idx):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != trg_pad_idx).unsqueeze(-2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.size(1)

        trg_sub_mask = torch.tril(
            torch.ones((1, trg_len, trg_len), device=trg.get_device())).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask
