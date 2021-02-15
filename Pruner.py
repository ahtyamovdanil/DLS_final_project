from typing import Union
from MyTransformer import Transformer
import torch
from torch import nn
from torch import Tensor
from torch.distributions.uniform import Uniform


class Sampler(nn.Module):
    """ Hard Concrete distribution sampler
    """
    def __init__(self,n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.empty(1, n_heads, 1), requires_grad=True)
        self.lmbd = 1.2

        nn.init.uniform(self.alpha, 0.2, 0.3)
    
    def forward(self, mask: Tensor, weights: Tensor):
        eta = 1e-10
        L = torch.log(mask + eta) - torch.log(1 - mask + eta)
        mask = self.sigmoid(L + torch.log(self.alpha + eta) / (self.lmbd + eta))
        d_size = weights.shape[0]
        #weights = mask.view(self.n_layers, self.n_heads, 1, 1) * weights.view(self.n_layers, self.n_heads, -1, emb_size)
        with torch.no_grad():
            weights = weights.view(d_size, self.n_heads, -1)
            weights *= mask.view(1, -1, 1)

        loss = mask.mean()
        return loss


class Pruner(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        # encoder attention weights
        self.enc_attn_weights = [ 
            *[model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)],
            *[model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)],
            *[model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)]
            ]

        # decoder attention weights
        self.dec_attn_weights = [
            *[model.decoder.layers[i].attn_1.q_linear.weight for i in range(model.n_dec_layers)],
            *[model.decoder.layers[i].attn_2.q_linear.weight for i in range(model.n_dec_layers)],
        ]

        m = Uniform(torch.tensor([0.97]), torch.tensor([0.99]))

        total_n_enc_layers = len(self.enc_attn_weights)
        total_n_dec_layers = len(self.dec_attn_weights)

        self.enc_attn_masks = nn.ParameterList([
            nn.Parameter(m.sample_n(model.n_enc_heads), requires_grad=False)
                for _ in self.enc_attn_weights
        ])

        self.dec_attn_masks = nn.ParameterList([
            nn.Parameter(m.sample_n(model.n_dec_heads), requires_grad=False)
                for _ in self.dec_attn_weights
        ])

        self.enc_sampler = Sampler(model.n_enc_heads)
        self.dec_sampler = Sampler(model.n_dec_heads)

    def apply_mask(self, masks, weights):
        
        pass

    def forward(self):

        # print(self.enc_sampler.get_device())
        # print(self.dec_sampler.get_device())

        enc_loss = 0
        for m, w in zip(self.enc_attn_masks, self.enc_attn_weights):
            enc_loss = enc_loss + self.enc_sampler(m, w)
        enc_loss = enc_loss / len(self.enc_attn_masks)

        dec_loss = 0
        for m, w in zip(self.dec_attn_masks, self.dec_attn_weights):
            dec_loss = self.enc_sampler(m, w)
        dec_loss = dec_loss / len(self.enc_attn_masks)

        # enc_loss = self.enc_sampler(self.enc_attn_masks, torch.stack(self.enc_attn_weights))
        # dec_loss = self.dec_sampler(self.dec_attn_masks, torch.stack(self.dec_attn_weights))

        loss = enc_loss + dec_loss

        return loss

    def __call__(self):
        return self.forward()
    
    
