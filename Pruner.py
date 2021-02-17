import torch
from torch import nn
from torch import Tensor
from torch.distributions.uniform import Uniform


class Sampler(nn.Module):
    """ Hard Concrete distribution sampler
    """
    def __init__(self, n_heads, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=0.0, l2_penalty=0.0, eps=1e-6, hard=False):
        super().__init__()

        if stretch_limits[0] >= 0 or (stretch_limits[0] > stretch_limits[1]):
            raise ValueError(f"{stretch_limits}")

        self.n_heads = n_heads
        self.temperature = temperature
        self.stretch_limits = stretch_limits
        self.l0_penalty = l0_penalty
        self.l2_penalty = l2_penalty
        self.eps = eps
        self.hard = hard
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(
            torch.empty(1, n_heads, 1),
            requires_grad=True
        )
        nn.init.uniform(self.alpha, 0.2, 0.3)

    def forward(self, weights: Tensor):

        mask = self.get_gates()
        d_size = weights.shape[0]

        loss = self.get_penalty(weights)

        with torch.no_grad():
            weights = weights.view(d_size, self.n_heads, -1)
            weights *= mask.view(1, -1, 1)

        return loss

    def get_gates(self):
        left, right = self.stretch_limits
        if self.training:
            noise = Uniform(self.eps, 1-self.eps).sample(self.n_heads)
            concrete = self.sigmoid(
                torch.log(noise)
                - torch.log(1 - noise)
                + self.alpha) / self.temperature
        else:
            concrete = self.sigmoid(self.alpha)

        stretched_concrete = concrete * (left - right) + left
        clipped_concrete = torch.clip(stretched_concrete, 0, 1)

        if self.hard:
            hard_concrete = (clipped_concrete > 0.5).float()
            clipped_concrete = clipped_concrete + (
                hard_concrete - clipped_concrete).detach()

        return clipped_concrete

    def get_penalty(self, weights):
        left, right = self.stretch_limits

        p_open = self.sigmoid(
            self.alpha - self.temperature * torch.log(-left / right)
        )

        p_open = torch.clip(p_open, self.eps, 1.0 - self.eps)
        total_reg = 0.0

        if self.l0_penalty != 0:
            l0_reg = self.l0_penalty * torch.sum(p_open)
            total_reg = torch.mean(l0_reg)

        if self.l2_penalty != 0:
            l2_reg = 0.5 * self.l2_penalty * p_open * torch.mean(weights ** 2)
            total_reg += torch.mean(l2_reg)

        return total_reg

    def get_sparsity_rate(self):
        return (self.get_gates() != 0).mean()


class Pruner(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        # encoder attention weights
        self.enc_attn_weights = [
            *[model.encoder.layers[i].attn.q_linear.weight
                for i in range(model.n_enc_layers)],
            *[model.encoder.layers[i].attn.q_linear.weight
                for i in range(model.n_enc_layers)],
            *[model.encoder.layers[i].attn.q_linear.weight
                for i in range(model.n_enc_layers)]
            ]

        # decoder attention weights
        self.dec_attn_weights = [
            *[model.decoder.layers[i].attn_1.q_linear.weight
                for i in range(model.n_dec_layers)],
            *[model.decoder.layers[i].attn_2.q_linear.weight
                for i in range(model.n_dec_layers)],
        ]

        m = Uniform(torch.tensor([0.97]), torch.tensor([0.99]))

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

        loss = enc_loss + dec_loss

        return loss

    def __call__(self):
        return self.forward()
