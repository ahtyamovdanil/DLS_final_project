import torch
from torch import nn, Tensor
from torch.distributions.uniform import Uniform
import MyTransformer


class Sampler(nn.Module):
    """ Hard Concrete distribution sampler
    """

    def __init__(
        self,
        n_heads,
        temperature=0.6,
        stretch_limits=(-0.1, 1.1),
        l0_penalty=2.0,
        l2_penalty=2.0,
        eps=1e-6,
    ):
        super().__init__()

        if stretch_limits[0] >= 0 or (stretch_limits[0] > stretch_limits[1]):
            raise ValueError(f"{stretch_limits}")

        self.n_heads = n_heads
        self.temperature = temperature
        self.stretch_limits = nn.Parameter(
            torch.Tensor(stretch_limits), requires_grad=False
        )
        self.l0_penalty = l0_penalty
        self.l2_penalty = l2_penalty
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.empty(1, n_heads, 1), requires_grad=True)
        self.noise = nn.Parameter(
            (Uniform(self.eps, 1 - self.eps).sample_n(self.n_heads).unsqueeze(-1))
        )
        nn.init.uniform(self.alpha, self.eps, 1 - self.eps)

    def forward(self, weights: Tensor):
        return self.get_penalty(weights)

    def get_gates(self, is_train=True):
        left, right = self.stretch_limits
        if self.training and is_train:

            concrete = self.sigmoid(
                (torch.log(self.noise) - torch.log(1 - self.noise) + self.alpha)
                / self.temperature
            )

        else:
            concrete = self.sigmoid(self.alpha)

        stretched_concrete = concrete * (right - left) + left
        concrete = torch.clip(stretched_concrete, 0, 1)

        concrete = (concrete > 0.5).float()
        return concrete

    def get_penalty(self, weights):
        left, right = self.stretch_limits

        p_open = self.sigmoid(self.alpha - self.temperature * torch.log(-left / right))

        p_open = torch.clip(p_open, self.eps, 1.0 - self.eps)
        total_reg = torch.zeros(1, device=self.alpha.device)

        if self.l0_penalty != 0:
            l0_reg = self.l0_penalty * torch.sum(p_open)
            total_reg += torch.mean(l0_reg)

        if self.l2_penalty != 0:
            l2_reg = 0.5 * self.l2_penalty * p_open * torch.sum(weights.detach() ** 2)
            total_reg += torch.mean(l2_reg)

        return total_reg

    def get_probs(self):
        left, right = self.stretch_limits
        concrete = self.sigmoid(self.alpha)
        stretched_concrete = concrete * (right - left) + left
        p_open = torch.clip(stretched_concrete, 0, 1)
        return p_open

    def get_sparsity_rate(self):
        return (self.get_gates(is_train=False) != 0).float().mean().cpu().item()


class Pruner(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        self.n_enc_heads = model.n_enc_heads
        self.n_dec_heads = model.n_dec_heads
        self.n_enc_layers = model.n_enc_layers
        self.n_dec_layers = model.n_dec_layers
        self.n_enc_heads = model.n_enc_heads
        self.model = model

        # freeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        self.enc_samplers = nn.ModuleList(
            [Sampler(self.n_enc_heads) for _ in self.enc_attn_weights()]
        )

    def enc_attn_weights(self):
        for _, module in self.model.encoder.named_modules():
            if isinstance(module, MyTransformer.MultiHeadAttention):
                yield module.fc_out.weight

    def weights_apply(self, func):
        """ applies function to each weight tensor in encoder """
        for _, module in self.model.encoder.named_modules():
            if isinstance(module, MyTransformer.MultiHeadAttention):
                module.fc_out.weight = nn.Parameter(func(module.fc_out.weight))

    def forward(self):
        loss = torch.mean(
            torch.stack(
                [smp(w) for w, smp in zip(self.enc_attn_weights(), self.enc_samplers)]
            )
        )
        return loss

    def get_all_gates(self):
        """ return all gates states '0'-closed, '1'-open """
        return torch.stack(
            [smp.get_gates(is_train=False) for smp in self.enc_samplers]
        ).view(-1, self.n_enc_heads)

    def get_total_sparsity_rate(self):
        return (
            sum([smp.get_sparsity_rate() for smp in self.enc_samplers])
            / self.n_enc_layers
        )

    def get_probs(self):
        return (
            torch.stack([smp.get_probs() for smp in self.enc_samplers])
            .view(-1, self.n_enc_heads)
            .detach()
            .cpu()
            .numpy()
            * self.get_all_gates().detach().cpu().numpy()
        )

    def prune(self):
        with torch.no_grad():
            mask = self.get_all_gates()
            for m, w in zip(mask, self.enc_attn_weights()):
                d_size = w.shape[0]
                f = lambda x: (
                    x.view(d_size, self.n_enc_heads, -1) * m.view(1, -1, 1)
                ).view(d_size, d_size)
                self.weights_apply(f)
