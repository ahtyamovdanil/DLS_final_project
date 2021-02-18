import torch
from torch import mean, nn
from torch import Tensor
from torch.distributions.uniform import Uniform


class Sampler(nn.Module):
    """ Hard Concrete distribution sampler
    """
    def __init__(self, n_heads, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=2.0, l2_penalty=2.0, eps=1e-6, hard=True):
        super().__init__()

        if stretch_limits[0] >= 0 or (stretch_limits[0] > stretch_limits[1]):
            raise ValueError(f"{stretch_limits}")

        self.n_heads = n_heads
        self.temperature = temperature
        self.stretch_limits = nn.Parameter(torch.Tensor(stretch_limits), requires_grad=False)
        self.l0_penalty = l0_penalty
        self.l2_penalty = l2_penalty
        self.eps = eps
        self.hard = hard
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(
            torch.empty(1, n_heads, 1),
            requires_grad=True
        )
        nn.init.uniform(self.alpha, -0.1, 0.1)

    def forward(self, weights: Tensor):

        mask = self.get_gates()
        d_size = weights.shape[0]

        loss = self.get_penalty(weights)

        # with torch.no_grad():
        #     new_weights = weights.view(d_size, self.n_heads, -1) * mask.view(1, -1, 1)
        #     weights.copy_(new_weights.view(d_size, d_size))

        return loss

    def get_gates(self):
        left, right = self.stretch_limits
        if self.training:
            noise = Uniform(self.eps, 1-self.eps) \
                .sample_n(self.n_heads) \
                .unsqueeze(-1) \
                .to(self.alpha.device)
            concrete = self.sigmoid(torch.log(noise) - torch.log(1 - noise) + self.alpha) / self.temperature
        else:
            concrete = self.sigmoid(self.alpha)

        stretched_concrete = concrete * (right - left) + left
        concrete = torch.clip(stretched_concrete, 0, 1)

        if self.hard:
            concrete = (concrete > 0.5).float()
            # clipped_concrete = clipped_concrete + (
            #     hard_concrete - clipped_concrete).detach()

        return concrete

    def get_penalty(self, weights):
        left, right = self.stretch_limits

        p_open = self.sigmoid(
            self.alpha - self.temperature * torch.log(-left / right)
        )

        p_open = torch.clip(p_open, self.eps, 1.0 - self.eps)
        total_reg = torch.zeros(1)

        if self.l0_penalty != 0:
            l0_reg = self.l0_penalty * torch.sum(p_open)
            total_reg = torch.mean(l0_reg)

        if self.l2_penalty != 0:
            l2_reg = 0.5 * self.l2_penalty * p_open * torch.mean(weights ** 2)
            total_reg = total_reg + torch.mean(l2_reg)

        return total_reg
 
    def get_probs(self):
        left, right = self.stretch_limits
        p_open = self.sigmoid(
            self.alpha - self.temperature * torch.log(-left / right)
        )
        return torch.clip(p_open, self.eps, 1.0 - self.eps)

    def get_sparsity_rate(self):
        return (self.get_gates() != 0).float().mean().cpu().item()


class Pruner(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()

        # encoder attention weights
        self.n_enc_heads = model.n_enc_heads
        self.n_dec_heads = model.n_dec_heads
        self.n_enc_layers = model.n_enc_layers
        self.n_dec_layers = model.n_dec_layers
        self.n_enc_heads = model.n_enc_heads

        self.enc_attn_weights = [
            model.encoder.layers[i].attn.fc_out.weight
            for i in range(model.n_enc_layers)
        ]

        # decoder attention weights
        self.dec_attn_weights = [
            *[model.decoder.layers[i].attn_1.fc_out.weight
                for i in range(model.n_dec_layers)],
            *[model.decoder.layers[i].attn_2.fc_out.weight
                for i in range(model.n_dec_layers)],
            ]

        self.enc_samplers = nn.ModuleList(
            [Sampler(self.n_enc_heads) for _ in self.enc_attn_weights]
            )

        self.dec_samplers = nn.ModuleList(
            [Sampler(self.n_dec_heads) for _ in self.dec_attn_weights]
        )

    def forward(self):
        enc_loss = torch.mean(torch.stack([smp(w) for w, smp in zip(self.enc_attn_weights, self.enc_samplers)]))
        # dec_loss = torch.mean(torch.stack([smp(w) for w, smp in zip(self.dec_attn_weights, self.dec_samplers)]))
        loss = enc_loss

        return loss

    def get_all_gates(self):
        return {
            "enc_gates": torch.stack([smp.get_gates() for smp in self.enc_samplers]).view(-1, self.n_enc_heads),
            "dec_gates": torch.stack([smp.get_gates() for smp in self.dec_samplers]).view(-1, self.n_dec_heads)
        }
    
    def get_total_sparsity_rate(self):
        return sum([smp.get_sparsity_rate() for smp in self.enc_samplers])/self.n_enc_layers

    def get_probs(self):
        return torch.stack([smp.get_probs() for smp in self.enc_samplers]).view(-1, self.n_enc_heads)

    def prune(self):
        with torch.no_grad():
            mask = self.get_all_gates()["enc_gates"]
            d_size = self.enc_attn_weights[0].shape[0]
            for m, w in zip(mask, self.enc_attn_weights):
                w_new = w.view(d_size, 8, -1) * m.view(1, -1, 1)
                w.copy_(w_new.view(d_size, d_size))   

