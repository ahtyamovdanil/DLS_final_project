from MyTransformer import Transformer
import torch


class Pruner:
    def __init__(self, model) -> None:
        attn_weights = [
            # encoder attention weights
            [model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)],
            [model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)],
            [model.encoder.layers[i].attn.q_linear.weight for i in range(model.n_enc_layers)],
            # decoder attention weights
            [model.decoder.layers[i].attn_1.q_linear.weight for i in range(model.n_dec_layers)],
            [model.decoder.layers[i].attn_2.q_linear.weight for i in range(model.n_dec_layers)],
        ]
        self.attention_weights = [item for sublist in attn_weights for item in sublist]
        device = self.attention_weights[0].get_device()
        self.attn_bit_masks = [torch.ones(model.n_heads, dtype=bool).to(device) for _ in self.attention_weights]

    def forward(self):
        for mask, weights in zip(self.attention_weights, self.attn_bit_masks):
            weights = weights * mask.view(1, -1, 1)
        
        # TODO: add hard concrete distribution
        loss = torch.sum(torch.stack(self.attn_bit_masks))
        return loss

    
