import torch
from torchtext.data.metrics import bleu_score
from typing import List, Union
from MyTransformer import Transformer
from torchtext.data import Field
from tqdm.notebook import tqdm


def translate_sentence(
    sentence: List[str],
    model: Transformer,
    src_field: Field,
    trg_field: Field,
    max_len: int,
    device: Union[torch.device, str],
):

    src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
    trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
    trg_eos_idx = src_field.vocab.stoi[src_field.eos_token]
    trg_unk_idx = src_field.vocab.stoi[src_field.unk_token]
    trg_init_idx = src_field.vocab.stoi[trg_field.init_token]

    src_tokens = [src_field.init_token] + sentence + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in src_tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor, src_pad_idx)
    with torch.no_grad():
        e_outputs = model.encoder(src_tensor, src_mask)
    outputs = torch.zeros(max_len).type_as(src_tensor)
    outputs[0] = torch.LongTensor([trg_init_idx])

    for i in range(1, max_len):
        trg_mask = model.make_trg_mask(outputs[:i].unsqueeze(0), trg_pad_idx)
        with torch.no_grad():
            out = model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask)
            probs = model.out(out)
        outputs[i] = probs.argmax(2)[:, -1].item()
        if outputs[i] == trg_eos_idx:
            break

    return [trg_field.vocab.itos[i] for i in outputs if i != trg_unk_idx][1:-1]


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

    trgs = []
    pred_trgs = []

    for datum in tqdm(data):

        src = vars(datum)["src"]
        trg = vars(datum)["trg"]

        pred_trg = translate_sentence(
            sentence=src,
            model=model,
            src_field=src_field,
            trg_field=trg_field,
            device=device,
            max_len=max_len,
        )

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)
