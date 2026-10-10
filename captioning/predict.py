"""Captioning model construction and beam-search inference on precomputed grid features."""
import itertools
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from .data import TextField
from .models import MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, Transformer


class _VocabUnpickler(pickle.Unpickler):
    """Resolve the original top-level `data.vocab` module path to `captioning.data.vocab`."""

    def find_class(self, module, name):
        if module == 'data' or module.startswith('data.'):
            module = 'captioning.' + module
        return super().find_class(module, name)


def load_vocab(path):
    """Load a pickled Vocab saved either by the original RSTNet code or by this package."""
    with open(path, 'rb') as f:
        return _VocabUnpickler(f).load()


def build_text_field(tokenize='spacy'):
    """Text pipeline used in training; inference only decodes, so any tokenizer works there."""
    return TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize=tokenize,
                     remove_punctuation=True, nopoints=False)


def build_captioning_model(vocab, cfg):
    """Meshed-Memory Transformer with memory-augmented encoder and meshed decoder."""
    encoder = MemoryAugmentedEncoder(cfg.n_enc_layers, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': cfg.memory_slots})
    decoder = MeshedDecoder(len(vocab), cfg.max_decoder_len, cfg.n_dec_layers, vocab.stoi['<pad>'])
    return Transformer(vocab.stoi['<bos>'], encoder, decoder)


def load_captioning_model(cfg, device, checkpoint=None):
    """Return (model, text_field) with vocab loaded and optional trained weights."""
    text_field = build_text_field(tokenize=str.split)
    text_field.vocab = load_vocab(cfg.vocab_path)
    model = build_captioning_model(text_field.vocab, cfg)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'])
    return model.to(device).eval(), text_field


def clean_caption(tokens):
    """Collapse repeated consecutive tokens and replace underscores with spaces."""
    return ' '.join(k for k, _ in itertools.groupby(tokens)).strip().replace('_', ' ')


@torch.no_grad()
def generate_captions(model, text_field, features, images, device, max_len=23, beam_size=3):
    """Return {image name without extension: caption}; `features` maps '<id>_grids' to (49, 2048) arrays."""
    # The submitted run looked up '</s>', which is not in the vocab and resolves to the <unk> index;
    # decoding then stops at '<eos>' inside text_field.decode. Kept as-is to reproduce the submission.
    eos_idx = text_field.vocab.stoi['</s>']
    model.eval()
    captions = {}
    for image in tqdm(images, desc='Captioning inference'):
        grid = features['%d_grids' % image['id']][()]
        visual = torch.from_numpy(np.array([grid])).float().to(device)
        out, _ = model.beam_search(visual, max_len, eos_idx, beam_size, out_size=1)
        tokens = text_field.decode(out, join_words=False)[0]
        name = os.path.basename(image['file_name']).split('.')[0]
        captions[name] = clean_caption(tokens)
    return captions
