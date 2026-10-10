"""Smoke test on a tiny random DACON-style dataset: IQA train + inference, captioning train step + decoding,
and the end-to-end submission writer. Pretrained backbones, detector features and trained checkpoints are
replaced by random weights and random numpy grid features of the real shape (49 x 2048).

Usage: python tests/test_smoke.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile

import cv2
import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config, ic_config, iqa_config  # noqa: E402
from captioning.data import COCO, DataLoader, ImageDetectionsField  # noqa: E402
from captioning.predict import build_captioning_model, build_text_field, load_vocab  # noqa: E402
from captioning.prepare_data import write_annotations  # noqa: E402
from captioning.train import xe_step  # noqa: E402
from inference import SUBMISSION_COLUMNS, read_test_names, run_captioning, write_submission  # noqa: E402
from iqa.predict import load_iqa_model, predict_mos  # noqa: E402
from iqa.train import train as train_iqa  # noqa: E402

DEVICE = os.environ.get('SMOKE_DEVICE', 'cpu')  # CPU by default so the test also runs on a busy or absent GPU
GRID_SHAPE = (49, 2048)
# The real ViT-R50 backbone (~150M params) can be forced with SMOKE_FULL_BACKBONE=1; by default a smaller hybrid ViT
# with the same 12-block layout keeps memory low.
SMALL_IQA = {} if os.environ.get('SMOKE_FULL_BACKBONE') else dict(
    backbone='vit_small_r26_s32_384', embed_dim=384, dim_mlp=384, patch_size=32)
WORDS = 'nice colors great light love the composition sharp focus too dark image'.split()


def write_csv(path, header, rows):
    """Write rows (lists) under a header."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_fake_dataset(root, n_train=6, n_valid=2, n_test=3, rng=np.random.default_rng(0)):
    """Random RGB images plus train/test/mos CSVs with the DACON column layout."""
    for split in ('train', 'test'):
        os.makedirs(os.path.join(root, split))
    train_rows, test_rows = [], []
    for i in range(n_train + n_valid):
        name = 'tr%06d' % i
        cv2.imwrite(os.path.join(root, 'train', name + '.jpg'), rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        caption = ' '.join(rng.choice(WORDS, size=5))
        train_rows.append([name, './dataset/train/%s.jpg' % name, float(rng.uniform(2, 9)), caption])
    for i in range(n_test):
        name = '%09d' % i if i == 0 else 'te%06d' % i  # first id lost its leading zero, as in the DACON CSV
        file_name = name.rjust(10, '0')
        cv2.imwrite(os.path.join(root, 'test', file_name + '.jpg'), rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        test_rows.append([name, './dataset/test/%s.jpg' % file_name])

    write_csv(os.path.join(root, 'train.csv'), ['img_name', 'img_path', 'mos', 'comments'], train_rows)
    write_csv(os.path.join(root, 'train_mos.csv'), ['img_path', 'mos'], [r[1:3] for r in train_rows[:n_train]])
    write_csv(os.path.join(root, 'valid_mos.csv'), ['img_path', 'mos'], [r[1:3] for r in train_rows[n_train:]])
    write_csv(os.path.join(root, 'test.csv'), ['img_name', 'img_path'], test_rows)


def write_fake_features(json_paths, out_path, rng=np.random.default_rng(1)):
    """Random grid features for every image id in the given annotation files."""
    with h5py.File(out_path, 'w') as f:
        for p in json_paths:
            for img in json.load(open(p))['images']:
                f.create_dataset('%d_grids' % img['id'], data=rng.standard_normal(GRID_SHAPE).astype(np.float32))


def check_iqa(root):
    """Train one epoch from random init, then predict MOS for the test images."""
    cfg = Config(iqa_config, **SMALL_IQA, batch_size=2, num_workers=0, n_epoch=1,
                 ckpt_dir=os.path.join(root, 'ckpt_iqa'), log_dir=os.path.join(root, 'logs_iqa'))
    train_iqa(cfg, train_csv=os.path.join(root, 'train_mos.csv'), valid_csv=os.path.join(root, 'valid_mos.csv'),
              img_dir=os.path.join(root, 'train'), pretrained=False, device=DEVICE)
    net = load_iqa_model(cfg, DEVICE, checkpoint=None, pretrained=False)
    mos = predict_mos(net, os.path.join(root, 'test.csv'), os.path.join(root, 'test'), DEVICE)
    values = np.array(list(mos.values()), dtype=np.float64)
    assert values.shape == (3,), values.shape
    assert np.isfinite(values).all(), values
    print('[iqa] train epoch + inference OK, MOS =', np.round(values, 4).tolist())
    return net, mos


def check_captioning(root):
    """Annotation/feature pipeline, one XE optimisation step, greedy and beam decoding on a tiny vocab."""
    write_annotations(os.path.join(root, 'train.csv'), os.path.join(root, 'valid_mos.csv'),
                      os.path.join(root, 'test.csv'), os.path.join(root, 'train.json'),
                      os.path.join(root, 'valid.json'), os.path.join(root, 'test.json'))
    write_fake_features([os.path.join(root, 'train.json'), os.path.join(root, 'valid.json')],
                        os.path.join(root, 'train.hdf5'))
    write_fake_features([os.path.join(root, 'test.json')], os.path.join(root, 'test.hdf5'))

    image_field = ImageDetectionsField(train_detections_path=os.path.join(root, 'train.hdf5'),
                                       valid_detections_path=os.path.join(root, 'train.hdf5'),
                                       max_detections=ic_config.max_detections, load_in_tmp=False,
                                       train_json_path=os.path.join(root, 'train.json'),
                                       valid_json_path=os.path.join(root, 'valid.json'), features=True)
    text_field = build_text_field(tokenize=str.split)
    train_set, val_set, _ = COCO(image_field, text_field, '', root, root).splits
    text_field.build_vocab(train_set, val_set, min_freq=1)

    cfg = Config(ic_config)
    model = build_captioning_model(text_field.vocab, cfg).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    detections, captions = next(iter(DataLoader(train_set, batch_size=3, shuffle=False)))
    assert detections.shape == (3,) + GRID_SHAPE, detections.shape
    model.train()
    loss = xe_step(model, detections.to(DEVICE), captions.to(DEVICE), loss_fn, len(text_field.vocab))
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss), loss

    model.eval()
    visual = torch.from_numpy(np.random.default_rng(2).standard_normal((2,) + GRID_SHAPE).astype(np.float32)).to(DEVICE)
    eos = text_field.vocab.stoi['<eos>']
    with torch.no_grad():
        greedy, _ = model.test(visual, 6, eos)
        beam, _ = model.beam_search(visual, 6, eos, 3, out_size=1)
    assert greedy.shape == (2, 6) and beam.shape == (2, 6), (greedy.shape, beam.shape)
    print('[captioning] vocab=%d, XE loss=%.3f, greedy/beam tokens %s OK'
          % (len(text_field.vocab), loss.item(), tuple(beam.shape)))

    vocab = load_vocab(ic_config.vocab_path)
    assert len(vocab) > 100 and '<eos>' in vocab.stoi
    print('[captioning] released vocab.pkl loads (%d tokens)' % len(vocab))


def check_submission(root, iqa_net):
    """End-to-end writer: random-init captioner on fake test features + IQA predictions -> submission CSV."""
    captions = run_captioning(DEVICE, None, os.path.join(root, 'test.hdf5'), os.path.join(root, 'test.json'),
                              os.path.join(root, 'outputs', 'captions.json'))
    mos = predict_mos(iqa_net, os.path.join(root, 'test.csv'), os.path.join(root, 'test'), DEVICE)
    names = read_test_names(os.path.join(root, 'test.csv'))
    out = os.path.join(root, 'outputs', 'submission.csv')
    write_submission(out, names, mos, captions)

    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == SUBMISSION_COLUMNS, rows[0]
    assert [r[0] for r in rows[1:]] == names and all(len(r) == 3 for r in rows[1:]), rows
    assert all(np.isfinite(float(r[1])) for r in rows[1:])
    print('[submission] %d rows with columns %s OK; first row: %s' % (len(rows) - 1, rows[0], rows[1]))


def main():
    """Run all checks in a temporary directory next to the repository."""
    tmp_parent = os.environ.get('TMP') or os.path.dirname(ROOT)
    root = tempfile.mkdtemp(prefix='samsung_smoke_', dir=tmp_parent)
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        make_fake_dataset(root)
        iqa_net, _ = check_iqa(root)
        check_captioning(root)
        check_submission(root, iqa_net)
        print('SMOKE TEST PASSED on', DEVICE)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
