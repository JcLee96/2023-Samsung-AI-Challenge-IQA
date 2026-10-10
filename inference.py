"""End-to-end inference: predict MOS (Task 1) and a caption (Task 2) for every test image and write the submission CSV.

Usage: python inference.py [--iqa-ckpt PATH] [--ic-ckpt PATH] [--out PATH]
"""
import argparse
import csv
import json
import os

import torch

from config import ic_config, iqa_config, paths
from captioning.predict import generate_captions, load_captioning_model
from iqa.predict import load_iqa_model, predict_mos

SUBMISSION_COLUMNS = ['img_name', 'mos', 'comments']


def read_test_names(test_csv):
    """Test image names in CSV order, left-padded with zeros to 10 characters as in the DACON ids."""
    with open(test_csv, newline='') as f:
        return [str(row['img_name']).rjust(10, '0') for row in csv.DictReader(f)]


def write_submission(path, names, mos, captions):
    """Write `img_name,mos,comments` rows in the given order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write(','.join(SUBMISSION_COLUMNS) + '\n')
        for name in names:
            f.write(name + ',' + str(mos[name]) + ',' + captions[name] + '\n')


def run_captioning(device, checkpoint, features_path, test_json, cache_path):
    """Captions for all test images; reuses `cache_path` when it already exists."""
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            return {item['id'].split('.')[0]: item['captions'] for item in json.load(f)}

    import h5py
    model, text_field = load_captioning_model(ic_config, device, checkpoint)
    with open(test_json) as f:
        images = json.load(f)['images']
    with h5py.File(features_path, 'r') as features:
        captions = generate_captions(model, text_field, features, images, device,
                                     max_len=ic_config.max_len, beam_size=ic_config.beam_size)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump([{'id': name, 'captions': cap} for name, cap in captions.items()], f, indent=4)
    return captions


def main():
    """Run both models on the test split and write the submission file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--iqa-ckpt', default=iqa_config.model_path)
    parser.add_argument('--ic-ckpt', default=ic_config.checkpoint_path)
    parser.add_argument('--out', default=paths.submission_csv)
    parser.add_argument('--caption-cache', default=paths.caption_cache_json)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    captions = run_captioning(device, args.ic_ckpt, paths.test_features, paths.test_json, args.caption_cache)
    iqa_net = load_iqa_model(iqa_config, device, args.iqa_ckpt)
    mos = predict_mos(iqa_net, paths.test_csv, paths.test_img_dir, device, num_avg_val=iqa_config.num_avg_val)
    write_submission(args.out, read_test_names(paths.test_csv), mos, captions)
    print('Submission written to', args.out)


if __name__ == '__main__':
    main()
