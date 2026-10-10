"""Prepare captioning inputs: COCO-style annotation JSONs and 7x7 X-152 grid features (HDF5) for every image.

Usage:
    python -m captioning.prepare_data                 # annotations + features
    python -m captioning.prepare_data --skip-features # annotations only (no detectron2 needed)
Grid features need detectron2 and facebookresearch/grid-feats-vqa cloned into ./grid-feats-vqa with X-152.pth.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT, feature_config, paths  # noqa: E402


def read_csv(path):
    """Rows of a CSV as dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_annotations(train_csv=paths.train_csv, valid_mos_csv=paths.valid_mos_csv, test_csv=paths.test_csv,
                      train_json=paths.train_json, valid_json=paths.valid_json, test_json=paths.test_json):
    """Write train/valid/test COCO-style JSONs; valid = images listed in valid_mos.csv, ids unique across splits."""
    valid_paths = {row['img_path'] for row in read_csv(valid_mos_csv)}
    splits = {'train': {'images': [], 'annotations': []}, 'valid': {'images': [], 'annotations': []}}
    image_ids = {}
    for row in read_csv(train_csv):
        split = 'valid' if row['img_path'] in valid_paths else 'train'
        if row['img_path'] not in image_ids:
            image_ids[row['img_path']] = len(image_ids)
            splits[split]['images'].append({'file_name': row['img_path'], 'id': image_ids[row['img_path']]})
        ann_id = sum(len(s['annotations']) for s in splits.values())
        splits[split]['annotations'].append({'image_id': image_ids[row['img_path']], 'id': ann_id,
                                             'caption': row['comments']})
    test = {'images': [{'file_name': row['img_path'], 'id': i} for i, row in enumerate(read_csv(test_csv))]}
    for path, data in ((train_json, splits['train']), (valid_json, splits['valid']), (test_json, test)):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def pool_grid(feature_map, grid_size=7):
    """Average-pool a (1, C, H, W) conv5 map to (grid_size**2, C)."""
    import torch
    x = torch.nn.functional.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
    x = torch.squeeze(x).permute(1, 2, 0)
    return x.reshape(-1, x.size(-1))


def build_detector(cfg=feature_config):
    """X-152 Faster R-CNN backbone from grid-feats-vqa (detectron2)."""
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    sys.path.append(os.path.join(ROOT, 'grid-feats-vqa'))
    from grid_feats import add_attribute_config

    det_cfg = get_cfg()
    add_attribute_config(det_cfg)
    det_cfg.merge_from_file(cfg.detector_config)
    det_cfg.MODEL.RESNETS.RES5_DILATION = 1  # force the final residual block to have dilation 1
    det_cfg.MODEL.WEIGHTS = cfg.detector_weights
    det_cfg.freeze()
    model = build_model(det_cfg)
    DetectionCheckpointer(model, save_dir=det_cfg.OUTPUT_DIR).resume_or_load(det_cfg.MODEL.WEIGHTS, resume=True)
    model.load_state_dict(torch.load(det_cfg.MODEL.WEIGHTS)['model'])
    return model.eval()


def extract_grid_features(model, json_paths, img_dir, out_path, cfg=feature_config):
    """Write '<id>_grids' -> (49, 2048) float arrays for every image listed in `json_paths`."""
    import cv2
    import h5py
    import torch
    from PIL import Image
    from tqdm import tqdm

    images = [img for p in json_paths for img in json.load(open(p))['images']]
    with h5py.File(out_path, 'w') as f, torch.no_grad():
        for sample in tqdm(images, desc=os.path.basename(out_path)):
            image = np.array(Image.open(os.path.join(img_dir, os.path.basename(sample['file_name']))))
            image = cv2.resize(image, (cfg.resize, cfg.resize))
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            image = image.transpose(2, 0, 1)
            inputs = model.preprocess_image([{"image": torch.from_numpy(image)}])
            features = model.backbone(inputs.tensor)
            conv5 = model.roi_heads.get_conv5_features(features)
            f.create_dataset('%d_grids' % sample['id'], data=pool_grid(conv5, cfg.grid_size).cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--skip-features', action='store_true')
    args = parser.parse_args()
    write_annotations()
    if not args.skip_features:
        detector = build_detector()
        extract_grid_features(detector, [paths.train_json, paths.valid_json], paths.train_img_dir, paths.train_features)
        extract_grid_features(detector, [paths.test_json], paths.test_img_dir, paths.test_features)
