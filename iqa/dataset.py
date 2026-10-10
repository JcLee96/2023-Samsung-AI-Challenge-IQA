"""DACON image datasets for IQA: 384x384 RGB in [-1, 1], plus MOS and its ordinal bin when labels exist."""
import bisect
import csv
import os

import cv2
import numpy as np
import torch

# Upper bounds of ordinal bins 0..19; MOS >= 8.0 falls into bin 20.
ORDINAL_BOUNDS = [2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,
                  6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0]


def mos_to_ordinal(mos):
    """Map a MOS value to one of 21 ordinal bins."""
    return bisect.bisect_right(ORDINAL_BOUNDS, mos)


def load_image(path, size=384):
    """Read an image as a float32 CHW RGB array in [0, 1] resized to size x size."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img.astype('float32') / 255, (2, 0, 1))


class Normalize:
    """(x - mean) / std on the image entry of a sample."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, sample):
        sample['d_img_org'] = (sample['d_img_org'] - self.mean) / self.std
        return sample


class RandFlip:
    """With probability `prob_aug`, flip axis 1 of the CHW array (kept identical to the submitted training run)."""

    def __init__(self, prob_aug):
        self.prob_aug = prob_aug

    def __call__(self, sample):
        if np.random.choice([1, 0], p=[self.prob_aug, 1 - self.prob_aug]) > 0.5:
            sample['d_img_org'] = np.fliplr(sample['d_img_org']).copy()
        return sample


class ToTensor:
    """Convert image (and MOS if present) to float tensors."""

    def __call__(self, sample):
        sample['d_img_org'] = torch.from_numpy(sample['d_img_org']).float()
        if 'score' in sample:
            sample['score'] = torch.from_numpy(sample['score']).float()
        return sample


class Compose:
    """Apply transforms in order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class SamsungIQADataset(torch.utils.data.Dataset):
    """Rows of a DACON CSV (`img_path` [+ `mos`]); images are looked up by basename inside `root`."""

    def __init__(self, csv_file, root, transform, with_labels=True):
        super().__init__()
        with open(csv_file, newline='') as f:
            rows = list(csv.DictReader(f))
        self.paths = [row['img_path'] for row in rows]
        self.mos = [np.array(float(row['mos'])).astype(np.float32) for row in rows] if with_labels else None
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        name = os.path.basename(self.paths[index])
        sample = {'d_name': name, 'd_img_org': load_image(os.path.join(self.root, name))}
        if self.mos is not None:
            sample['score'] = self.mos[index]
            sample['label'] = mos_to_ordinal(self.mos[index])
        return self.transform(sample)


def train_transform(prob_aug):
    """Augmentation used for training."""
    return Compose([Normalize(0.5, 0.5), RandFlip(prob_aug), ToTensor()])


def eval_transform():
    """Deterministic transform for validation and inference."""
    return Compose([Normalize(0.5, 0.5), ToTensor()])
