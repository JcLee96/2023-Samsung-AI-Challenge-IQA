"""Paths and hyperparameters for both tasks; every path is relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SAMSUNG_DATA_DIR", os.path.join(ROOT, "data"))
CKPT_DIR = os.environ.get("SAMSUNG_CKPT_DIR", os.path.join(ROOT, "checkpoints"))
OUTPUT_DIR = os.environ.get("SAMSUNG_OUTPUT_DIR", os.path.join(ROOT, "outputs"))


class Config(dict):
    """Dict with attribute access (taken from MANIQA)."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def data_path(*parts):
    """Join a path under the dataset directory."""
    return os.path.join(DATA_DIR, *parts)


paths = Config({
    "train_img_dir": data_path("train"),
    "test_img_dir": data_path("test"),
    "train_csv": data_path("train.csv"),          # img_name, img_path, mos, comments
    "test_csv": data_path("test.csv"),            # img_name, img_path
    "train_mos_csv": data_path("train_mos.csv"),  # img_path, mos (IQA train split)
    "valid_mos_csv": data_path("valid_mos.csv"),  # img_path, mos (IQA valid split)
    "train_json": data_path("train.json"),        # COCO-style caption annotations
    "valid_json": data_path("valid.json"),
    "test_json": data_path("test.json"),
    "train_features": data_path("train.hdf5"),    # grid features of train + valid images
    "test_features": data_path("test.hdf5"),
    "submission_csv": os.path.join(OUTPUT_DIR, "submission.csv"),
    "caption_cache_json": os.path.join(OUTPUT_DIR, "predicted_captions.json"),
})

iqa_config = Config({
    # model (MANIQA with a ViT-R50 backbone and an ordinal head)
    "backbone": "vit_base_r50_s16_384",
    "patch_size": 16,
    "img_size": 384,
    "embed_dim": 768,
    "dim_mlp": 768,
    "num_heads": [4, 4],
    "window_size": 4,
    "depths": [2, 2],
    "num_outputs": 1,
    "num_tab": 2,
    "scale": 0.8,
    # optimisation
    "seed": 42,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "n_epoch": 300,
    "val_freq": 1,
    "T_max": 50,
    "eta_min": 0,
    "num_avg_val": 1,
    "num_workers": 8,
    "crop_size": 384,
    "prob_aug": 0.7,
    "num_ordinal_classes": 21,
    # checkpoints
    "ckpt_dir": os.path.join(CKPT_DIR, "iqa"),
    "log_dir": os.path.join(OUTPUT_DIR, "iqa_logs"),
    "model_path": os.path.join(CKPT_DIR, "epoch_iqa.pt"),
})

ic_config = Config({
    # model (Meshed-Memory Transformer on 7x7 X-152 grid features)
    "n_enc_layers": 3,
    "n_dec_layers": 3,
    "memory_slots": 40,
    "max_decoder_len": 90,
    "max_detections": 49,
    "vocab_path": os.path.join(ROOT, "captioning", "vocab.pkl"),
    # decoding
    "max_len": 23,
    "beam_size": 3,
    # optimisation
    "seed": 1234,
    "batch_size": 50,
    "workers": 4,
    "xe_least": 20,
    "xe_most": 30,
    "refine_epoch_rl": 5,
    "xe_base_lr": 1e-4,
    "rl_base_lr": 5e-6,
    # checkpoints
    "ckpt_dir": os.path.join(CKPT_DIR, "captioning"),
    "log_dir": os.path.join(OUTPUT_DIR, "captioning_logs"),
    "checkpoint_path": os.path.join(CKPT_DIR, "epoch_ic.pt"),
})

feature_config = Config({
    "detector_config": os.path.join(ROOT, "grid-feats-vqa", "configs", "X-152-grid.yaml"),
    "detector_weights": os.path.join(ROOT, "grid-feats-vqa", "X-152.pth"),
    "resize": 512,
    "grid_size": 7,
})
