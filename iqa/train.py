"""Train the IQA model (MSE on MOS + CEO ordinal loss); keeps the checkpoint with the best SRCC + PLCC on valid.

Usage: python -m iqa.train
"""
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import iqa_config, paths  # noqa: E402
from iqa.dataset import SamsungIQADataset, eval_transform, train_transform  # noqa: E402
from iqa.model import CEOLoss, build_iqa_model  # noqa: E402


def setup_seed(seed):
    """Make runs deterministic."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def nearest_ordinal(label_logits, num_levels=21):
    """Snap ordinal outputs to the closest integer level in [0, num_levels)."""
    levels = np.linspace(0.0, num_levels - 1, num=num_levels)
    return np.argmin(np.abs(label_logits.detach().cpu().numpy().reshape(-1, 1) - levels), axis=1)


def summarize(preds, mos, ordinal_preds, ordinal_targets):
    """SRCC, PLCC, ordinal accuracy and macro-F1 for one epoch."""
    srcc, _ = spearmanr(preds, mos)
    plcc, _ = pearsonr(preds, mos)
    acc = accuracy_score(ordinal_targets, ordinal_preds)
    f1 = f1_score(ordinal_targets, ordinal_preds, average='macro')
    return srcc, plcc, acc, f1


def run_epoch(net, loader, mse, ceo, device, optimizer=None, scheduler=None, desc=''):
    """One pass over `loader`; trains when an optimizer is given. Returns (losses, metrics)."""
    training = optimizer is not None
    net.train(training)
    losses, or_losses = [], []
    preds, targets, ord_preds, ord_targets = [], [], [], []
    with torch.set_grad_enabled(training), tqdm(loader, desc=desc, unit='it') as pbar:
        for data in pbar:
            x = data['d_img_org'].to(device)
            score = data['score'].float().view(-1).to(device)
            label = data['label'].long().to(device)

            pred, ordinal = net(x)
            loss = mse(pred.view(-1), score)
            or_loss = ceo(ordinal.view(-1), label)
            if training:
                optimizer.zero_grad()
                (loss + or_loss).backward()
                optimizer.step()
                scheduler.step()

            losses.append(loss.item())
            or_losses.append(or_loss.item())
            preds.extend(pred.detach().cpu().numpy().reshape(-1))
            targets.extend(score.cpu().numpy().reshape(-1))
            ord_preds.extend(nearest_ordinal(ordinal))
            ord_targets.extend(label.cpu().numpy())
            pbar.set_postfix(loss=np.mean(losses), or_loss=np.mean(or_losses))

    srcc, plcc, acc, f1 = summarize(preds, targets, ord_preds, ord_targets)
    return {'loss': np.mean(losses), 'or_loss': np.mean(or_losses), 'srcc': srcc, 'plcc': plcc,
            'acc': acc, 'f1': f1}


def build_loaders(cfg, train_csv, valid_csv, img_dir):
    """Train/valid data loaders over the DACON MOS splits."""
    train_set = SamsungIQADataset(train_csv, img_dir, train_transform(cfg.prob_aug))
    valid_set = SamsungIQADataset(valid_csv, img_dir, eval_transform())
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              drop_last=True, shuffle=False)
    return train_loader, valid_loader


def train(cfg=iqa_config, train_csv=paths.train_mos_csv, valid_csv=paths.valid_mos_csv,
          img_dir=paths.train_img_dir, pretrained=True, device=None):
    """Full training loop; returns the best SRCC + PLCC reached on the valid split."""
    from torch.utils.tensorboard import SummaryWriter

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(cfg.log_dir, 'train.log'), filemode='w',
                        format='[%(asctime)s %(levelname)-8s] %(message)s', datefmt='%Y%m%d %H:%M:%S', force=True)
    logging.info(dict(cfg))
    writer = SummaryWriter(cfg.log_dir)

    train_loader, valid_loader = build_loaders(cfg, train_csv, valid_csv, img_dir)
    net = build_iqa_model(cfg, pretrained=pretrained)
    logging.info('#Params: %.2f M', sum(p.numel() for p in net.parameters()) / 1e6)
    net = nn.DataParallel(net).to(device) if device == 'cuda' else net.to(device)

    mse, ceo = nn.MSELoss(), CEOLoss(num_classes=cfg.num_ordinal_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)

    best = 0.0
    for epoch in range(cfg.n_epoch):
        start = time.time()
        stats = run_epoch(net, train_loader, mse, ceo, device, optimizer, scheduler, desc=f'Epoch {epoch} - train')
        logging.info('train epoch %d: %s', epoch + 1, stats)
        for k, v in stats.items():
            writer.add_scalar(f'train/{k}', v, epoch)

        if (epoch + 1) % cfg.val_freq == 0:
            stats = run_epoch(net, valid_loader, mse, ceo, device, desc=f'Epoch {epoch} - valid')
            logging.info('valid epoch %d: %s', epoch + 1, stats)
            for k, v in stats.items():
                writer.add_scalar(f'valid/{k}', v, epoch)
            if stats['srcc'] + stats['plcc'] > best:
                best = stats['srcc'] + stats['plcc']
                model = net.module if isinstance(net, nn.DataParallel) else net
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, f'epoch{epoch + 1}.pt'))
                logging.info('saved epoch %d (SRCC %.4f, PLCC %.4f)', epoch + 1, stats['srcc'], stats['plcc'])
        logging.info('epoch %d done in %.1f min', epoch + 1, (time.time() - start) / 60)
    writer.close()
    for handler in logging.getLogger().handlers[:]:
        handler.close()
        logging.getLogger().removeHandler(handler)
    return best


if __name__ == '__main__':
    train()
