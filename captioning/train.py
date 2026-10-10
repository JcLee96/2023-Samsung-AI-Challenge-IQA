"""Train the M2 Transformer captioner: cross-entropy first, then SCST (CIDEr reward) once validation CIDEr plateaus.

Usage: python -m captioning.train [--resume_last | --resume_best]
"""
import argparse
import itertools
import multiprocessing
import os
import pickle
import random
import sys
from shutil import copyfile

import numpy as np
import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, ic_config, paths  # noqa: E402
from captioning import evaluation  # noqa: E402
from captioning.data import COCO, DataLoader, ImageDetectionsField, RawField  # noqa: E402
from captioning.evaluation import Cider, PTBTokenizer  # noqa: E402
from captioning.predict import build_captioning_model, build_text_field, load_vocab  # noqa: E402


def xe_lr(base_lr):
    """Step schedule for the cross-entropy stage (warm-up over 3 epochs, then x0.2 decays)."""
    def fn(s):
        if s <= 3:
            return base_lr * s / 4
        if s <= 10:
            return base_lr
        if s <= 12:
            return base_lr * 0.2
        return base_lr * 0.2 * 0.2
    return fn


def rl_lr(base_lr, refine_epoch):
    """Step schedule for the SCST stage."""
    def fn(s):
        if s <= refine_epoch:
            return base_lr
        if s <= refine_epoch + 3:
            return base_lr * 0.2
        if s <= refine_epoch + 6:
            return base_lr * 0.2 * 0.2
        return base_lr * 0.2 * 0.2 * 0.2
    return fn


def xe_step(model, detections, captions, loss_fn, vocab_size):
    """Teacher-forced next-token NLL for one batch."""
    out = model(detections, captions)
    captions_gt = captions[:, 1:].contiguous()
    out = out[:, :-1].contiguous()
    return loss_fn(out.view(-1, vocab_size), captions_gt.view(-1))


def train_xe(model, dataloader, optim, scheduler, loss_fn, text_field, device, epoch):
    """One cross-entropy epoch; returns the mean loss."""
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            loss = xe_step(model, detections, captions, loss_fn, len(text_field.vocab))
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    return running_loss / len(dataloader)


def train_scst(model, dataloader, optim, scheduler, cider, text_field, device, epoch, seq_len=20, beam_size=5):
    """One self-critical epoch with a mean-of-beams baseline; returns (loss, reward, baseline)."""
    tokenizer_pool = multiprocessing.Pool()
    running_reward = running_reward_baseline = running_loss = .0
    model.train()
    scheduler.step()
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = (-torch.mean(log_probs, -1) * (reward - reward_baseline)).mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
    tokenizer_pool.close()
    n = len(dataloader)
    return running_loss / n, running_reward / n, running_reward_baseline / n


@torch.no_grad()
def evaluate_loss(model, dataloader, loss_fn, text_field, device, epoch):
    """Mean teacher-forced validation loss."""
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            running_loss += xe_step(model, detections, captions, loss_fn, len(text_field.vocab)).item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_metrics(model, dataloader, text_field, device, epoch):
    """BLEU / METEOR / ROUGE / CIDEr of beam-search captions against references."""
    model.eval()
    gen, gts = {}, {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            out, _ = model.beam_search(images.to(device), 20, text_field.vocab.stoi['<eos>'], 3, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d' % (it, i)] = [' '.join(k for k, _ in itertools.groupby(gen_i)), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def rng_state():
    """Snapshot of every RNG used, stored in checkpoints."""
    return {'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(), 'random_rng_state': random.getstate()}


def restore_rng_state(data):
    """Restore RNGs from a checkpoint dict."""
    torch.set_rng_state(data['torch_rng_state'])
    if data.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(data['cuda_rng_state'])
    np.random.set_state(data['numpy_rng_state'])
    random.setstate(data['random_rng_state'])


def new_rl_optimizer(model, cfg, epoch):
    """Fresh SCST optimizer whose schedule is fast-forwarded to `epoch`."""
    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, rl_lr(cfg.rl_base_lr, cfg.refine_epoch_rl))
    for _ in range(epoch - 1):
        scheduler_rl.step()
    return optim_rl, scheduler_rl


def main(args, cfg=ic_config):
    """Run XE -> SCST training and write <exp_name>_last/_best and rl<epoch> checkpoints to cfg.ckpt_dir."""
    from torch.utils.tensorboard import SummaryWriter

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, args.exp_name))
    ckpt = lambda name: os.path.join(cfg.ckpt_dir, name)  # noqa: E731

    image_field = ImageDetectionsField(train_detections_path=paths.train_features,
                                       valid_detections_path=paths.train_features,
                                       max_detections=cfg.max_detections, load_in_tmp=False,
                                       train_json_path=paths.train_json, valid_json_path=paths.valid_json,
                                       features=True)
    text_field = build_text_field('spacy')
    dataset = COCO(image_field, text_field, '', DATA_DIR, DATA_DIR)
    train_dataset, val_dataset, _ = dataset.splits

    if os.path.isfile(cfg.vocab_path):
        text_field.vocab = load_vocab(cfg.vocab_path)
    else:
        text_field.build_vocab(train_dataset, val_dataset, min_freq=3)
        with open(cfg.vocab_path, 'wb') as f:
            pickle.dump(text_field.vocab, f)

    model = build_captioning_model(text_field.vocab, cfg).to(device)
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    cider_train = Cider(PTBTokenizer.tokenize(list(train_dataset.text)))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, xe_lr(cfg.xe_base_lr))
    optim_rl, scheduler_rl = new_rl_optimizer(model, cfg, 1)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl, best_cider, best_test_cider, patience, start_epoch = False, .0, .0, 0, 0

    if args.resume_last or args.resume_best:
        fname = ckpt('%s_%s.pth' % (args.exp_name, 'last' if args.resume_last else 'best'))
        if os.path.exists(fname):
            data = torch.load(fname)
            restore_rng_state(data)
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider, best_test_cider = data['best_cider'], data['best_test_cider']
            patience, use_rl = data['patience'], data['use_rl']
            if use_rl:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])
            else:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])
            print('Resuming from epoch %d, best cider %f' % (data['epoch'], best_cider))

    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                      num_workers=cfg.workers, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=cfg.batch_size // 5, shuffle=True,
                                           num_workers=cfg.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=cfg.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, scheduler, loss_fn, text_field, device, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, scheduler_rl,
                                                             cider_train, text_field, device, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
        writer.add_scalar('data/train_loss', train_loss, e)

        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, device, e)
        writer.add_scalar('data/val_loss', val_loss, e)
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, device, e)
        print('Validation scores', scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        best = val_cider >= best_cider
        if best:
            best_cider, patience = val_cider, 0
        else:
            patience += 1

        switch_to_rl = exit_train = False
        if patience == 5:
            if e < cfg.xe_least:
                patience = 0
            elif not use_rl:
                use_rl = switch_to_rl = True
                patience = 0
                optim_rl, scheduler_rl = new_rl_optimizer(model, cfg, e)
                print('Switching to RL')
            else:
                print('patience reached.')
                exit_train = True

        if e == cfg.xe_most and not use_rl:
            use_rl = switch_to_rl = True
            patience = 0
            optim_rl, scheduler_rl = new_rl_optimizer(model, cfg, e)
            print('Switching to RL')

        if switch_to_rl and not best:
            data = torch.load(ckpt('%s_best.pth' % args.exp_name))
            restore_rng_state(data)
            model.load_state_dict(data['state_dict'])
            print('Resuming from best epoch %d (cider %f)' % (data['epoch'], data['best_cider']))

        torch.save(dict(rng_state(), **{
            'epoch': e, 'val_loss': val_loss, 'val_cider': val_cider, 'state_dict': model.state_dict(),
            'optimizer': optim_rl.state_dict() if use_rl else optim.state_dict(),
            'scheduler': scheduler_rl.state_dict() if use_rl else scheduler.state_dict(),
            'patience': patience, 'best_cider': best_cider, 'best_test_cider': best_test_cider, 'use_rl': use_rl,
        }), ckpt('%s_last.pth' % args.exp_name))
        if best:
            copyfile(ckpt('%s_last.pth' % args.exp_name), ckpt('%s_best.pth' % args.exp_name))
        if use_rl:
            copyfile(ckpt('%s_last.pth' % args.exp_name), ckpt('rl%d.pth' % e))
        if exit_train:
            break
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the M2 Transformer captioner')
    parser.add_argument('--exp_name', type=str, default='m2')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    main(parser.parse_args())
