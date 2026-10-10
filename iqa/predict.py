"""IQA inference: predicted MOS for every image of the test CSV."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SamsungIQADataset, eval_transform
from .model import build_iqa_model


def load_iqa_model(cfg, device, checkpoint=None, pretrained=False):
    """Build the IQA network and load trained weights when a checkpoint path is given."""
    net = build_iqa_model(cfg, pretrained=pretrained)
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return net.to(device).eval()


@torch.no_grad()
def predict_mos(net, csv_file, img_dir, device, num_avg_val=1, batch_size=1, num_workers=0):
    """Return {image name without extension: MOS as np.float32} averaged over `num_avg_val` passes."""
    dataset = SamsungIQADataset(csv_file, img_dir, eval_transform(), with_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    net.eval()
    results = {}
    for data in tqdm(loader, desc='IQA inference'):
        x = data['d_img_org'].to(device)
        pred = sum(net(x)[0] for _ in range(num_avg_val)) / num_avg_val
        for name, mos in zip(data['d_name'], pred.cpu().numpy()):
            results[name.split('.')[0]] = mos  # np.float32, written to the CSV with str()
    return results
