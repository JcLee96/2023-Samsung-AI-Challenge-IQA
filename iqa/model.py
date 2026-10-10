"""MANIQA-based MOS regressor with an extra ordinal (CEO) head, the model used for the final submission."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Block

from .swin import SwinTransformer



class TABlock(nn.Module):
    """Transposed attention block: attention over the channel axis (from MANIQA)."""

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        return x + _x


class SaveOutput:
    """Forward hook that collects the outputs of every ViT block."""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQAOrdinal(nn.Module):
    """Three-stage MANIQA on a ViT-R50 backbone; returns (MOS per image, ordinal logit from the CLS token)."""

    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, pretrained=True,
                 backbone='vit_base_r50_s16_384', **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model(backbone, pretrained=pretrained)
        self.save_output = SaveOutput()
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                layer.register_forward_hook(self.save_output)

        self.tablock1 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])
        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution, depths=depths, num_heads=num_heads,
            embed_dim=embed_dim, window_size=window_size, dim_mlp=dim_mlp, scale=scale)

        self.tablock2 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution, depths=depths, num_heads=num_heads,
            embed_dim=embed_dim // 2, window_size=window_size, dim_mlp=dim_mlp, scale=scale)

        self.tablock3 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.swintransformer3 = SwinTransformer(
            patches_resolution=self.patches_resolution, depths=depths, num_heads=num_heads,
            embed_dim=embed_dim // 2, window_size=window_size, dim_mlp=dim_mlp, scale=scale)

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs), nn.ReLU())
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs), nn.Sigmoid())
        self.fc_ce = nn.Sequential(nn.Linear(embed_dim, 1))

    def extract_feature(self, save_output):
        """Concatenate patch tokens of ViT blocks 7-10 along the channel axis."""
        x = [save_output.outputs[i][:, 1:] for i in (6, 7, 8, 9)]
        return torch.cat(x, dim=2)

    def _stage(self, x, tablocks, conv, swin):
        """Run one TAB -> 1x1 conv -> Swin stage on a (B, C, H, W) map."""
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in tablocks:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        return swin(conv(x))

    def forward(self, x):
        feature = self.vit.forward_features(x)
        cls_x = feature[:, 0, :]

        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        x = self._stage(x, self.tablock1, self.conv1, self.swintransformer1)
        x = self._stage(x, self.tablock2, self.conv2, self.swintransformer2)
        x = self._stage(x, self.tablock3, self.conv3, self.swintransformer3)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        scores = []
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            scores.append(torch.sum(f * w) / torch.sum(w))
        return torch.stack(scores), self.fc_ce(cls_x)


class CEOLoss(nn.Module):
    """Cross-entropy ordinal loss: distance of a scalar prediction to each ordinal level used as logits."""

    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.level = torch.arange(self.num_classes)

    def forward(self, x, y):
        levels = self.level.repeat(len(y), 1).to(x.device)
        logit = x.repeat(self.num_classes, 1).permute(1, 0)
        logit = torch.abs(logit - levels)
        return F.cross_entropy(-logit, y, reduction='mean')


def build_iqa_model(cfg, pretrained=True):
    """Build the IQA network from an `iqa_config`-style dict."""
    return MANIQAOrdinal(
        embed_dim=cfg.embed_dim, num_outputs=cfg.num_outputs, dim_mlp=cfg.dim_mlp,
        patch_size=cfg.patch_size, img_size=cfg.img_size, window_size=cfg.window_size,
        depths=cfg.depths, num_heads=cfg.num_heads, num_tab=cfg.num_tab, scale=cfg.scale,
        pretrained=pretrained, backbone=cfg.get('backbone', 'vit_base_r50_s16_384'))
