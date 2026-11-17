from torch import nn
from torch.nn import functional as F
import torch
from blocks import REBlock, StaticBlock


def conv_norm(inp, out, kernel, cfg, stride=1):
    return nn.Sequential(nn.Conv2d(inp, out, kernel, stride=stride, padding=kernel // 2, bias=False),
                         nn.GroupNorm(cfg["norm_groups"], out, eps=cfg["norm_epsilon"], affine=cfg["norm_affine"]), nn.SiLU())


class BlockStack(nn.Module):
    def __init__(self, channels, number, cfg):
        super().__init__()
        block = StaticBlock if cfg["block"]["kind"] == "static" else REBlock
        self.blocks = nn.ModuleList([block(channels, cfg["block"], groups=cfg["norm_groups"],
                                          eps=cfg["norm_epsilon"], affine=cfg["norm_affine"]) for _ in range(number)])

    def forward(self, x, c, diagnostics=False):
        for block in self.blocks:
            x = block(x, c, diagnostics=diagnostics)
        return x


class UpFuse(nn.Module):
    def __init__(self, deep, skip, cfg):
        super().__init__()
        self.reduce = conv_norm(deep, skip, 1, cfg)
        self.fuse = conv_norm(2 * skip, skip, 1, cfg)
        self.blocks = BlockStack(skip, cfg["decoder_blocks_per_stage"], cfg)

    def forward(self, deep, skip, c, diagnostics=False):
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(self.fuse(torch.cat((self.reduce(deep), skip), dim=1)), c, diagnostics)


class Backbone(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        a, b, c, d = cfg["channels"]
        self.stem = conv_norm(inputs, a, 3, cfg)
        self.has_t = cfg.get("full_resolution_enabled", True)
        self.target_path = BlockStack(a, cfg["full_resolution_blocks"], cfg) if self.has_t else None
        self.down = nn.ModuleList([conv_norm(i, o, 3, cfg, 2) for i, o in zip((a, b, c), (b, c, d))])
        self.encoders = nn.ModuleList([BlockStack(width, cfg["encoder_blocks_per_stage"], cfg) for width in (b, c, d)])
        self.decoders = nn.ModuleList([UpFuse(d, c, cfg), UpFuse(c, b, cfg), UpFuse(b, a, cfg)])
        self.final = conv_norm(2 * a if self.has_t else a, a, 1, cfg)

    def forward(self, x, c, diagnostics=False):
        stem = self.stem(x)
        target = self.target_path(stem, c, diagnostics) if self.has_t else None
        skips = [stem]
        encoded = stem  # Context path branches from F0, never from T.
        for down, blocks in zip(self.down, self.encoders):
            encoded = blocks(down(encoded), c, diagnostics)
            skips.append(encoded)
        decoded = skips[-1]
        for fuse, skip in zip(self.decoders, reversed(skips[:-1])):
            decoded = fuse(decoded, skip, c, diagnostics)
        fused = torch.cat((target, decoded), dim=1) if self.has_t else decoded
        return self.final(fused), target

