import torch
from torch import nn
from torch.nn import functional as F
from backbone import Backbone


class PopulationGN(nn.GroupNorm):
    """Original affine parameters; source-population moments when activated."""
    def __init__(self, groups, channels, eps=1e-5, affine=True):
        super().__init__(groups, channels, eps, affine)
        self.register_buffer('population_mean', torch.zeros(groups))
        self.register_buffer('population_variance', torch.ones(groups))
        self.use_population = False

    def forward(self, value, weight=None, bias=None):
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        if not self.use_population:
            return F.group_norm(value, self.num_groups, weight, bias, self.eps)
        b, c, h, w = value.shape
        grouped = value.float().reshape(b, self.num_groups, c//self.num_groups, h, w)
        mean = self.population_mean[None, :, None, None, None]
        variance = self.population_variance[None, :, None, None, None]
        result = ((grouped-mean)*torch.rsqrt(variance+self.eps)).reshape_as(value)
        if weight is not None:
            result = result*weight[None, :, None, None]
        if bias is not None:
            result = result+bias[None, :, None, None]
        return result


class PopulationBackbone(Backbone):
    def __init__(self, inputs, cfg):
        super().__init__(inputs, cfg)
        assert self.has_t
        a = cfg['channels'][0]
        # The affine tensors are supplied by the unchanged context stem GN.
        # This adds only population buffers, never a second affine parameter set.
        self.target_stem_norm = PopulationGN(cfg['norm_groups'], a, cfg['norm_epsilon'], affine=False)
        for block in self.target_path.blocks:
            old = block.norm
            block.norm = PopulationGN(old.num_groups, old.num_channels, old.eps, old.affine)
        old = self.final[1]
        self.final[1] = PopulationGN(old.num_groups, old.num_channels, old.eps, old.affine)

    def forward(self, x, c, diagnostics=False):
        pre = self.stem[0](x)
        stem = self.stem[2](self.stem[1](pre))
        target_stem = self.stem[2](self.target_stem_norm(pre, self.stem[1].weight, self.stem[1].bias))
        target = self.target_path(target_stem, c, diagnostics)
        skips = [stem]
        encoded = stem
        for down, blocks in zip(self.down, self.encoders):
            encoded = blocks(down(encoded), c, diagnostics)
            skips.append(encoded)
        decoded = skips[-1]
        for fuse, skip in zip(self.decoders, reversed(skips[:-1])):
            decoded = fuse(decoded, skip, c, diagnostics)
        return self.final(torch.cat((target, decoded), 1)), target
