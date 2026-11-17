import math
import torch
from torch import nn
from torch.nn import functional as F


class REBlock(nn.Module):
    def __init__(self, channels, cfg, groups=8, eps=1e-5, affine=True):
        super().__init__()
        self.channels = channels
        self.dilations = tuple(cfg["dilations"])
        self.zero_sum = cfg["zero_sum_kernel"]
        self.conditional = cfg["conditional_mixing"]
        self.kernels = nn.ParameterList([nn.Parameter(torch.empty(channels, 1, 3, 3)) for _ in self.dilations])
        for kernel in self.kernels:
            nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))
        if self.conditional:
            self.gate = nn.Sequential(nn.Conv2d(4, cfg["gate_hidden"], 1), nn.SiLU(),
                                      nn.Conv2d(cfg["gate_hidden"], len(self.dilations), 1))
            if cfg["gate_last_layer_zero_init"]:
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
        else:
            self.gate = None
        expanded = channels * cfg["expansion"]
        self.w1 = nn.Conv2d(channels, expanded, 1, bias=False)
        self.norm = nn.GroupNorm(groups, expanded, eps=eps, affine=affine)
        self.w2 = nn.Conv2d(expanded, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), cfg["residual_gamma_init"]))
        self.last_stats = {}

    def effective_kernels(self):
        return [q - q.mean(dim=(-2, -1), keepdim=True) if self.zero_sum else q for q in self.kernels]

    def mixing_weights(self, feature, condition):
        if self.gate is None:
            return feature.new_full((feature.shape[0], 3, *feature.shape[-2:]), 1 / 3)
        if condition.shape[-2:] != feature.shape[-2:]:
            condition = F.interpolate(condition, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return self.gate(condition).softmax(dim=1)

    def forward(self, feature, condition, diagnostics=False):
        alpha = self.mixing_weights(feature, condition)
        mixed = torch.zeros_like(feature)
        for j, (kernel, dilation) in enumerate(zip(self.effective_kernels(), self.dilations)):
            response = F.conv2d(F.pad(feature, (dilation,) * 4, mode="reflect"), kernel,
                                dilation=dilation, groups=self.channels)
            mixed = mixed + alpha[:, j:j + 1] * response
        out = feature + self.gamma * self.w2(F.silu(self.norm(self.w1(mixed))))
        if diagnostics:
            weights = alpha.detach().float()
            self.last_stats = {"alpha_mean": weights.mean(dim=(0, 2, 3)).tolist(),
                               "alpha_entropy": float(-(weights * weights.clamp_min(1e-12).log()).sum(1).mean()),
                               "gamma_min": float(self.gamma.detach().min()), "gamma_max": float(self.gamma.detach().max())}
        return out


class StaticBlock(REBlock):
    def __init__(self, channels, cfg, **kwargs):
        if cfg["zero_sum_kernel"] or cfg["conditional_mixing"]:
            raise ValueError("StaticBlock is unconstrained, with fixed equal mixing")
        super().__init__(channels, cfg, **kwargs)

