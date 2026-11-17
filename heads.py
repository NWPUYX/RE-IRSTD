import math
from torch import nn


class Head(nn.Sequential):
    def __init__(self, channels, hidden, out):
        super().__init__(nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, out, 1))


class DetectorHeads(nn.Module):
    def __init__(self, channels, cfg, size_head):
        super().__init__()
        self.center = Head(channels, cfg["hidden_channels"], 1)
        self.offset = Head(channels, cfg["hidden_channels"], 2)
        self.size = Head(channels, cfg["hidden_channels"], 2) if size_head else None
        self.aux = nn.Conv2d(channels, 1, 1) if cfg["auxiliary_center"] else None
        bias = math.log(cfg["center_init_probability"] / (1 - cfg["center_init_probability"]))
        for layer in (self.center[-1], self.aux):
            if layer is not None:
                nn.init.normal_(layer.weight, std=.001)
                nn.init.constant_(layer.bias, bias)
        for head in (self.offset, self.size):
            if head is not None:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, feature, target):
        return {"center_logits": self.center(feature), "offsets": self.offset(feature),
                "log_size": self.size(feature) if self.size is not None else None,
                "auxiliary_center_logits": self.aux(target) if self.aux is not None else None}

