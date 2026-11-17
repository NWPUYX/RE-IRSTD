import torch
from torch import nn
from torch.nn import functional as F


class RelativeEvidenceEncoder(nn.Module):
    """Fixed annular statistics; no absolute-intensity bypass for Z models."""
    def __init__(self, cfg):
        super().__init__()
        self.floor = float(cfg["sigma_floor"])
        self.clip = float(cfg["clip_abs"])
        self.radii = []
        if len(cfg["annuli"]) != 3:
            raise ValueError("Exactly three evidence scales are required")
        for i, (inner, outer) in enumerate(cfg["annuli"]):
            if inner < 1 or outer <= inner or inner % 2 != 1 or outer % 2 != 1 or outer > 63:
                raise ValueError("Odd annuli 1 <= inner < outer <= 63 required for H,W>=32")
            kernel = torch.ones(outer, outer, dtype=torch.float32)
            begin = (outer - inner) // 2
            kernel[begin:begin + inner, begin:begin + inner] = 0
            kernel /= outer * outer - inner * inner
            self.register_buffer(f"kernel_{i}", kernel[None, None])
            self.radii.append(outer // 2)
        self.register_buffer("dx", torch.tensor([-.5, 0., .5]).view(1, 1, 1, 3))
        self.register_buffer("dy", torch.tensor([-.5, 0., .5]).view(1, 1, 3, 1))
        self.last_stats = {}

    def forward(self, image, diagnostics=False):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float()
            means, scales, residuals = [], [], []
            for i, radius in enumerate(self.radii):
                padded = F.pad(image, (radius,) * 4, mode="reflect")
                kernel = getattr(self, f"kernel_{i}").float()
                mean = F.conv2d(padded, kernel)
                variance = (F.conv2d(padded.square(), kernel) - mean.square()).clamp_min(0)
                scale = (variance + self.floor ** 2).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
                residuals.append((image - mean) / scale)
                means.append(mean)
                scales.append(scale)
            signed = torch.cat(residuals, dim=1).clamp(-self.clip, self.clip)
            z = torch.cat((signed, signed.abs()), dim=1)
            gx = F.conv2d(F.pad(means[2], (1, 1, 0, 0), mode="reflect"), self.dx.float()) / scales[2]
            gy = F.conv2d(F.pad(means[2], (0, 0, 1, 1), mode="reflect"), self.dy.float()) / scales[2]
            c = torch.cat((gx, gy, (scales[0] / scales[2]).log(), (scales[1] / scales[2]).log()), dim=1)
            c = c.clamp(-self.clip, self.clip)
            if diagnostics:
                raw = torch.cat(residuals, dim=1).detach()
                s = torch.cat(scales, dim=1).detach()
                self.last_stats = {"z_clip_fraction": float((raw.abs() >= self.clip).float().mean()),
                                   "s_min": float(s.min()), "s_mean": float(s.mean()), "s_max": float(s.max())}
            return z, c

