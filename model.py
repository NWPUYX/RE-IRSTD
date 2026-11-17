"""Paper RE-IRSTD, with state-dict keys preserved from the frozen experiment."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from heads import DetectorHeads
from population import PopulationBackbone
from integral_evidence import IntegralEvidence


class REIRSTD(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = json.loads(Path(__file__).with_name('model_config.json').read_text())
        self.evidence = IntegralEvidence(cfg['evidence'])
        self.backbone = PopulationBackbone(6, cfg)
        self.heads = DetectorHeads(cfg['channels'][0], cfg['heads'], True)
        self.set_mode()

    def set_mode(self):
        """Fixed source moments only in T; context and fusion use online GN."""
        self.backbone.target_stem_norm.use_population = True
        for block in self.backbone.target_path.blocks:
            block.norm.use_population = True
        self.backbone.final[1].use_population = False

    def forward(self, image, return_features=False, diagnostics=False):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 32:
            raise ValueError('Expected grayscale [B,1,H,W], H/W >= 32')
        h, w = image.shape[-2:]
        z, c = self.evidence(image, diagnostics)
        padding = (0, (-w) % 8, 0, (-h) % 8)
        final, target = self.backbone(F.pad(z, padding, mode='replicate'),
                                     F.pad(c, padding, mode='replicate'), diagnostics)
        main = self.heads.center(final)[..., :h, :w]
        auxiliary = self.heads.aux(target)[..., :h, :w]
        out = {'main_peak_logits': main, 'center_logits': .75 * main + .25 * auxiliary,
               'auxiliary_center_logits': auxiliary,
               'offsets': self.heads.offset(final)[..., :h, :w],
               'log_size': self.heads.size(target)[..., :h, :w]}
        if return_features:
            out['final_features'] = final[..., :h, :w]
        return out


def load_model(weights=None, device='cpu'):
    weights = Path(weights) if weights else Path(__file__).with_name('re_irstd.pt')
    model = REIRSTD()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()
