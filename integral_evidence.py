"""Same annular equations using local FP64 integral sums; FP64 accumulation with FP32 outputs."""
import torch
from torch.nn import functional as F
from relative_evidence import RelativeEvidenceEncoder


class IntegralEvidence(RelativeEvidenceEncoder):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.annuli=tuple(tuple(pair) for pair in cfg['annuli'])

    def forward(self,image,diagnostics=False):
        with torch.autocast(device_type=image.device.type,enabled=False):
            value=image.float().double();h,w=value.shape[-2:];radius=max(self.radii)
            padded=F.pad(value,(radius,)*4,mode='reflect')
            integral=F.pad(padded.cumsum(-2).cumsum(-1),(1,0,1,0))
            integral2=F.pad(padded.square().cumsum(-2).cumsum(-1),(1,0,1,0))
            def window(table,side):
                offset=radius-side//2
                y0,y1=slice(offset,offset+h),slice(offset+side,offset+side+h)
                x0,x1=slice(offset,offset+w),slice(offset+side,offset+side+w)
                return table[...,y1,x1]-table[...,y0,x1]-table[...,y1,x0]+table[...,y0,x0]
            means,scales,residuals=[],[],[]
            for inner,outer in self.annuli:
                n=outer*outer-inner*inner
                mean=(window(integral,outer)-window(integral,inner))/n
                second=(window(integral2,outer)-window(integral2,inner))/n
                variance=(second-mean.square()).clamp_min(0)
                scale=(variance+self.floor**2).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
                means.append(mean);scales.append(scale);residuals.append((value-mean)/scale)
            signed=torch.cat(residuals,1).clamp(-self.clip,self.clip)
            z=torch.cat((signed,signed.abs()),1).float()
            mx=F.pad(means[2],(1,1,0,0),mode='reflect');my=F.pad(means[2],(0,0,1,1),mode='reflect')
            gx=(mx[...,2:]-mx[...,:-2])*.5/scales[2]
            gy=(my[...,2:,:]-my[...,:-2,:])*.5/scales[2]
            c=torch.cat((gx,gy,(scales[0]/scales[2]).log(),(scales[1]/scales[2]).log()),1).clamp(-self.clip,self.clip).float()
            if diagnostics:
                raw=torch.cat(residuals,1);s=torch.cat(scales,1)
                self.last_stats={'z_clip_fraction':float((raw.abs()>=self.clip).float().mean()),
                    's_min':float(s.min()),'s_mean':float(s.mean()),'s_max':float(s.max())}
            return z,c
