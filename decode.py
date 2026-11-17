"""Unclipped benchmark boxes sampled at original peaks, before offset rounding."""
import torch
from torch.nn import functional as F
import plateaus as fast


@torch.no_grad()
def decode_raw(outputs,reference,threshold=1e-5):
    if not 0<threshold<=1 or min(reference)<=0:raise ValueError('invalid threshold/reference')
    scores=outputs['center_logits'].float().sigmoid()
    if not torch.isfinite(scores).all() or not torch.isfinite(outputs['offsets']).all():raise ValueError('nonfinite predictions')
    candidate=(scores==F.max_pool2d(scores,3,1,1))&(scores>=threshold)
    h,w=scores.shape[-2:];results=[]
    for b in range(scores.shape[0]):
        positions=fast.representatives(scores[b,0].cpu().numpy(),candidate[b,0].cpu().numpy())
        anchors=torch.tensor(positions,device=scores.device,dtype=torch.long).reshape(-1,2)
        x,y=anchors.unbind(1)
        center=anchors.float()+outputs['offsets'][b,:,y,x].T.float().clamp(0,1-1e-6)
        bounds=center.new_tensor([w,h]);center=torch.minimum(center,torch.nextafter(bounds,torch.zeros_like(bounds)))
        size=outputs['log_size'][b,:,y,x].T.float()
        if not torch.isfinite(size).all():raise ValueError('nonfinite sizes')
        ref=size.new_tensor(reference)
        wh=ref*torch.minimum(size.clamp_min(-20),(size.new_tensor([w,h])/ref).log()).exp()
        confidence=scores[b,0,y,x]
        raw=torch.cat((center,wh,confidence[:,None],torch.ones_like(confidence[:,None])),1)
        results.append({'raw':raw,'anchors':anchors})
    return results


def decode_calibrated(outputs,reference):
    original={**outputs,'center_logits':outputs['main_peak_logits']}
    score_map=outputs['center_logits'].float().sigmoid();result=[]
    for b,item in enumerate(decode_raw(original,reference,threshold=torch.finfo(torch.float32).tiny)):
        x,y=item['anchors'].unbind(1)
        scores=score_map[b,0,y,x];keep=scores>=1e-5
        raw=item['raw'][keep].clone();raw[:,4]=scores[keep]
        result.append({'raw':raw,'anchors':item['anchors'][keep]})
    return result
