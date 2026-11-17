"""One class, IoU matching, 101-point AP; thresholds selected only on source val."""
import json, math
import numpy as np
import torch
from torchvision.ops import nms, box_iou
IOUS=np.arange(.5,.951,.05)

def decode(name,outputs):
    if name=='AA-YOLO': return outputs[0].float()
    values=[]
    for output in outputs:
        output=output.float(); h,w=output.shape[-2:]
        assert 512/h==512/w==8, 'Unexpected head stride'
        y,x=torch.meshgrid(torch.arange(h,device=output.device),torch.arange(w,device=output.device),indexing='ij')
        grid=torch.stack((x,y),-1).reshape(1,-1,2)
        v=output.flatten(2).permute(0,2,1).clone()
        v[:,:,:2]=(v[:,:,:2]+grid)*8; v[:,:,2:4]=v[:,:,2:4].exp()*8
        v[:,:,4:]=v[:,:,4:].sigmoid(); values.append(v)
    return torch.cat(values,1)

def postprocess(pred,record,trans):
    score=pred[:,4]*pred[:,5]; selected=score>=1e-5
    pred=pred[selected]; score=score[selected]
    if not len(pred): return np.zeros((0,5),dtype=np.float32)
    # Guard against divergent exponentials and bound NMS cost before clipping.
    valid=torch.isfinite(pred).all(1); pred=pred[valid]; score=score[valid]
    top=score.argsort(descending=True)[:1000]; pred=pred[top]; score=score[top]
    xyxy=torch.cat((pred[:,:2]-pred[:,2:4]/2,pred[:,:2]+pred[:,2:4]/2),1)
    keep=nms(xyxy,score,.5)[:100]; xyxy=xyxy[keep]; score=score[keep]
    sx,sy,dx,dy=trans
    xyxy[:,[0,2]]=(xyxy[:,[0,2]]-dx)/sx; xyxy[:,[1,3]]=(xyxy[:,[1,3]]-dy)/sy
    xyxy[:,[0,2]]=xyxy[:,[0,2]].clamp(0,record['width']); xyxy[:,[1,3]]=xyxy[:,[1,3]].clamp(0,record['height'])
    valid=(xyxy[:,2]>xyxy[:,0]) & (xyxy[:,3]>xyxy[:,1])
    return torch.cat((xyxy,score[:,None]),1)[valid].cpu().numpy()

def match_frame(pred,record):
    gt=np.array(record['boxes'],dtype=np.float32).reshape(-1,4)
    if len(gt): gt[:,2:]+=gt[:,:2]
    flags=np.zeros((len(pred),len(IOUS)),dtype=bool)
    if len(gt) and len(pred):
        overlaps=box_iou(torch.from_numpy(pred[:,:4]),torch.from_numpy(gt)).numpy()
        for k,thr in enumerate(IOUS):
            used=set()
            for i,overlap in enumerate(overlaps):
                available=[j for j in range(len(gt)) if j not in used and overlap[j]>=thr-1e-7]
                if available:
                    j=max(available,key=lambda j:overlap[j]); used.add(j); flags[i,k]=True
    center_flags=np.zeros(len(pred),dtype=bool)
    if len(gt) and len(pred):
        gc=(gt[:,:2]+gt[:,2:])/2; pc=(pred[:,:2]+pred[:,2:4])/2
        dist=np.linalg.norm(pc[:,None]-gc[None],axis=2); used=set()
        for i,row in enumerate(dist):
            candidates=[j for j in range(len(gt)) if j not in used and row[j]<=3.]
            if candidates:
                j=min(candidates,key=lambda j:row[j]); used.add(j); center_flags[i]=True
    return {'group':record['group'],'gt':len(gt),'scores':pred[:,4].tolist(),'tp':flags.tolist(),'center_tp':center_flags.tolist()}

def arrays(frames):
    scores=np.concatenate([np.asarray(f['scores']) for f in frames])
    tp=np.concatenate([np.asarray(f['tp'],dtype=bool).reshape(-1,10) for f in frames])
    center=np.concatenate([np.asarray(f['center_tp'],dtype=bool) for f in frames])
    order=np.argsort(-scores,kind='stable')
    return scores[order],tp[order],center[order],sum(f['gt'] for f in frames)

def ap101(tp,total,weights=None):
    if not total: return [None]*10
    if not len(tp): return [0.]*10
    weight=np.ones(len(tp)) if weights is None else weights
    tp_cum=np.cumsum(tp*weight[:,None],axis=0); fp_cum=np.cumsum((1-tp)*weight[:,None],axis=0)
    recall=tp_cum/total; precision=tp_cum/np.maximum(tp_cum+fp_cum,1e-12)
    values=[]
    for k in range(10):
        p=np.maximum.accumulate(precision[::-1,k])[::-1]; idx=np.searchsorted(recall[:,k],np.linspace(0,1,101),side='left')
        sampled=np.zeros(101); valid=idx<len(p); sampled[valid]=p[idx[valid]]; values.append(float(sampled.mean()))
    return values

def select_threshold(frames):
    scores,tp,_,total=arrays(frames)
    if not len(scores) or not total: return 1.0
    t=np.cumsum(tp[:,0]); n=np.arange(1,len(scores)+1)
    f1=2*t/(n+total)
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    return float(scores[ends[np.argmax(f1[ends])]])

def metrics(frames,threshold,bootstrap=0):
    scores,tp,center,total=arrays(frames); ap=ap101(tp,total); keep=scores>=threshold
    hit=int(tp[keep,0].sum()); n=int(keep.sum()); fp=n-hit; recall=hit/total if total else 0.
    precision=hit/n if n else 0.; f1=2*hit/(n+total) if n+total else 0.
    result={'AP50':ap[0],'AP50_95':float(np.mean(ap)) if total else None,'Precision':precision,'Recall':recall,'F1':f1,'FP_per_frame':fp/len(frames),'center_Pd_3px':int(center[keep].sum())/total if total else None,'frames':len(frames),'ground_truth':total,'threshold':threshold}
    if bootstrap and len({f['group'] for f in frames})>1:
        group_names=sorted({f['group'] for f in frames}); gi={g:i for i,g in enumerate(group_names)}
        gt=np.zeros(len(gi)); frame_group=[]
        raw_scores=[]
        for f in frames:
            gt[gi[f['group']]]+=f['gt']; frame_group.extend([gi[f['group']]]*len(f['scores'])); raw_scores.extend(f['scores'])
        pg=np.asarray(frame_group,dtype=int)[np.argsort(-np.array(raw_scores),kind='stable')]
        rng=np.random.default_rng(3407); vals=[]
        for _ in range(bootstrap):
            counts=np.bincount(rng.integers(0,len(gi),len(gi)),minlength=len(gi)); den=float((counts*gt).sum())
            if den: vals.append(ap101(tp,den,counts[pg])[0])
        result['AP50_cluster_CI95']=np.quantile(vals,[.025,.975]).tolist()
    return result

@torch.inference_mode()
def evaluate(name,model,loader,device,save_path=None):
    model.eval(); dataset=loader.dataset; frames=[]; predictions=[]
    for images,targets,multi,indices,trans in loader:
        images=images.to(device,non_blocking=True).float()/255
        output=decode(name,model(images))
        for v,index,tr in zip(output,indices,trans):
            record=dataset.records[index]; pred=postprocess(v,record,tr); frames.append(match_frame(pred,record))
            for p in pred:
                x,y,x2,y2,score=map(float,p); predictions.append({'image_id':record['id'],'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':score})
    if save_path:
        save_path.write_text(json.dumps({'frames':frames,'predictions':predictions},separators=(',',':')),encoding='utf8')
    return frames
