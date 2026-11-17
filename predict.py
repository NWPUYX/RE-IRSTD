"""Single-image inference using the paper's frozen letterbox and decode rules."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import load_model
from decode import decode_calibrated
from evaluation import postprocess

SOURCE_THRESHOLD = 0.5791078209877014


def preprocess(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        ratio = min(512 / w, 512 / h)
        nw, nh = int(w * ratio), int(h * ratio)
        if min(nw, nh) < 1:
            raise ValueError('Image aspect ratio gives an empty resized dimension')
        dx, dy = (512 - nw) // 2, (512 - nh) // 2
        canvas = Image.new('RGB', (512, 512), (128, 128, 128))
        canvas.paste(im.resize((nw, nh), Image.Resampling.BICUBIC), (dx, dy))
        gray = np.asarray(canvas.convert('L'), dtype=np.float32) / 255.
    return torch.from_numpy(gray.copy())[None, None], (nw / w, nh / h, dx, dy), (w, h)


@torch.inference_mode()
def infer(model, path, device='cpu'):
    image, trans, (w, h) = preprocess(path)
    out = model(image.to(device))
    raw = decode_calibrated(out, (16., 16.))[0]['raw']
    return postprocess(raw, {'width': w, 'height': h}, trans)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=SOURCE_THRESHOLD)
    p.add_argument('--output', type=Path, default=Path('prediction.json'))
    a = p.parse_args()
    if not 0 <= a.threshold <= 1:
        p.error('--threshold must be in [0,1]')
    pred = infer(load_model(a.weights, a.device), a.image, a.device)
    boxes = [dict(bbox_xyxy=row[:4].tolist(), score=float(row[4]))
             for row in pred if row[4] >= a.threshold]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'image': a.image.name, 'threshold': a.threshold,
                                   'detections': boxes}, indent=2))
    print(f'{len(boxes)} detections saved to {a.output}')


if __name__ == '__main__':
    main()
