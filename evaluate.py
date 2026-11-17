"""Evaluate a manifest without fitting a threshold on test data."""
import argparse
import json
from pathlib import Path
from model import load_model
from predict import infer, SOURCE_THRESHOLD
from evaluation import match_frame, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=Path('metrics.json'))
    p.add_argument('--threshold', type=float, default=SOURCE_THRESHOLD)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    if not 0 <= a.threshold <= 1 or a.bootstrap < 0:
        p.error('Invalid threshold or bootstrap count')
    records = json.loads(a.manifest.read_text(encoding='utf-8'))
    if not records:
        p.error('Manifest is empty')
    root = a.data_root.resolve()
    model = load_model(a.weights, a.device)
    frames = []
    for record in records:
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path leaves --data-root')
        pred = infer(model, path, a.device)
        frames.append(match_frame(pred, record))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'metrics': metrics(frames, a.threshold, a.bootstrap),
                                   'frames': frames}, indent=2))
    print(json.dumps(metrics(frames, a.threshold), indent=2))


if __name__ == '__main__':
    main()
