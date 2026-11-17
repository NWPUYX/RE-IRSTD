"""Convert the released content manifest to one domain/split inference manifest."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', required=True, choices=['DAUB-R', 'IRDST-H', 'IRSTD-UAV', 'ITSDT-15K'])
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--data-root', required=True, type=Path,
                   help='Directory containing the relative_id paths from the content manifest')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--verify-hashes', action='store_true')
    a = p.parse_args()
    manifest = Path(__file__).with_name('dataset_file_manifest.csv.gz')
    rows = []
    root = a.data_root.resolve()
    with gzip.open(manifest, 'rt', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['domain'] != a.domain or r['split'] != a.split:
                continue
            path = (root / r['relative_id']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image path leaves --data-root')
            if not path.is_file():
                raise FileNotFoundError(path)
            if a.verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != r['file_sha256']:
                raise ValueError(f'File hash mismatch: {r["relative_id"]}')
            rows.append(dict(id=r['record_id'], group=r['group'], path=r['relative_id'],
                             width=int(r['width']), height=int(r['height']),
                             boxes=json.loads(r['boxes_xywh'])))
    if not rows:
        p.error('No records for the specified domain/split')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(f'{len(rows)} records saved to {a.output}')


if __name__ == '__main__':
    main()
