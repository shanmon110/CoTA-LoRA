"""Check every file in a local Hugging Face snapshot against immutable revision metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, metadata_path):
    meta = json.loads(metadata_path.read_text())
    if meta['id'] != 'Qwen/Qwen2.5-Omni-3B' or meta['sha'] != 'f75b40e3da2003cdd6e1829b1f420ca70797c34e':
        raise ValueError('Unexpected upstream model identity')
    verified = {}
    for entry in meta['siblings']:
        name = entry['rfilename']
        path = root / name
        size = path.stat().st_size
        assert size == entry['size'], name
        sha = hashlib.sha256()
        git = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as f:
            while chunk := f.read(8 << 20):
                sha.update(chunk)
                if 'lfs' not in entry:
                    git.update(chunk)
        if 'lfs' in entry:
            assert sha.hexdigest() == entry['lfs']['sha256'], name
        else:
            assert git.hexdigest() == entry['blobId'], name
        verified[name] = {'sha256': sha.hexdigest(), 'bytes': size}
    return dict(model=meta['id'], revision=meta['sha'], path=str(root.resolve()), files=verified)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('metadata', type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.model, args.metadata), indent=2))
