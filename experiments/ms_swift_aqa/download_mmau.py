"""Fetch the pinned official MMAU Test-mini release, with resumable parallel HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import file_digest, sha256
from pathlib import Path
import tempfile
import time

import requests

HF_REPO = 'gamma-lab-umd/MMAU-test-mini'
HF_REVISION = 'ccd9696c0111ea7060827598f310558df0b71b0a'
GITHUB_REVISION = '110127f54c0dfba3faa5ec9feee4a7e4148679c5'
PARQUET = 'test_mini.parquet'
PARQUET_SIZE = 1_213_407_227
CHUNK_SIZE = 8 * 1024 * 1024
SOURCE_HASHES = {
    PARQUET: '54b76f440db843546f9ad49a1518cc22371f63a266f034ba6bd4343b7b6fde86',
    'mmau-test-mini.json': '9f18fda99f8dbc2bc5ecd6323fb5063309f54969810b5c6f1caeb3b8d904cf1c',
    'evaluation.py': '85480e1c0dfe8ee1406e9c6e598eff0dca9e0216701f076faf69081c6aab1558',
}
RELEASE_HASHES = {**SOURCE_HASHES,
                  'README.md': 'e9ae1ff75576f13d49e436e2d41fe761ec89dbde0bec10d316a12f01ee6d2075',
                  'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def fetch_part(parts_dir, start):
    last = min(start + CHUNK_SIZE, PARQUET_SIZE) - 1
    size = last - start + 1
    path = parts_dir / str(start)
    if path.is_file() and path.stat().st_size == size:
        return path
    url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{PARQUET}'
    for attempt in range(6):
        try:
            with requests.get(url, params={'offset': start, 'end': last, 'attempt': attempt},
                              headers={'Range': f'bytes={start}-{last}'}, stream=True, timeout=(10, 60)) as resp:
                resp.raise_for_status()
                expected = f'bytes {start}-{last}/{PARQUET_SIZE}'
                if resp.status_code != 206 or resp.headers.get('Content-Range') != expected:
                    raise OSError(f'Unexpected server range for {expected}')
                partial = path.with_suffix('.partial')
                with partial.open('wb') as out:
                    for data in resp.iter_content(1024 * 1024):
                        out.write(data)
                if partial.stat().st_size != size:
                    raise OSError(f'Incomplete range {expected}')
                partial.replace(path)
                return path
        except (OSError, requests.RequestException):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))


def fetch_parquet(destination, workers):
    parts_dir = destination.parent / f'.{PARQUET}.{HF_REVISION}.parts'
    parts_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda start: fetch_part(parts_dir, start), range(0, PARQUET_SIZE, CHUNK_SIZE)))
    partial = destination.with_suffix('.partial')
    checksum = sha256()
    with partial.open('wb') as out:
        for part in parts:
            with part.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    checksum.update(chunk)
                    out.write(chunk)
    if checksum.hexdigest() != SOURCE_HASHES[PARQUET]:
        # The assembled digest cannot identify which cached range needs another download.
        for part in parts:
            part.unlink()
        partial.unlink()
        raise OSError('Parquet SHA-256 mismatch. Re-run to fetch fresh ranges.')
    partial.replace(destination)
    for part in parts:
        part.unlink()
    parts_dir.rmdir()


def fetch_document(path, expected):
    url = f'https://raw.githubusercontent.com/Sakshi113/MMAU/{GITHUB_REVISION}/{path.name}'
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        payload = response.content
    if sha256(payload).hexdigest() != expected:
        raise OSError(f'SHA-256 mismatch for {url}')
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
    Path(temporary.name).replace(path)


def download(root, workers, check_only=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.mmau-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, expected in RELEASE_HASHES.items():
            path = root / name
            if path.is_file():
                if digest(path) != expected:
                    raise ValueError(f'Existing file failed SHA-256 verification: {path}')
                print(f'reused {path} {expected}', flush=True)
            elif check_only:
                raise FileNotFoundError(path)
            else:
                if name == PARQUET:
                    fetch_parquet(path, workers)
                else:
                    fetch_document(path, expected)
                print(f'downloaded {path} {expected}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory for original release files')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent ranges for the audio Parquet')
    parser.add_argument('--check-only', action='store_true', help='Verify all files without using the network')
    args = parser.parse_args()
    download(args.root.resolve(), args.workers, args.check_only)
