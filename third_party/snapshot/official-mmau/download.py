"""Download the pinned official MMAU artifacts and verify the Parquet LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep, monotonic
from hashlib import sha256, file_digest
import os
import requests

root=Path(__file__).resolve().parent
hf_revision='ccd9696c0111ea7060827598f310558df0b71b0a'
git_revision='110127f54c0dfba3faa5ec9feee4a7e4148679c5'
base=f'https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini/resolve/{hf_revision}/test_mini.parquet'
length=1213407227
expected='54b76f440db843546f9ad49a1518cc22371f63a266f034ba6bd4343b7b6fde86'
size=8*1024*1024

def small_file(name):
    url=f'https://raw.githubusercontent.com/Sakshi113/MMAU/{git_revision}/{name}'
    r=requests.get(url,timeout=30);r.raise_for_status();(root/name).write_bytes(r.content)

def fetch(offset):
    stop=min(offset+size,length)-1
    part=parts / ('%012d'%offset)
    if part.exists() and part.stat().st_size==stop-offset+1:return
    for attempt in range(6):
        try:
            url=f'{base}?offset={offset}&end={stop}&attempt={attempt}'
            with requests.get(url,headers={'Range':f'bytes={offset}-{stop}'},stream=True,timeout=(10,60)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {offset}-{stop}/{length}',(r.status_code,r.headers)
                tmp=part.with_suffix('.partial')
                with tmp.open('wb') as out:
                    for chunk in r.iter_content(1024*1024):out.write(chunk)
            assert tmp.stat().st_size==stop-offset+1
            tmp.replace(part);return
        except Exception as e:
            print(f'offset {offset} retry {attempt+1}: {type(e).__name__} {str(e)[:100]}',flush=True)
            if attempt==5:raise
            sleep(min(2**attempt,16))

if __name__=='__main__':
    parts=root/'parts';parts.mkdir(exist_ok=True)
    t=monotonic();dest=root/'test_mini.parquet'
    for name in ['mmau-test-mini.json','evaluation.py','README.md','LICENSE']:small_file(name)
    if not dest.exists():
        offsets=list(range(0,length,size))
        with ThreadPoolExecutor(max_workers=16) as pool:
            for i,future in enumerate(as_completed(pool.submit(fetch,start) for start in offsets),1):
                future.result()
                if i%10==0:print(f'{i}/{len(offsets)} blocks after {monotonic()-t:.1f}s',flush=True)
        digest=sha256(); merged=dest.with_suffix('.tmp')
        with merged.open('wb') as out:
            for start in offsets:
                part=parts/('%012d'%start)
                with part.open('rb') as f:
                    while chunk:=f.read(1048576):digest.update(chunk);out.write(chunk)
        if digest.hexdigest()!=expected:raise ValueError(digest.hexdigest())
        merged.replace(dest)
        for start in offsets:(parts/('%012d'%start)).unlink()
        parts.rmdir()
    with dest.open('rb') as f: assert file_digest(f,'sha256').hexdigest()==expected
    print(f'Verified {length} bytes SHA256 {expected} after {monotonic()-t:.1f}s',flush=True)
