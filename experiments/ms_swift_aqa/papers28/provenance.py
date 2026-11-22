"""Capture or verify a content-addressed snapshot of the local WEST/Swift speech experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
from importlib import metadata, util
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IGNORE = {'.git', '.pytest_cache', '__pycache__', '.ruff_cache', '.mypy_cache'}
DIST_NAMES = ('ms-swift','peft','transformers','torch','accelerate','datasets','tokenizers','safetensors',
              'huggingface-hub','triton','numpy','scipy','av','soundfile','soxr','librosa','qwen-omni-utils',
              'modelscope','sentencepiece','scikit-learn','trl','deepspeed','swanlab','bitsandbytes',
              'pulp','highspy','einops','einops-exts','tensorboard','pytest','torchaudio','flash-attn')
SCOPE = ('Capture of local source bytes at the recorded time. Historical training processes and earlier '
         'checkpoints may have used other bytes. Weights, audio, manifests, predictions and training logs '
         'are kept in their separately identified experiment locations.')


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def source_tree(protocol, run):
    cfg = json.loads(Path(protocol['base_config']).read_text())
    swift = Path(cfg['swift_source']).resolve()
    if Path(util.find_spec('swift').origin).resolve().parent != swift/'swift':
        raise ValueError('The imported Swift package differs from the configured editable source tree')
    paths = {}

    def include(prefix,base,selected):
        for sub in selected:
            item=base/sub
            if not item.exists():
                raise FileNotFoundError(item)
            files=item.rglob('*') if item.is_dir() else [item]
            for f in files:
                name=f.relative_to(base)
                if (any(i in IGNORE for i in name.parts) or f.is_symlink() or not f.is_file()
                        or f.suffix in {'.pyc','.pyo','.log'}):
                    continue
                paths[f'{prefix}/{name.as_posix()}']=f
    include('west',ROOT,('experiments/ms_swift_aqa','experiments/ms_swift_librispeech','experiments/ms_swift_iemocap',
                         'examples/omni_talora','west_adapters','west','test','requirements.txt','README.md','LICENSE'))
    include('ms-swift',swift,('swift','tests','requirements','requirements.txt','setup.py','setup.cfg','MANIFEST.in',
                              'README.md','README_CN.md','LICENSE'))
    include('official-mmau',Path(protocol['official_mmau']),('evaluation.py','download.py','LICENSE','README.md'))
    include('paper01-reference',run/'theory-01',('official','transformer.diff','run_tfn.py'))
    for sub in ('mini-swift-speech/audit_clients.py','mini-swift-speech/verify_roundtrip.py','theory-01/verify_native.py'):
        if (run/sub).is_file():
            include('cpu-diagnostics',run,(sub,))
    # Both actual PEFT implementations are small enough to archive along with their package licenses.
    randpath = Path(protocol['randlora_peft'])
    std = metadata.distribution('peft')
    rand, = [d for d in metadata.distributions(path=[str(randpath)]) if d.metadata['Name'].lower()=='peft']
    for label,dist in (('peft-default',std),('peft-randlora',rand)):
        location = Path(dist.locate_file(''))
        # The experiment venv links the default package into a common site-packages directory.
        include(label,Path(dist.locate_file('peft')).resolve().parent,('peft',))
        for f in dist.files or ():
            if 'dist-info' in f.parts[0] and (f.name in {'METADATA','WHEEL','LICENSE'} or 'licenses' in f.parts):
                paths[f'{label}/{f.as_posix()}']=(location/f).resolve()
    return paths,rand.version


def environment(rand_version):
    import torch
    versions={}
    for name in DIST_NAMES:
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name]=None
    return dict(python=platform.python_version(),python_executable=sys.executable,system=platform.platform(),
                torch_compiled_cuda=torch.version.cuda,key_distributions=versions,
                randlora=dict(inherits_default_except=['peft'],peft=rand_version),
                installed_distributions=sorted({(d.metadata['Name'],d.version,str(d.locate_file(''))) for d in
                                              metadata.distributions() if d.metadata['Name']}))


def verify_snapshot(folder):
    folder=Path(folder)
    manifest_raw=(folder/'manifest.json').read_bytes()
    if sha(manifest_raw)!=folder.name:
        raise ValueError('Snapshot address differs from manifest')
    manifest=json.loads(manifest_raw)
    raw_env=(folder/'environment.json').read_bytes()
    receipt=json.loads((folder/'capture.json').read_text())
    archive=folder/'sources.tar.gz'
    if sha(raw_env)!=manifest['environment_sha256'] or receipt['archive_sha256']!=file_sha(archive):
        raise ValueError('Snapshot archive or environment digest differs')
    actual={}
    with tarfile.open(archive,'r:gz') as src:
        for entry in src:
            if not entry.isfile() or entry.name in actual:
                raise ValueError('Invalid or repeated source archive member')
            raw=src.extractfile(entry).read()
            actual[entry.name]=dict(bytes=len(raw),sha256=sha(raw),executable=bool(entry.mode&0o111))
    if actual!=manifest['files']:
        raise ValueError('Restored archive inventory or source bytes differ')
    return manifest


def create_snapshot(output, paths, env):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    modes={}
    files={}
    for name,p in sorted(paths.items()):
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path)!=name or name in ('.',''):
            raise ValueError(f'Unexpected source archive name {name}')
        stat=p.stat()
        files[name]=p.read_bytes()
        after=p.stat()
        if (after.st_mtime_ns,after.st_size,after.st_ino)!=(stat.st_mtime_ns,stat.st_size,stat.st_ino):
            raise ValueError(f'Source changed during capture: {p}')
        modes[name]=bool(stat.st_mode&0o111)
    environment_raw=encoded(env)
    manifest=dict(format=1,environment_sha256=sha(environment_raw),
                  files={name:dict(sha256=sha(raw),bytes=len(raw),executable=modes[name]) for name,raw in files.items()})
    manifest_raw=encoded(manifest)
    final=output/sha(manifest_raw)
    with (output/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if final.exists():
            verify_snapshot(final)
            return final
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            stage=Path(tmp)/final.name
            stage.mkdir()
            archive=stage/'sources.tar.gz'
            with archive.open('wb') as fp, gzip.GzipFile(fileobj=fp,mode='wb',filename='',mtime=0) as gz:
                with tarfile.open(mode='w|',fileobj=gz) as tar:
                    for name,raw in files.items():
                        info=tarfile.TarInfo(name)
                        info.size=len(raw)
                        info.mode=0o755 if modes[name] else 0o644
                        tar.addfile(info,io.BytesIO(raw))
            (stage/'manifest.json').write_bytes(manifest_raw)
            (stage/'environment.json').write_bytes(environment_raw)
            (stage/'capture.json').write_bytes(encoded(dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
                                   scope=SCOPE,archive_sha256=file_sha(archive),files=len(files))))
            verify_snapshot(stage)
            stage.rename(final)
    return final


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    parser=cli.add_subparsers(dest='action',required=True)
    capture=parser.add_parser('capture')
    capture.add_argument('--run',type=Path,required=True)
    capture.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    check=parser.add_parser('verify')
    check.add_argument('snapshot',type=Path)
    args=cli.parse_args()
    if args.action=='capture':
        if not args.run.is_dir():
            cli.error('--run must refer to the existing experiment')
        files,rand=source_tree(json.loads(args.protocol.read_text()),args.run)
        print(create_snapshot(args.run/'sources',files,environment(rand)),flush=True)
    else:
        print(f'Verified {len(verify_snapshot(args.snapshot)["files"])} source files in {args.snapshot}')
