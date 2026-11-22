"""Extract utterance WAVs and consensus labels, then create Swift audio chat manifests."""
import hashlib
import json
import re
import shutil
import sys
import wave
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

IEMOCAP_EMOTIONS = {
    "ang": "angry", "hap": "happy", "exc": "excited", "sad": "sad", "neu": "neutral",
}


def _iemocap(root, sessions, skipped):
    for session in sessions:
        base = root / f"Session{session}"
        paths = sorted((base / "dialog" / "EmoEvaluation").glob("*.txt"))
        if not paths:
            raise ValueError(f"{base}: missing dialog/EmoEvaluation/*.txt")
        for path in paths:
            # Consensus rows begin with a time range; other lines are comments
            # or individual annotators' votes. Only ASCII IDs/labels are used.
            with path.open(encoding="latin-1") as f:
                for n, line in enumerate(f, 1):
                    line = line.lstrip()
                    if not line.startswith("["):
                        continue
                    _, delim, rest = line.partition("]")
                    fields = rest.split()
                    if not delim or len(fields) < 2:
                        raise ValueError(f"{path}:{n}: malformed consensus annotation")
                    utt, label = fields[:2]
                    if label not in IEMOCAP_EMOTIONS:
                        skipped[label] += 1
                        continue
                    turn = re.fullmatch(re.escape(path.stem) + r"_([FM])\d+", utt)
                    if not turn or not utt.startswith(f"Ses{session:02}"):
                        raise ValueError(f"{path}:{n}: invalid utterance id {utt!r} for this session/dialog")
                    wav = base / "sentences" / "wav" / path.stem / f"{utt}.wav"
                    # The first letter in Ses01F names the dialogue; the FINAL
                    # F/M utterance suffix identifies who actually speaks.
                    yield {"id": utt, "audio": wav, "emotion": IEMOCAP_EMOTIONS[label],
                           "session": session, "speaker": f"Ses{session:02}{turn[1]}"}


def prepare(cfg, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'data'
        stage.mkdir()
        with ZipFile(cfg['archive']) as archive:
            for name in archive.namelist():
                if re.fullmatch(r'IEMOCAP_full_release/Session[1-5]/(?:sentences/wav/[\w-]+/[\w-]+\.wav|dialog/EmoEvaluation/[\w-]+\.txt)',name):
                    archive.extract(name,stage)
        ids, speakers, summary = set(), set(), {}
        for split,sessions in cfg['sessions'].items():
            excluded, counts, talkers = Counter(), Counter(), set()
            path=stage/(split+'.jsonl')
            with path.open('x') as stream:
                for row in _iemocap(stage/'IEMOCAP_full_release', sessions, excluded):
                    uid, wav = row['id'], row['audio']
                    if uid in ids:
                        raise ValueError(uid)
                    ids.add(uid)
                    talkers.add(row['speaker'])
                    with wave.open(str(wav)) as audio:
                        assert audio.getnchannels()==1 and audio.getframerate()==16000 and 0<audio.getnframes()<=4800000,uid
                    label=cfg['labels'][row['emotion']]
                    counts[label]+=1
                    item=dict(id=uid,speaker=row['speaker'],session=row['session'],audios=[str(output/wav.relative_to(stage))],
                              messages=[dict(role='user',content='<audio>'+cfg['prompt']),dict(role='assistant',content=label)])
                    print(json.dumps(item),file=stream)
            if set(counts)!=set('ABCD') or speakers & talkers:
                raise ValueError(f'{split}: class or speaker split mismatch')
            speakers.update(talkers)
            summary[split]=dict(counts=counts,utterances=sum(counts.values()),speakers=sorted(talkers),excluded=excluded,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (stage/'splits.json').write_text(json.dumps(summary,indent=2))
        stage.rename(output)
if __name__=='__main__':
    prepare(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]).resolve())
