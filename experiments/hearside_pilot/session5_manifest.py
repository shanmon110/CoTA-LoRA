"""Build the IEMOCAP Session 5 evaluation manifest, for the confirmation of the class-marginal correction.

Session 5 has never been read by the hear-side line: `data.py` says "Session 5 is not read" and
`README.md` line 296 already names Session 5 as the split on which an exploratory result is to
be confirmed.  This module turns it into the same four-option-order manifest the Session 4
evaluation used, so the frozen base and the seven adapters can be scored on it with the
existing extraction path.

It follows `experiments/trr_diagnostics/d2.py`'s rules exactly:

  * the same selection: utterances with a transcript and with lexical content, so every
    condition is scored on the same set;
  * the same row fields: id, label, rotation, option_order, reference, dialogue, speaker,
    session, transcript;
  * the same four cyclic option orders, with `id` suffixed `#r<shift>`;
  * the same question, in the same four orderings.

The one difference is that this builds only the `orig` audio condition, with no prosody twins.
The Session 5 confirmation needs the original recordings only, and the prosody twins need
parselmouth, which the server environment does not have.  The twins are a different
experiment (the D2 line); nothing here depends on them.

Run from the repository root:

    python3 -m experiments.hearside_pilot.session5_manifest --output <folder>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.trr_diagnostics import d2

CLASSES = 'ABCD'
DEFAULT_SOURCE = Path('/root/autodl-tmp/iemocap/swift-ser/data/test.jsonl')
DEFAULT_ARCHIVE = Path('/root/autodl-tmp/iemocap/iemocapfullrelease-v1.zip')
SESSION = 5


def read_transcripts(archive, session, *, wanted=None):
    """Utterance id -> transcript line, for one session.

    `d2.read_transcripts` refuses to choose between two lines that carry the same utterance
    id.  In Session 5 two ids do: `Ses05F_impro03_FXX3` and `Ses05M_impro03_FXX0`.  Both are
    IEMOCAP placeholders -- the `FXX` marks an utterance the release does not contain -- and
    both carry two lines because the placeholder stands for two different turns.  Neither id
    is in the Session 5 manifest, so they are not utterances anyone scores.

    This reader therefore reads only the ids that are wanted, and reports every id it had to
    drop, so the drop is recorded rather than silently accepted.  `d2.read_transcripts` is left
    untouched: it is used by every other session, and the sessions it has read have no such
    conflict.
    """
    from zipfile import ZipFile

    found, dropped = {}, []
    with ZipFile(archive) as release:
        for name in release.namelist():
            if not name.endswith('.txt') or f'/Session{session}/dialog/transcriptions/' not in name:
                continue
            for line in release.read(name).decode('latin1').splitlines():
                match = d2.TRANSCRIPT_LINE.match(line)
                if not match:
                    continue
                uid, text = match[1], match[2]
                if wanted is not None and uid not in wanted:
                    continue
                if uid in found and found[uid] != text:
                    dropped.append((uid, found[uid], text))
                    continue
                found.setdefault(uid, text)
    if dropped:
        raise ValueError(f'{len(dropped)} wanted utterances have conflicting transcripts: {dropped[:3]}')
    return found


def build(source, archive, output, *, classes=CLASSES, rotations=(0, 1, 2, 3), protocol=None):
    """Write `<output>/orig.jsonl` and `<output>/splits.json`."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists; a manifest is written once')
    question = protocol['prompts']['question']
    template = protocol['prompts']['question_template']
    options = protocol['prompts']['options']
    if template.format(*(options[c] for c in classes)) != question:
        raise ValueError('question_template with the options in class order must reproduce the question')

    rows = d2.read_jsonl(source)
    for row in rows:
        session = row.get('session')
        if session != SESSION:
            raise ValueError(f"{row['id']}: session {session!r} is not Session {SESSION}")
    transcripts = read_transcripts(archive, SESSION, wanted={row['id'] for row in rows})
    items, excluded = d2.select_items(rows, transcripts, question, classes)
    if not items:
        raise ValueError('no utterance survived the selection')

    k = len(classes)
    out = []
    for item in items:
        for shift in rotations:
            order = [classes[(j + shift) % k] for j in range(k)]
            text = template.format(*(options[c] for c in order))
            letter = classes[order.index(item['reference'])]
            meta = {key: value for key, value in item.items() if key != 'source_audio'}
            out.append(dict(meta, id=f"{item['id']}#r{shift}", utterance=item['id'],
                            rotation=shift, option_order=order, label=item['reference'], reference=letter,
                            audios=[item['source_audio']],
                            messages=[dict(role='user', content='<audio>' + text),
                                      dict(role='assistant', content=letter)]))
    output.mkdir(parents=True)
    d2.write_jsonl(output / 'orig.jsonl', out)
    summary = dict(
        session=SESSION, source=dict(path=str(Path(source).resolve()), sha256=d2.digest(source)),
        archive=str(Path(archive).resolve()), rows=len(out), utterances=len(items),
        excluded=excluded, rotations=list(rotations),
        speakers=sorted({item['speaker'] for item in items}),
        dialogues=sorted({item['dialogue'] for item in items}),
        classes=dict(sorted(Counter(item['reference'] for item in items).items())),
        labels=dict(sorted(Counter(row['label'] for row in out).items())),
        same_text_different_label=sum(bool(item['same_text_different_label']) for item in items),
        manifest_sha256=d2.digest(output / 'orig.jsonl'))
    (output / 'splits.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--protocol', type=Path,
                        default=Path('experiments/hearside_pilot/pilot-iemocap-session4.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.source, args.archive, args.output,
                    protocol=json.loads(args.protocol.read_text()))
    print(json.dumps(summary, indent=2))
