"""Post-test diagnostic on repeated transcripts bearing differing consensus labels."""
from collections import defaultdict,Counter
import json
import re
from zipfile import ZipFile

def same_text(archive, sessions, predictions):
    rows={r['id']:r for r in map(json.loads,predictions.read_text().splitlines())}
    groups=defaultdict(list)
    with ZipFile(archive) as z:
        for n in z.namelist():
            if not any(f'/Session{s}/dialog/transcriptions/' in n for s in sessions) or not n.endswith('.txt'):
                continue
            for line in z.read(n).decode('latin1').splitlines():
                m=re.match(r'^(\S+)\s+\[[^\]]+\]:\s*(.+)$',line)
                if not m or m[1] not in rows:
                    continue
                text=m[2].lower()
                if '[' in text or '(' in text:  # Exclude transcripts with annotated nonverbal events.
                    continue
                words=re.findall(r"[a-z0-9]+(?:'[a-z]+)*",text)
                if words:
                    groups[' '.join(words)].append(rows[m[1]])
    eligible=[rr for rr in groups.values() if len({r['reference'] for r in rr})>1]
    subset=[r for group in eligible for r in group]
    return dict(unique_texts=len(eligible),utterances=len(subset),support=dict(Counter(r['reference'] for r in subset)),
                accuracy=sum(r['reference']==r['prediction'] for r in subset)/len(subset) if subset else None)
