"""Save the public review thread and the PDF of a few OpenReview submissions. Standard library only, guest access.

    python research/tools/openreview_reviews.py --output iclr2027-threads v4bOJQnRfX QeGW2SC1I1 DDylDVKez5 v9o0A0YbXt hKNOcBcEpB

Writes <id>.md (abstract, every review and reply in order) and <id>.pdf into the output folder.
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://api2.openreview.net'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'peft-survey/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def value(field):
    return field.get('value') if isinstance(field, dict) else field


def thread(forum):
    notes = json.loads(fetch(f'{API}/notes?' + urllib.parse.urlencode(dict(forum=forum, limit=1000))))['notes']
    notes.sort(key=lambda n: (n['id'] != forum, n.get('cdate', 0)))
    lines = []
    for note in notes:
        kind = next((i.rsplit('/-/', 1)[-1] for i in note.get('invitations', []) if '/-/' in i), 'Note')
        who = ', '.join(s.rsplit('/', 1)[-1] for s in note.get('signatures', []))
        lines += [f'## {"Submission" if note["id"] == forum else kind} ({who})', '']
        for key, field in note.get('content', {}).items():
            text = value(field)
            if key in ('pdf', 'paperhash', '_bibtex', 'supplementary_material') or text in (None, '', []):
                continue
            lines += [f'**{key}**: {text if not isinstance(text, list) else "; ".join(map(str, text))}', '']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('forums', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for forum in args.forums:
        (args.output / f'{forum}.md').write_text(thread(forum))
        try:
            (args.output / f'{forum}.pdf').write_bytes(fetch(f'https://openreview.net/pdf?id={forum}'))
        except Exception as error:               # the thread is still useful without the PDF
            print(f'{forum}: PDF not saved ({error})')
        print(f'{forum}: saved')
