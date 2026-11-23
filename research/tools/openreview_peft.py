"""List the parameter-efficient fine-tuning submissions of an OpenReview venue with their review ratings.

Standard library only. Reads public notes through the OpenReview API as a guest; nothing is posted.

    python research/tools/openreview_peft.py --venue ICLR.cc/2027/Conference --output iclr2027-peft

Writes, into the output folder:
    submissions.jsonl   every submission of the venue (id, title, keywords, abstract, venue status)
    peft.csv            the PEFT submissions with their ratings, highest mean rating first
    peft.jsonl          the same with the abstracts and the individual ratings
Running again reuses submissions.jsonl and the ratings already fetched (delete the folder to refresh).
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://api2.openreview.net'
PEFT = re.compile(r'parameter[- ]efficient|\bpeft\b|\blora\b|low[- ]rank adapt|\badapters?\b|prompt[- ]tuning|prefix[- ]tuning'
                  r'|\bdora\b|\bpissa\b|orthogonal fine-?tuning|\bia3\b|\bbitfit\b|representation fine-?tuning|\breft\b', re.I)
SPEECH = re.compile(r'speech|audio|spoken|acoustic|paralinguistic|asr\b|voice|emotion recognition', re.I)


def value(field):
    """OpenReview API 2 wraps content fields as {'value': ...}."""
    return field.get('value') if isinstance(field, dict) else field


def get(path, params, retries=5):
    url = f'{API}{path}?{urllib.parse.urlencode(params)}'
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'peft-survey/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 429 or error.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'giving up on {url}')


def submissions(venue, page=1000):
    """Every submission note of the venue, active, withdrawn and desk-rejected alike."""
    offset = 0
    while True:
        data = get('/notes', {'invitation': f'{venue}/-/Submission', 'limit': page, 'offset': offset})
        notes = data.get('notes', [])
        for note in notes:
            content = note.get('content', {})
            yield dict(id=note['id'], number=note.get('number'), title=value(content.get('title', '')) or '',
                       keywords=value(content.get('keywords', [])) or [], abstract=value(content.get('abstract', '')) or '',
                       venue=value(content.get('venue', '')) or '')
        offset += len(notes)
        print(f'  {offset} submissions read', file=sys.stderr)
        if len(notes) < page:
            return
        time.sleep(0.5)


def number(rating):
    """'6: marginally above the acceptance threshold' -> 6.0; 6 -> 6.0."""
    if isinstance(rating, (int, float)):
        return float(rating)
    match = re.match(r'\s*(-?\d+(?:\.\d+)?)', str(rating or ''))
    return float(match.group(1)) if match else None


def ratings(forum):
    """Ratings (and confidences) of the official reviews posted so far on one submission."""
    data = get('/notes', {'forum': forum, 'limit': 1000})
    out = []
    for note in data.get('notes', []):
        if not any(i.endswith('/-/Official_Review') for i in note.get('invitations', [])):
            continue
        content = note.get('content', {})
        score = number(value(content.get('rating')))
        if score is not None:
            out.append(dict(rating=score, confidence=number(value(content.get('confidence')))))
    return out


def is_peft(row):
    return bool(PEFT.search(' '.join([row['title'], ' '.join(row['keywords']), row['abstract']])))


def run(venue, output, sleep=0.3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'submissions.jsonl'
    if not cache.exists():
        print(f'reading the submissions of {venue}', file=sys.stderr)
        rows = list(submissions(venue))
        if not rows:
            raise SystemExit(f'no submissions found under {venue}/-/Submission: check the venue id')
        cache.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    chosen = [r for r in rows if is_peft(r)]
    print(f'{len(rows)} submissions, {len(chosen)} match the PEFT terms', file=sys.stderr)

    done_path = output / 'ratings.jsonl'
    done = {}
    if done_path.exists():
        done = {d['id']: d['reviews'] for d in map(json.loads, done_path.read_text().splitlines())}
    with done_path.open('a') as sink:
        for index, row in enumerate(chosen, 1):
            if row['id'] in done:
                continue
            done[row['id']] = ratings(row['id'])
            sink.write(json.dumps(dict(id=row['id'], reviews=done[row['id']])) + '\n')
            sink.flush()
            if index % 50 == 0:
                print(f'  ratings for {index}/{len(chosen)}', file=sys.stderr)
            time.sleep(sleep)

    table = []
    for row in chosen:
        reviews = done.get(row['id'], [])
        scores = [r['rating'] for r in reviews]
        status = ('withdrawn' if 'withdrawn' in row['venue'].lower() else
                  'desk rejected' if 'desk rejected' in row['venue'].lower() else 'active')
        table.append(dict(row, status=status, reviews=reviews, n_reviews=len(scores),
                          mean_rating=round(statistics.mean(scores), 2) if scores else None,
                          min_rating=min(scores) if scores else None, max_rating=max(scores) if scores else None,
                          speech_audio=bool(SPEECH.search(row['title'] + ' ' + row['abstract'])),
                          url=f'https://openreview.net/forum?id={row["id"]}'))
    table.sort(key=lambda r: (r['mean_rating'] is None, -(r['mean_rating'] or 0)))
    (output / 'peft.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in table))
    columns = ['mean_rating', 'min_rating', 'max_rating', 'n_reviews', 'status', 'speech_audio', 'title', 'keywords', 'url']
    with (output / 'peft.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in table:
            writer.writerow(dict(r, keywords='; '.join(r['keywords'])))
    rated = [r for r in table if r['mean_rating'] is not None]
    print(f'wrote {output / "peft.csv"}: {len(table)} PEFT submissions, {len(rated)} with ratings, '
          f'{sum(r["speech_audio"] for r in table)} mention speech or audio', file=sys.stderr)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--venue', default='ICLR.cc/2027/Conference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.venue, args.output)
