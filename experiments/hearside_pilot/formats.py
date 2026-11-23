"""Does what the adapters learned on the four-letter question carry over to other ways of asking?

Three formats that no arm was trained on, asked about the same Session 4 recordings and answered in free text:

yesno   "Does the speaker sound <emotion>?"  once with the true emotion, once with a wrong one
word    the four emotions offered as words (two option orders), answered with a word
open    no options at all

Subcommands: `build` writes the manifests, `run` scores one model, `summarize` compares the models.
The parsing rules, the lexicon and the reading of the result are fixed in this file and in README.md
("跨格式评测") before any model was scored.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.trr_diagnostics import d2

EMOTIONS = {'A': 'angry', 'B': 'happy or excited', 'C': 'sad', 'D': 'neutral'}
PROMPTS = dict(
    yesno='Does the speaker sound {0}? Answer yes or no.',
    word='How does the speaker sound? Choose one: {0}; {1}; {2}; {3}. Answer with the chosen words only.',
    open="Describe the speaker's emotional state in one or two words.")
WORD_ORDERS = (0, 2)
GENERATION = dict(do_sample=False, num_beams=1, batch_size=4, max_new_tokens=16)
# Whole words, after lower-casing. An answer naming no class, or more than one, is invalid and counts as wrong.
LEXICON = {'A': ('angry', 'anger', 'angered', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'frustration', 'outraged'),
           'B': ('happy', 'happiness', 'excited', 'excitement', 'joy', 'joyful', 'cheerful', 'delighted', 'amused', 'pleased', 'glad'),
           'C': ('sad', 'sadness', 'sorrow', 'sorrowful', 'depressed', 'unhappy', 'melancholy', 'melancholic', 'grief', 'mournful'),
           'D': ('neutral', 'calm', 'neutrality')}
FORMATS = ('yesno', 'word', 'open')
# Written down before any model was scored.
CRITERIA = dict(min_transfer_share=0.5, base_ceiling=0.85)
REFERENCE = 'D-all-shuffled'                       # the simple recipe a new method would have to beat
FAMILIES = {arm: [arm, f'{arm}-s43', f'{arm}-s44'] for arm in ('A-last4', 'A-all', 'B-audio', 'D-all-shuffled')}
SINGLE = ['frozen-base', 'C-post-audio']


def parse_yesno(text):
    words = re.findall(r'[a-z]+', text.lower())
    return words[0] if words and words[0] in ('yes', 'no') else None


def parse_emotion(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    named = [c for c, terms in LEXICON.items() if words & set(terms)]
    return named[0] if len(named) == 1 else None


def build(orig_manifest, output):
    """Manifests for the three formats from the evaluation rows of the trained question (one row per utterance)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in d2.read_jsonl(orig_manifest) if r['rotation'] == 0]
    classes = list(EMOTIONS)
    keep = ('speaker', 'session', 'dialogue', 'scripted', 'audios')
    out = {name: [] for name in FORMATS}
    for index, row in enumerate(rows):
        base = dict({k: row[k] for k in keep}, utterance=row['utterance'], label=row['label'])
        wrong = classes[(classes.index(row['label']) + 1 + index % 3) % 4]
        for asked, answer in ((row['label'], 'yes'), (wrong, 'no')):
            out['yesno'].append(dict(base, id=f"{row['utterance']}#{answer}", asked=asked, group=answer, messages=[
                dict(role='user', content='<audio>' + PROMPTS['yesno'].format(EMOTIONS[asked])),
                dict(role='assistant', content=answer)]))
        for shift in WORD_ORDERS:
            order = [classes[(j + shift) % 4] for j in range(4)]
            out['word'].append(dict(base, id=f"{row['utterance']}#w{shift}", group=row['label'], messages=[
                dict(role='user', content='<audio>' + PROMPTS['word'].format(*(EMOTIONS[c] for c in order))),
                dict(role='assistant', content=EMOTIONS[row['label']])]))
        out['open'].append(dict(base, id=f"{row['utterance']}#open", group=row['label'], messages=[
            dict(role='user', content='<audio>' + PROMPTS['open']), dict(role='assistant', content=EMOTIONS[row['label']])]))
    info = dict(source=str(Path(orig_manifest).resolve()), source_sha256=d2.digest(orig_manifest), utterances=len(rows),
                prompts=PROMPTS, generation=GENERATION, lexicon=LEXICON, manifests={})
    for name, content in out.items():
        d2.write_jsonl(output / f'{name}.jsonl', content)
        info['manifests'][name] = dict(sha256=d2.digest(output / f'{name}.jsonl'), rows=len(content))
    d2.write(output / 'formats.json', info)
    return info


def correct(name, row, text):
    """(answered correctly, answer was parseable) for one generated answer."""
    answer = parse_yesno(text) if name == 'yesno' else parse_emotion(text)
    return answer == (row['group']), answer is not None


def run(folder, model_config, method, checkpoint, arm):
    """Generate the answers of one model for every format; finished formats are kept."""
    from experiments.ms_swift_librispeech.evaluate import score

    folder = Path(folder)
    info = d2.load(folder / 'formats.json')
    out = folder / 'runs' / arm
    pending = [n for n in FORMATS if not (out / n / 'metrics.json').exists()]
    if not pending:
        return
    if method == 'frozen-base':
        from experiments.ms_swift_aqa.evaluate_frozen import load_frozen
        model, template = load_frozen(d2.load(model_config))
    else:
        from experiments.ms_swift_librispeech.train import load_best
        model, template = load_best(d2.load(model_config), method, Path(checkpoint))
        model.requires_grad_(False)
    for name in pending:
        manifest = folder / f'{name}.jsonl'
        if d2.digest(manifest) != info['manifests'][name]['sha256']:
            raise ValueError(f'{name}: manifest changed after it was built')
        score(model, template, manifest, GENERATION, out / name / 'predictions.jsonl')
        d2.write(out / name / 'metrics.json', dict(format=name, arm=arm, checkpoint=str(checkpoint) if checkpoint else None,
                                                   manifest_sha256=info['manifests'][name]['sha256'],
                                                   predictions_sha256=d2.digest(out / name / 'predictions.jsonl')))
        print(f'{name}: done', flush=True)


class Scores:
    """Mean per-group recall of one format (groups: yes/no, or the four classes), with shared dialogue resamples."""

    def __init__(self, manifest, name, resamples, rng):
        self.manifest, self.name = manifest, name
        groups = sorted({r['group'] for r in manifest})
        names = sorted({r['dialogue'] for r in manifest})
        self.group = np.array([groups.index(r['group']) for r in manifest])
        self.dialogue = np.array([names.index(r['dialogue']) for r in manifest])
        self.shape = (len(names), len(groups))
        draws = rng.integers(0, len(names), size=(resamples, len(names)))
        self.weights = np.stack([np.bincount(row, minlength=len(names)) for row in draws]).astype(float)
        self.total = self.count(np.ones(len(manifest)))
        self.valid = (self.weights @ self.total > 0).all(axis=1)

    def count(self, values):
        out = np.zeros(self.shape)
        np.add.at(out, (self.dialogue, self.group), values)
        return out

    def measure(self, rows):
        if [r['id'] for r in rows] != [m['id'] for m in self.manifest]:
            raise ValueError(f'{self.name}: predictions do not cover the manifest in order')
        judged = [correct(self.name, m, r['prediction']) for m, r in zip(self.manifest, rows)]
        hits = self.count(np.array([j[0] for j in judged], float))
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = ((self.weights @ hits) / (self.weights @ self.total)).mean(axis=1)
        return dict(score=float((hits.sum(0) / self.total.sum(0)).mean()), parseable=float(np.mean([j[1] for j in judged])),
                    recalls=[float(v) for v in hits.sum(0) / self.total.sum(0)]), np.where(self.valid, boot, np.nan)


def interval(boot):
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return [100 * float(low), 100 * float(high)]


def summarize(folder, mcq_gain_pp, *, resamples=10000, seed=42):
    """`mcq_gain_pp`: REFERENCE minus frozen base on the trained question, all option orders (from the follow-up summary)."""
    folder = Path(folder)
    info = d2.load(folder / 'formats.json')
    rng = np.random.default_rng(seed)
    arms = sorted(p.name for p in (folder / 'runs').iterdir() if all((p / n / 'metrics.json').exists() for n in FORMATS))
    value, boot = {a: {} for a in arms}, {a: {} for a in arms}
    for name in FORMATS:
        scores = Scores(d2.read_jsonl(folder / f'{name}.jsonl'), name, resamples, rng)
        for arm in arms:
            path = folder / 'runs' / arm / name
            if d2.digest(path / 'predictions.jsonl') != d2.load(path / 'metrics.json')['predictions_sha256']:
                raise ValueError(f'{arm}/{name}: predictions changed after they were written')
            value[arm][name], boot[arm][name] = scores.measure(d2.read_jsonl(path / 'predictions.jsonl'))
    draws = {}

    def mean_boot(members, name):
        if len(members) not in draws:
            draws[len(members)] = rng.integers(0, len(members), size=(resamples, len(members)))
        stacked = np.stack([boot[m][name] for m in members])
        return stacked[draws[len(members)].T, np.arange(resamples)].mean(axis=0)

    groups = {a: [a] for a in SINGLE if a in arms}
    groups.update({f: [m for m in members if m in arms] for f, members in FAMILIES.items() if any(m in arms for m in members)})
    table = {g: {n: dict(per_model={m: value[m][n]['score'] for m in members},
                         mean=float(np.mean([value[m][n]['score'] for m in members])),
                         parseable=float(np.mean([value[m][n]['parseable'] for m in members]))) for n in FORMATS}
             for g, members in groups.items()}
    contrasts = {}
    if 'frozen-base' in groups:
        for g, members in groups.items():
            if g == 'frozen-base':
                continue
            contrasts[f'{g} - frozen-base'] = {
                n: dict(delta_pp=100 * (table[g][n]['mean'] - table['frozen-base'][n]['mean']),
                        interval_95_pp=interval(mean_boot(members, n) - boot['frozen-base'][n])) for n in FORMATS}
    for a, b in (('B-audio', 'A-all'), ('D-all-shuffled', 'A-all'), ('B-audio', 'D-all-shuffled'), ('A-last4', 'A-all')):
        common = [i for i in range(3) if FAMILIES[a][i] in arms and FAMILIES[b][i] in arms]
        if common:
            contrasts[f'{a} - {b}'] = {
                n: dict(delta_pp=100 * float(np.mean([value[FAMILIES[a][i]][n]['score'] - value[FAMILIES[b][i]][n]['score'] for i in common])),
                        interval_95_pp=interval(mean_boot([FAMILIES[a][i] for i in common], n) - mean_boot([FAMILIES[b][i] for i in common], n)))
                for n in FORMATS}
    return dict(folder=str(folder), utterances=info['utterances'], arms=arms, criteria=CRITERIA, mcq_gain_pp=mcq_gain_pp,
                table=table, contrasts=contrasts, readings=judge(table, contrasts, mcq_gain_pp))


def judge(table, contrasts, mcq_gain_pp):
    """Per format: does the gain of the reference recipe on the trained question carry over?"""
    key = f'{REFERENCE} - frozen-base'
    if key not in contrasts or mcq_gain_pp is None:
        return {n: None for n in FORMATS}
    out = {}
    for n in FORMATS:
        gain, (low, high) = contrasts[key][n]['delta_pp'], contrasts[key][n]['interval_95_pp']
        if table['frozen-base'][n]['mean'] >= CRITERIA['base_ceiling']:
            out[n] = 'base at ceiling'
        elif high < 0:
            out[n] = 'harmed'
        elif gain < CRITERIA['min_transfer_share'] * mcq_gain_pp:
            out[n] = 'gap'
        else:
            out[n] = 'transfers'
    return out


def render(result):
    lines = ['# Answers in formats no arm was trained on', '',
             f"Session 4, {result['utterances']} utterances. Score: mean per-group recall in % (yes/no: the two answers; "
             'word and open: the four emotions); an unparseable answer counts as wrong. In brackets: share of parseable answers.', '',
             '| model (mean over its seeds) | ' + ' | '.join(FORMATS) + ' |', '|---|' + '---:|' * len(FORMATS)]
    for g, row in result['table'].items():
        seeds = len(next(iter(row.values()))['per_model'])
        lines.append(f'| {g} ({seeds}) | ' + ' | '.join(f"{100 * row[n]['mean']:.1f} [{100 * row[n]['parseable']:.0f}%]" for n in FORMATS) + ' |')
    lines += ['', '| difference pp [95% CI] | ' + ' | '.join(FORMATS) + ' |', '|---|' + '---|' * len(FORMATS)]
    for key, by in result['contrasts'].items():
        lines.append(f'| {key} | ' + ' | '.join(f"{by[n]['delta_pp']:+.1f} [{by[n]['interval_95_pp'][0]:+.1f}, {by[n]['interval_95_pp'][1]:+.1f}]"
                                               for n in FORMATS) + ' |')
    gain = result['mcq_gain_pp']
    lines += ['', f"Gain of {REFERENCE} over the frozen base on the trained question: "
              + ('not available' if gain is None else f'{gain:+.1f} pp') + '. Reading declared before scoring:', '']
    lines += [f"- {n}: **{'not available' if v is None else v}**" for n, v in result['readings'].items()]
    return '\n'.join(lines) + '\n'


def mcq_gain(followup_json):
    """REFERENCE (mean over its seeds) minus the frozen base, all option orders, in pp, from followup.json."""
    data = d2.load(followup_json)
    family = data['families'].get(REFERENCE)
    reference = family['mean']['uar_all_orders'] if family else data['models'][REFERENCE]['uar_all_orders']
    return 100 * (reference - data['models']['frozen-base']['uar_all_orders'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build')
    p.add_argument('--manifest', type=Path, required=True, help='eval/manifests/orig.jsonl of the pilot')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('run')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--method', choices=['frozen-base', 'ta-lora'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--arm', required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--followup', type=Path, help='followup.json holding the scores on the trained question')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build(args.manifest, args.output)['manifests'], indent=2))
    elif args.command == 'run':
        run(args.folder, args.model_config, args.method, args.checkpoint, args.arm)
    else:
        summary = summarize(args.folder, mcq_gain(args.followup) if args.followup else None)
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'formats.json', summary)
        (args.output / 'formats.md').write_text(render(summary))
        print(render(summary))
