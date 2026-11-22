"""Independently replay both target scores and compare verified adapters to frozen and ordinary LoRA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from experiments.ms_swift_aqa.summarize_pilot import load_official


TAG = {'ang':'A', 'hap':'B', 'exc':'B', 'sad':'C', 'neu':'D'}
FROZEN_PREDICTIONS = {
    'ser':'b72c04f5f75ff36d30bf99adbd26f6c1d8d66743920df5ee7c726420a3c7d529',
    'mmau':'c92fdc09191d714779d631a895edaf158922127c71945cb29496085fbb863c19',
}
PCM_INDEX = '40a30f11d833be067fedddeb4935018f208c6abf11722195e1708da6e44b8ddc'
MMAU_ANNOTATIONS = '9f18fda99f8dbc2bc5ecd6323fb5063309f54969810b5c6f1caeb3b8d904cf1c'


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def source_ser(manifest):
    items = read_rows(manifest)
    source = manifest.parent/'IEMOCAP_full_release/Session5/dialog/EmoEvaluation'
    refs = {}
    for path in sorted(source.glob('*.txt')):
        for line in path.read_text(encoding='latin-1').splitlines():
            if line.lstrip().startswith('['):
                uid, tag, *_ = line.split(']',maxsplit=1)[1].split()
                if tag in TAG:
                    check(uid not in refs and uid.rsplit('_',1)[0] == path.stem, f'Invalid source turn {uid}')
                    refs[uid] = (TAG[tag],path.stem)
    check(len(items) == len(refs) == len({r['id'] for r in items}) == 1241, 'Incomplete IEMOCAP Session 5')
    gold,group = [],[]
    for r in items:
        label, dialogue = refs[r['id']]
        check(r['session'] == 5 and r['messages'][1]['content'] == label, f'Invalid Session 5 label for {r["id"]}')
        gold.append('ABCD'.index(label))
        group.append(dialogue)
    check(len(set(group)) == 31 and np.bincount(gold).tolist() == [170,442,245,384], 'Wrong Session 5 composition')
    return items,np.asarray(gold),group


def paired(hits, groups, classes, seed, resamples):
    """Sample complete clusters together for every arm and reference in a dataset."""
    names = sorted(hits)
    values = np.column_stack([hits[n] for n in names])
    _, cluster = np.unique(groups,return_inverse=True)
    count = len(set(cluster))
    labels, codes = np.unique(classes, return_inverse=True)
    totals = np.zeros((count,len(labels)))
    net = np.zeros((count,len(labels),len(names)))
    np.add.at(totals,(cluster,codes),1)
    np.add.at(net,(cluster,codes),values)
    observed = (net.sum(0)/totals.sum(0)[:,None]).mean(0)
    rng = np.random.default_rng(seed)
    rates = []
    for start in range(0,resamples,200):
        n = min(200,resamples-start)
        size = rng.multinomial(count,np.full(count,1/count),size=n)
        denom = size @ totals
        keep = (denom > 0).all(1)
        numer = (size[keep] @ net.reshape(count,-1)).reshape(-1,len(labels),len(names))
        rates.append((numer/denom[keep,:,None]).mean(1))
    rates = np.concatenate(rates)
    check(len(rates) > 0, 'No bootstrap draw contained all reference classes')
    output = {}
    for baseline in ('lora','frozen-base'):
        if baseline not in names:
            continue
        j = names.index(baseline)
        for i, name in enumerate(names):
            if name == baseline or name == 'frozen-base':
                continue
            diff = 100*(rates[:,i]-rates[:,j])
            output[f'{name} vs {baseline}'] = dict(delta_pp=float(100*(observed[i]-observed[j])),
                percentile_95_pp=np.quantile(diff,[.025,.975]).tolist(),
                candidate_only_correct=int((values[:,i] & ~values[:,j]).sum()),
                reference_only_correct=int((~values[:,i] & values[:,j]).sum()))
    return dict(clusters=count, draws=resamples,valid_draws=len(rates),seed=seed,contrasts=output)


def replay_ser(items, gold, folder):
    predictions = read_rows(folder/'predictions.jsonl')
    actual = []
    for item, row in zip(items,predictions,strict=True):
        letter = row['prediction']
        check(row['id'] == item['id'] and row['reference'] == item['messages'][1]['content'] and
              isinstance(letter,str) and len(letter) == 1 and letter in 'ABCD', f'Invalid SER prediction {row["id"]}')
        actual.append('ABCD'.index(letter))
    matrix = np.bincount(4*gold + np.asarray(actual),minlength=16).reshape(4,4)
    hit = gold == actual
    recalls = matrix.diagonal()/matrix.sum(1)
    return (dict(uar=float(recalls.mean()), accuracy=float(hit.mean()), recalls=dict(zip('ABCD',recalls)),
                 confusion_matrix=matrix.tolist(), utterances=len(hit)), hit)


def replay_mmau(test, source, official, folder):
    predictions = read_rows(folder/'predictions.jsonl')
    hits, exact = [],[]
    by_domain = Counter()
    for item, gold, row in zip(test,source,predictions,strict=True):
        key = row['prediction']
        check(isinstance(key,str) and len(key) == 1 and 0 <= ord(key)-65 < len(gold['choices']),
              f'Invalid MMAU choice for {item["id"]}')
        choice = gold['choices'][ord(key)-65]
        correct = choice == gold['answer']
        check(row['id'] == gold['id'] == item['id'] and item['choices'] == gold['choices'] and
              item['question'] == gold['question'] and row['reference'] == item['messages'][1]['content']
              and row['correct_letters'] == item['correct_letters'] and row['answer_text'] == choice
              and row['is_correct'] is correct and row['domain'] == gold['task'], f'MMAU alignment at {item["id"]}')
        exact.append(correct)
        hit = bool(official.string_match(gold['answer'],choice,gold['choices']))
        by_domain[gold['task']] += int(hit)
        hits.append(hit)
    return dict(official_correct=sum(hits), official_accuracy=float(np.mean(hits)),
                correct_by_domain=dict(by_domain), exact_correct=sum(exact), n=len(hits)),np.asarray(hits)


def audit_record(folder, expected_manifest, expected_weights=None, frozen=None):
    met = read(folder/'metrics.json')
    actual = digest(folder/'predictions.jsonl')
    check(met['manifest_sha256'] == expected_manifest and met.get('predictions_sha256',actual) == actual,
          f'Manifest/prediction checksum mismatch at {folder}')
    if frozen is None:
        check((met.get('adapter_sha256') or met.get('identity',{}).get('weights_sha256')) == expected_weights,
              f'Evaluation used a different source-selected checkpoint at {folder}')
    else:
        check(actual == FROZEN_PREDICTIONS[frozen], f'Frozen reference predictions differ at {folder}')
    return met, actual


def check_inference_config(metrics, selection, training_dir, archived):
    """Bind prediction semantics to the saved config, including archived evaluations."""
    identity = metrics.get('identity', {})
    checksum = metrics.get('config_sha256') or identity.get('adapter_config_sha256')
    if checksum is None and archived:
        archive = training_dir.parent/'selection.json'
        if archive.is_file():
            historical = read(archive).get(selection['method'], {})
            fields = ('checkpoint','step','dev_accuracy','adapter_sha256')
            if all(historical.get(k) == selection.get(k) for k in fields):
                checksum = historical.get('config_sha256')
    check(checksum == selection['config_sha256'], f'Adapter inference configuration differs for {training_dir}')
    if 'files' in metrics:
        check(metrics['files'] == selection['files'], f'Scored sidecar hashes differ for {training_dir}')


def compare(out, seed=42, resamples=10000, frozen_ser_run=None):
    protocol = read(out/'protocol.json')
    frozen_src = Path(frozen_ser_run or protocol.get('frozen_ser_run') or out.parent/'ta-conditional-transfer-01')
    archive = Path(protocol['archive'])
    ser_path = Path(protocol['ser'])
    data = Path(protocol['avqa_mmau'])
    original = Path(protocol['official_mmau'])
    check(resamples > 0 and digest(ser_path) == protocol['hashes']['ser'] and
          digest(data/'test.jsonl') == protocol['hashes']['test'] and
          digest(original/'mmau-test-mini.json') == MMAU_ANNOTATIONS and
          digest(data/'audio-hashes.json') == PCM_INDEX, 'Invalid source manifest, PCM index or MMAU release')
    items,gold,dialogues = source_ser(ser_path)
    test,source,scorer,official = load_official(data,original)
    pcm_lookup = read(data/'audio-hashes.json')
    pcm = [pcm_lookup[r['audios'][0]] for r in test]
    check(len(set(pcm)) == 904, 'Unexpected PCM grouping')
    check(read(frozen_src/'protocol.json')['model_revision'] == protocol['model_revision'] and
          read(archive/'base.json')['generation'] == {'batch_size':2}, 'Frozen decoding protocol mismatch')
    directories = {'frozen-base':(frozen_src/'ser/frozen-base',archive/'frozen-base/test',None)}
    records = {}
    for filename in sorted((out/'results').glob('*.json')):
        r = read(filename)
        method = r['method']
        phase = read(out/f'status-{method}.json')['stage']
        if phase != 'verified':
            continue
        selected = read(out/'configs'/f'{method}-selected.json')
        check(selected == r['checkpoint'] and method == filename.stem, f'Selection drift for {method}')
        cp = Path(selected['checkpoint'])
        for file,sha in selected['files'].items():
            check(digest(cp/file) == sha, f'Changed checkpoint artifact for {method}/{file}')
        cfg = read(out/'configs'/f'{method}.json')
        check(cfg['model_revision'] == protocol['model_revision'] and cfg['generation'] == {'batch_size':2},
              f'Base model or decode config differs for {method}')
        records[method] = r
        directories[method] = (Path(r['ser_dir']),Path(r['train_dir'])/'reload-test',selected['adapter_sha256'])
    scores,ser_hits,mmau_hits = {},{},{}
    for method,(ser,mmau,weights) in directories.items():
        frozen = method == 'frozen-base'
        s,ser_sha = audit_record(ser,protocol['hashes']['ser'],weights,'ser' if frozen else None)
        m,mmau_sha = audit_record(mmau,protocol['hashes']['test'],weights,'mmau' if frozen else None)
        if frozen:
            check(s['identity']['model_revision'] == m['revision'] == protocol['model_revision'] and
                  s['tuner'] == 'frozen-base' and m['trainable_parameters'] == 0,'Frozen reference identity differs')
        sr,ser_hits[method] = replay_ser(items,gold,ser)
        mr,mmau_hits[method] = replay_mmau(test,source,official,mmau)
        check(sr['confusion_matrix'] == s['confusion_matrix'] and np.isclose(sr['uar'],s['uar']) and
              mr['exact_correct'] == m['correct'] and mr['n'] == m['utterances'], f'Raw metrics disagree for {method}')
        if method in records:
            rec = records[method]
            for metrics in (s,m):
                check_inference_config(metrics,rec['checkpoint'],Path(rec['train_dir']),rec['archived_training'])
            check(rec['ser']['confusion_matrix'] == sr['confusion_matrix'] and
                  np.isclose(rec['mmau']['official_text_accuracy'],mr['official_accuracy']),
                  f'Published scores disagree with replay for {method}')
        scores[method] = dict(ser=sr,mmau=mr,ser_predictions_sha256=ser_sha,
                              mmau_predictions_sha256=mmau_sha,selected_weights_sha256=weights)
    return dict(model_revision=protocol['model_revision'],score_protocol='frozen source-selected adapters; no target fitting',
                frozen_ser_run=str(frozen_src),frozen_mmau_run=str(archive/'frozen-base/test'),
                scorer_sha256=digest(scorer),pcm_index_sha256=PCM_INDEX,
                intervals='Paired percentile cluster bootstrap for fixed checkpoints; unadjusted across comparisons.',
                scores=scores,ser_by_dialogue_uar=paired(ser_hits,dialogues,gold,seed,resamples),
                mmau_by_pcm_official_accuracy=paired(mmau_hits,pcm,np.zeros(len(pcm),dtype=int),seed,resamples))


def publish(out, frozen_ser_run=None):
    result = compare(out,frozen_ser_run=frozen_ser_run)
    filename = out/'paired-target-scores.json'
    with NamedTemporaryFile('w',dir=out,delete=False) as tmp:
        json.dump(result,tmp,indent=2,ensure_ascii=False,allow_nan=False)
        tmp.write('\n')
    Path(tmp.name).replace(filename)
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frozen-ser-run',type=Path,help='Archived run containing protocol.json and ser/frozen-base')
    args = parser.parse_args()
    print(publish(args.out,args.frozen_ser_run))
