"""Finish an existing transfer run, including LoRAGen's source-expert dependency."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from .report import current_replay, digest, evidence, frozen_evidence, publish, replace, theory_evidence


HERE = Path(__file__).resolve().parent
ARCHIVED = {'lora', 'vera', 'dora', 'fourierft'}


def completed(out, method):
    return evidence(out, method)[0]['status'] == 'verified'


def write_disposition(out, errors=()):
    verified = {}
    def target(method, source, paper=None):
        state, record = evidence(out, method)
        state['source'] = source
        if paper is not None:
            state['paper'] = paper
        if record is not None:
            verified[method] = record
        return state

    catalog = json.loads((HERE/'catalog.json').read_text())
    papers = [target(p['method'], 'archived_fit_current_verification' if p['method'] in ARCHIVED else
                     'generated_from_avqa_experts' if p['method']=='loragen' else 'current_source_fit', p['id'])
              for p in catalog if p['id'] != 1]
    theory = theory_evidence(out)
    teacher = target('known-target-qo','same_source_trained_teacher_diagnostic',1)
    frozen, _ = frozen_evidence(out)
    refs = [target('lora','archived_fit_current_verification'),target('lora-recompute','current_source_fit'),frozen]
    groups = {'paper_methods':papers,'theory':[theory],'teacher_diagnostic':[teacher],'references':refs}
    counts = {name:{'expected':len(group),'verified':sum(x['status']=='verified' for x in group)}
              for name,group in groups.items()}
    replay_ok = frozen['status'] == 'verified' and current_replay(out,verified)
    ok = not errors and replay_ok and all(g['expected']==g['verified'] for g in counts.values())
    disposition = dict(status='complete' if ok else 'failed', finished_utc=datetime.now(timezone.utc).isoformat(),
                       run=str(out), protocol_sha256=digest(out/'protocol.json') if (out/'protocol.json').is_file() else None,
                       paper_methods=papers,theory=theory,teacher_diagnostic=teacher,references=refs,counts=counts,
                       paired_replay={'status':'verified' if replay_ok else 'missing_or_invalid',
                                      'evidence':str(out/'paired-target-scores.json')},
                       errors=list(map(str,errors)))
    replace(out/'disposition.json',json.dumps(disposition,indent=2,ensure_ascii=False)+'\n')
    return disposition


@contextmanager
def record_failure(out):
    try:
        yield
    except Exception as exc:
        write_disposition(out, [f'Queue terminated: {exc!r}'])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worker-pids', nargs='*', type=int, default=[])
    parser.add_argument('--theory-cache', type=Path,
                        default=Path('/root/autodl-tmp/r1-aqa/paper-envs/expressive'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    worker_ids = {}
    scheduled = {}
    for pid in args.worker_pids:
        stat = Path(f'/proc/{pid}/stat')
        if stat.is_file():
            worker_ids[pid] = stat.read_text().split(') ', 1)[1].split()[19]
            argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'experiments.ms_swift_aqa.papers28.run' in argv:
                for method in ('known-target-qo','loragen'):
                    if method.encode() in argv:
                        scheduled[method] = pid

    def active(pid):
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        except FileNotFoundError:
            return False
        return fields[0] != 'Z' and fields[19] == worker_ids[pid]

    def command(method):
        cmd = [str(HERE/'run.sh'), '--out', str(out)]
        if method in ARCHIVED:
            cmd.append('--archived')
        return [*cmd, method]

    def sources_ready():
        folder = out/'train/lora-mixer'
        return all((folder/file).is_file() for file in ('initialization.json','source-experts/library.json'))

    priorities = ['lora', 'lora-recompute', *(p['method'] for p in json.loads((HERE/'catalog.json').read_text()) if p['id'] > 1)]
    # Generation consumes the trained source experts. It never acquires the GPU while waiting for them.
    priorities.remove('loragen')
    priorities.append('loragen')
    with (out/'report-queue.lock').open('a') as lock, ExitStack() as guard:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.enter_context(record_failure(out))
        protocol = out/'protocol.json'
        if not protocol.is_file():
            replace(protocol, (HERE/'protocol.json').read_text())
        if theory_evidence(out)['status'] != 'verified':
            subprocess.run([sys.executable,'-m','experiments.ms_swift_aqa.papers28.expressive',
                            '--output',str(out/'theory-01'),'--cached-sources',str(args.theory_cache/'official'),
                            '--extra-dependencies',str(args.theory_cache/'deps')],check=False)
        waiting = {'loragen':sources_ready,
                   'known-target-qo':lambda: (out/'theory-01/omni-control/construction.json').is_file()}
        launched = {}
        stamp = None
        scores_stamp = None
        while True:
            scores = sorted((out/'results').glob('*.json'))
            paths = [*out.glob('status-*.json'), *scores]
            current_scores = tuple((p.name,p.stat().st_mtime_ns) for p in scores if completed(out,p.stem))
            current = current_scores, tuple(sorted((p.name, p.stat().st_mtime_ns) for p in paths))
            try:
                if current_scores != scores_stamp and (current_scores or (out/'paired-target-scores.json').is_file()):
                    from .paired import publish as publish_paired
                    publish_paired(out)
                    scores_stamp = current_scores
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print('Could not verify paired targets', repr(exc), flush=True)
            try:
                if current != stamp:
                    publish(out)
                    stamp = current
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print('Could not verify reporting snapshot', repr(exc), flush=True)
            for method,ready in waiting.items():
                if (method not in launched and not completed(out, method) and ready()
                        and (method not in scheduled or not active(scheduled[method]))):
                    launched[method] = subprocess.Popen(command(method))
            if not any(active(pid) for pid in worker_ids) and all(job.poll() is not None for job in launched.values()):
                break
            time.sleep(30)
        for method in priorities:
            if completed(out,method) or (method == 'loragen' and not sources_ready()):
                continue
            result = subprocess.run(command(method), check=False)
            print(method, 'driver exit', result.returncode, flush=True)
        if not completed(out,'known-target-qo'):
            if not waiting['known-target-qo']():
                subprocess.run([sys.executable,'-m','experiments.ms_swift_aqa.papers28.known_target',
                                '--protocol',str(protocol),'--output',str(out/'theory-01/omni-control')],check=False)
            subprocess.run(command('known-target-qo'), check=False)
        errors = []
        try:
            from .paired import publish as publish_paired
            publish_paired(out)
        except Exception as exc:
            errors.append(f'Independent paired replay failed: {exc!r}')
        try:
            publish(out)
        except Exception as exc:
            errors.append(f'Report publication failed: {exc!r}')
        disposition = write_disposition(out,errors)
        print('Queue',disposition['status'],out/'disposition.json',flush=True)
        return int(disposition['status'] != 'complete')


if __name__ == '__main__':
    sys.exit(main())
