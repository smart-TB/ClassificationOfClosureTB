"""Execute declared correction arms with frozen inputs and bounded node lanes."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import sys

from tb_outcomes.dl_parallel import atomic_json, prepare, schedule, merge, source_signature, runtime_signature
from tb_outcomes.gpu_fans import GPUFanLease

ROOT = Path('data/reanalysis/c6_remaining_20260923')


def now():
    return datetime.now(timezone.utc).isoformat()


def declarations():
    from tb_outcomes.executor import load_executor_config
    from tb_outcomes.dl_config import load_dl_config
    from tb_outcomes.models import load_models_config, build_registry, DL_MODELS
    cfg = load_executor_config('configs/executor.yaml')
    registry = build_registry(load_models_config('configs/models.yaml'), load_dl_config('configs/deep_learning.yaml'))
    contenders = list(cfg.sweep.models)
    definitions = [
        ('clinical_baseline', ['logistic_regression'], {'scope': 'clinical_baseline'}, 'primary', None),
        ('individual', list(cfg.ablation.models), {'scope': 'individual'}, 'primary', None),
        ('municipal', list(cfg.ablation.models), {'scope': 'municipal'}, 'primary', None),
        ('k27', contenders, {'k': 27}, 'secondary', 'primary'),
        ('k75', contenders, {'k': 75}, 'secondary', 'primary'),
        ('contemporary', contenders, {'context_profile': 'contemporary_sensitivity'}, 'primary', 'secondary'),
        ('without_diagnostics', contenders, {'availability_profile': 'without_diagnostic_tests'}, 'primary', 'secondary'),
        ('hiv_uncertain', contenders, {'availability_profile': 'hiv_uncertain'}, 'secondary', 'secondary'),
        ('all_cause', contenders, {'outcome_schema': 'all_cause_3class'}, 'secondary', 'secondary'),
        ('random_group', contenders, {'fold_method': 'random_group'}, 'primary', 'primary'),
    ]
    arms = []
    for name, models, options, cpu_node, gpu_node in definitions:
        pairs, unsupported = [], []
        for model in models:
            for strategy in ['none', *cfg.strategies]:
                item = {'model': model, 'strategy': strategy}
                if strategy in registry[model].capabilities.supported_imbalance:
                    pairs.append(item)
                else:
                    unsupported.append({**item, 'reason': 'adapter_incompatible'})
        arms.append(dict(name=name, options=options, models=models, pairs=pairs,
                         unsupported=unsupported, jobs=len(pairs)*cfg.n_outer_folds,
                         cpu_models=[m for m in models if m not in DL_MODELS],
                         gpu_models=[m for m in models if m in DL_MODELS],
                         cpu_node=cpu_node, gpu_node=gpu_node))
    return dict(created_at=now(), arms=arms, sources=source_signature(), packages=runtime_signature(),
                protocol='nested', max_configs=3, outer_folds=5, inner_folds=5,
                interpretation='Exploratory robustness within the existing configured contender families; no universal ranking claim.',
                model_universe_origin='Existing executor.yaml sweep and ablation sets; sensitivity analyses use the pre-existing sweep set. No new choice from corrected external metrics.',
                maturation_sensitivity='9/15 months and 365 days/15 months have identical eligible development and 2024 records per C1 audit; verify identity, no redundant refit.',
                reporting_status='not_released')


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Campaign interrupted: {signum}')


def register(node, compute, path, state):
    lane = ROOT / f'{node}_{compute}_status.json'
    atomic_json(lane, {**state, 'updated_at': now()})
    if node == 'primary':
        status_path = Path('docs/manuscrito/colaboracao/codex/status.json')
        with (ROOT / 'dashboard.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = json.loads(status_path.read_text())
            runs = []
            for other in ROOT.glob('primary_*_status.json'):
                s = json.loads(other.read_text())
                if s.get('run'):
                    runs.append(s['run'])
            status['active_run'] = dict(kind='Remaining correction arms, declared 2026-09-23', run_paths=runs,
                                      declaration=str(ROOT / 'declaration.json'))
            status['updated_at'] = now()
            status['current_task'] = 'C6: bracos restantes em execucao; C7: auditoria e inferencia. Manifesto nao liberado.'
            status['validation']['remaining_campaign'] = str(ROOT / 'declaration.json')
            atomic_json(status_path, status)
    else:
        monitor = Path('docs/node_setup/monitor_runs.json')
        with (ROOT / 'dashboard.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            items = json.loads(monitor.read_text()) if monitor.exists() else []
            if path and not any(x['root'] == str(path.parent) for x in items):
                items.append({'root': str(path.parent), 'label': f"{state.get('arm')} {compute.upper()}"})
                atomic_json(monitor, items)


def run_lane(node, compute):
    declaration = json.loads((ROOT / 'declaration.json').read_text())
    if source_signature() != declaration['sources'] or runtime_signature() != declaration['packages']:
        raise RuntimeError('Campaign sources or packages changed; do not silently fork experiments')
    arms = [a for a in declaration['arms'] if a[f'{compute}_node'] == node and a[f'{compute}_models']]
    status = {'node': node, 'compute': compute, 'state': 'preparing', 'completed_arms': [],
              'total_arms': len(arms), 'pid': __import__('os').getpid()}
    with (ROOT / f'{node}_{compute}.lock').open('a') as lane_lock:
        fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, interrupted)
        for arm in arms:
            status.update(arm=arm['name'], state='preparing', run=None)
            register(node, compute, None, status)
            if source_signature() != declaration['sources']:
                raise RuntimeError('Sources changed between arms')
            args = argparse.Namespace(fold_method='binpack', outcome_schema='sensitive_tb_3class',
                dl_config=Path('configs/deep_learning.yaml'), sanity=False, compute=compute,
                models=','.join(arm[f'{compute}_models']), strategies=None, protocol='nested',
                k=50, scope='combined', context_profile='baseline', availability_profile='primary',
                mode='spatial', out_root=ROOT / (arm['name']+'_'+compute), max_configs=3, threads=4)
            vars(args).update(arm['options'])
            run, jobs, threads, lock = prepare(args)
            status.update(state='running', run=str(run.relative_to(Path.cwd())), jobs=len(jobs))
            register(node, compute, Path(status['run']), status)
            print(json.dumps({'event': 'arm_started', **status}), flush=True)
            try:
                from tb_outcomes.dl_parallel import completed
                if compute == 'cpu':
                    schedule(run, jobs, ['cpu0', 'cpu1'] if node == 'secondary' else ['cpu0', 'cpu1', 'cpu2', 'cpu3'], threads)
                elif any(not completed(run, j) for j in jobs):
                    with GPUFanLease(run / 'fan_events.jsonl', display=':1',
                        xauthority='/run/user/1000/gdm/Xauthority', expected_gpu_count=1 if node == 'secondary' else 2):
                        schedule(run, jobs, ['0'] if node == 'secondary' else ['0', '1'], threads)
                merge(run, jobs)
            except BaseException as exc:
                status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                register(node, compute, Path(status['run']), status)
                raise
            finally:
                lock.close()
            status['completed_arms'].append(arm['name'])
            status['state'] = 'arm_completed'
            register(node, compute, Path(status['run']), status)
            print(json.dumps({'event': 'arm_completed', **status}), flush=True)
        status['state'] = 'completed'
        register(node, compute, Path(status['run']), status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--node', choices=['primary', 'secondary'])
    parser.add_argument('--compute', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.declare:
        doc = declarations()
        with (ROOT / 'declaration.json').open('x') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(json.dumps({'arms': len(doc['arms']), 'jobs': sum(a['jobs'] for a in doc['arms'])}))
    elif args.node and args.compute:
        run_lane(args.node, args.compute)
    else:
        parser.error('Choose --declare or --node and --compute')
