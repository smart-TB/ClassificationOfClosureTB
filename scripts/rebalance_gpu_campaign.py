"""Operational handoff at arm boundaries; never signal an existing worker.

The source's not-yet-created output directory is reserved by an exclusive,
persistent routing file. The old launcher stops at mkdir before any fitting in
that arm. A successor waits for its lane lock and verifies the preceding arm.
The destination waits for its original lane to finish before taking the arm.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path('data/reanalysis/c6_remaining_20260923')
OPS = ROOT / 'routing_gpu_20260924'
TARGET = 'without_diagnostics'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def event(node, state, **fields):
    value = dict(timestamp=datetime.now(timezone.utc).isoformat(), node=node,
                 state=state, pid=os.getpid(), **fields)
    write_json(OPS / (node + '_handoff_status.json'), value)
    with (OPS / (node + '_events.jsonl')).open('a') as f:
        f.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def route(declaration, node):
    original = {n: [a['name'] for a in declaration['arms']
                    if a['gpu_node'] == n and a['gpu_models']]
                for n in ['primary', 'secondary']}
    if original != {'primary': ['k27', 'k75', 'random_group'],
                    'secondary': ['contemporary', TARGET, 'hiv_uncertain', 'all_cause']}:
        raise ValueError('Unexpected original routing')
    assigned = {'primary': original['primary'] + [TARGET],
                'secondary': [a for a in original['secondary'] if a != TARGET]}
    flat = assigned['primary'] + assigned['secondary']
    if len(set(flat)) != len(flat) or set(flat) != set(sum(original.values(), [])):
        raise ValueError('Routing drops or duplicates an arm')
    return assigned[node]


def arguments(arm, out_root):
    args = argparse.Namespace(fold_method='binpack', outcome_schema='sensitive_tb_3class',
        dl_config=Path('configs/deep_learning.yaml'), sanity=False, compute='gpu',
        models=','.join(arm['gpu_models']), strategies=None, protocol='nested',
        k=50, scope='combined', context_profile='baseline', availability_profile='primary',
        mode='spatial', out_root=out_root, max_configs=3, threads=4)
    vars(args).update(arm['options'])
    return args


def load_plan(path, expected):
    if digest(path) != expected:
        raise ValueError('Operational amendment changed')
    plan = json.loads(path.read_text())
    if digest(ROOT / 'declaration.json') != plan['declaration_sha256']:
        raise ValueError('Original scientific declaration changed')
    if digest(__file__) != plan['controller_sha256']:
        raise ValueError('Handoff controller changed')
    declaration = json.loads((ROOT / 'declaration.json').read_text())
    for node in ['primary', 'secondary']:
        if route(declaration, node) != plan['assigned'][node]:
            raise ValueError('Invalid assignment')
    return plan, declaration


def check_engine(declaration, plan):
    from tb_outcomes.dl_parallel import source_signature, runtime_signature, file_hash
    if source_signature() != declaration['sources']:
        raise ValueError('Scientific sources changed')
    if runtime_signature() != declaration['packages']:
        raise ValueError('Package versions changed')
    if file_hash('data/harmonized.parquet') != plan['harmonized_sha256']:
        raise ValueError('Harmonized input changed')


def identity(run):
    m = json.loads((run / 'run.json').read_text())
    return {k: m[k] for k in ['experiment_id', 'input_content_sha256',
                              'harmonized_sha256', 'jobs', 'n_expected_predictions_per_pair']}


def reserve_source(path, receipt):
    # Exclusive creation is the race guard: an existing directory/run is never replaced.
    with path.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def check_gate(path, expected):
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError('Persistent source routing barrier absent or changed')


def check_completed_arm(name):
    from tb_outcomes.dl_parallel import completed
    runs = list((ROOT / (name + '_gpu')).glob('*/run.json'))
    if len(runs) != 1:
        raise ValueError('Expected one preceding run: ' + name)
    run = runs[0].parent
    m = json.loads(runs[0].read_text())
    if not (run / 'complete.json').exists() or not all(completed(run, j) for j in m['jobs']):
        raise ValueError('Preceding arm incomplete: ' + name)
    return run


def keep_ticket(stop):
    # Refresh only the ticket authenticated interactively when this tmux session starts.
    while not stop.wait(60):
        subprocess.run(['sudo', '-n', '-v'], capture_output=True, timeout=15)


def execute(node, plan, declaration):
    from tb_outcomes.dl_parallel import prepare, completed, schedule, merge
    from tb_outcomes.gpu_fans import GPUFanLease
    from run_remaining_campaign import register
    gate = json.loads((OPS / 'source_gate_receipt.json').read_text())
    if gate['plan_sha256'] != digest(OPS / 'amendment.json'):
        raise ValueError('Source barrier belongs to a different amendment')
    if node == 'secondary':
        check_gate(ROOT / (TARGET + '_gpu'), gate)
    event(node, 'waiting_for_original_lane', assigned=plan['assigned'][node])
    with (ROOT / (node + '_gpu.lock')).open('a') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX)
        # Lock transfer, not a timer, establishes that the old coordinator has exited.
        old = json.loads((ROOT / (node + '_gpu_status.json')).read_text())
        if node == 'primary':
            if old.get('state') != 'completed':
                raise ValueError('Primary lane did not complete normally')
            prior, future = ['k27', 'k75', 'random_group'], [TARGET]
        else:
            check_gate(ROOT / (TARGET + '_gpu'), gate)
            if old.get('arm') != TARGET or old.get('state') != 'preparing':
                raise ValueError('Source lane stopped outside the reserved boundary')
            prior, future = ['contemporary'], ['hiv_uncertain', 'all_cause']
        for arm in prior:
            check_completed_arm(arm)
        check_engine(declaration, plan)
        state = dict(node=node, compute='gpu', state='preparing', pid=os.getpid(),
                     completed_arms=prior.copy(), total_arms=len(plan['assigned'][node]),
                     operational_amendment=str(OPS / 'amendment.json'))
        event(node, 'handoff_verified', completed_arms=prior)
        for name in future:
            check_engine(declaration, plan)
            if node == 'secondary':
                check_gate(ROOT / (TARGET + '_gpu'), gate)
            arm = next(a for a in declaration['arms'] if a['name'] == name)
            state.update(arm=name, run=None, state='preparing')
            register(node, 'gpu', None, state)
            run, jobs, threads, lock = prepare(arguments(arm, ROOT / (name + '_gpu')))
            try:
                if name == TARGET and identity(run) != plan['target_identity']:
                    raise ValueError('Transferred arm differs from two-node preflight')
                state.update(run=str(run.relative_to(Path.cwd())), jobs=len(jobs), state='running')
                register(node, 'gpu', Path(state['run']), state)
                if any(not completed(run, j) for j in jobs):
                    with GPUFanLease(run / 'fan_events.jsonl', display=':1',
                            xauthority='/run/user/1000/gdm/Xauthority',
                            expected_gpu_count=2 if node == 'primary' else 1):
                        schedule(run, jobs, ['0', '1'] if node == 'primary' else ['0'], threads)
                merge(run, jobs)
            except BaseException as exc:
                state.update(state='failed', error=str(exc))
                register(node, 'gpu', Path(state['run']) if state['run'] else None, state)
                raise
            finally:
                lock.close()
            state['completed_arms'].append(name)
            state['state'] = 'arm_completed'
            register(node, 'gpu', Path(state['run']), state)
            event(node, 'arm_completed', arm=name)
        state['state'] = 'completed'
        register(node, 'gpu', Path(state['run']), state)
        event(node, 'completed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=['primary', 'secondary'], required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OPS.mkdir(exist_ok=True)
    plan, declaration = load_plan(OPS / 'amendment.json', args.plan_sha256)
    check_engine(declaration, plan)
    if args.preflight:
        from tb_outcomes.dl_parallel import prepare
        arm = next(a for a in declaration['arms'] if a['name'] == TARGET)
        run, jobs, threads, lock = prepare(arguments(arm, OPS / ('preflight_' + args.node)))
        try:
            write_json(OPS / ('preflight_' + args.node + '.json'), identity(run))
            print(json.dumps(identity(run)), flush=True)
        finally:
            lock.close()
        return
    stop = threading.Event()
    thread = threading.Thread(target=keep_ticket, args=(stop,), daemon=True)
    thread.start()
    try:
        execute(args.node, plan, declaration)
    except BaseException as exc:
        event(args.node, 'attention_required', error=repr(exc))
        raise
    finally:
        stop.set()
        thread.join(timeout=20)


if __name__ == '__main__':
    main()
