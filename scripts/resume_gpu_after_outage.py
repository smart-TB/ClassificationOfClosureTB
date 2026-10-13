"""Resume the amended GPU assignment after a verified host reboot.

Uses the frozen scientific engine and its integrity-aware job resumption.
The original routing barrier and immutable amendment remain in force.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil

from rebalance_gpu_campaign import (
    ROOT, OPS, TARGET, arguments, check_engine, check_gate, identity, load_plan,
)
from run_remaining_campaign import register
from tb_outcomes.dl_parallel import completed, prepare, schedule, merge
from tb_outcomes.gpu_fans import GPUFanLease

PLAN_HASH = '4228865aa3159a256d42cbccbe2ef2b71cabf4103c1a01fbe2b7a4076e95d12c'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=['primary', 'secondary'], required=True)
    parser.add_argument('--expected-boot-id', required=True)
    args = parser.parse_args()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != args.expected_boot_id:
        raise RuntimeError('Boot identity changed; inspect host before resuming')
    plan, declaration = load_plan(OPS / 'amendment.json', PLAN_HASH)
    check_engine(declaration, plan)
    gate = json.loads((OPS / 'source_gate_receipt.json').read_text())
    if gate['plan_sha256'] != PLAN_HASH:
        raise RuntimeError('Routing receipt mismatch')
    if args.node == 'secondary':
        check_gate(ROOT / (TARGET + '_gpu'), gate)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    evidence = ROOT / 'outage_recovery_20260928' / (args.node + '_gpu_' + stamp)
    with (ROOT / (args.node + '_gpu.lock')).open('a') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Kernel locks vanish at reboot; verify recorded old workers are absent.
        for name in plan['assigned'][args.node]:
            for progress in (ROOT / (name + '_gpu')).glob('*/progress.json'):
                old = json.loads(progress.read_text())
                for item in old.get('active', []):
                    proc = Path('/proc') / str(item['pid']) / 'cmdline'
                    if proc.exists() and b'tb_outcomes.dl_parallel' in proc.read_bytes():
                        raise RuntimeError('An old training worker is still running')
        evidence.mkdir(parents=True, exist_ok=False)
        old_state = ROOT / (args.node + '_gpu_status.json')
        if old_state.exists():
            shutil.copy2(old_state, evidence / 'previous_lane_status.json')
        (evidence / 'recovery.json').write_text(json.dumps(dict(
            boot_id=boot, node=args.node, pid=os.getpid(), plan_sha256=PLAN_HASH,
            assigned=plan['assigned'][args.node], created_at=stamp), indent=2))
        state = dict(node=args.node, compute='gpu', state='preparing',
                     pid=os.getpid(), completed_arms=[],
                     total_arms=len(plan['assigned'][args.node]),
                     operational_amendment=str(OPS / 'amendment.json'),
                     outage_recovery=str(evidence))
        for name in plan['assigned'][args.node]:
            check_engine(declaration, plan)
            if args.node == 'secondary':
                check_gate(ROOT / (TARGET + '_gpu'), gate)
            arm = next(a for a in declaration['arms'] if a['name'] == name)
            existing = list((ROOT / (name + '_gpu')).glob('*/run.json'))
            if len(existing) > 1:
                raise RuntimeError('Multiple existing runs require reconciliation')
            state.update(arm=name, run=None, state='preparing')
            register(args.node, 'gpu', None, state)
            run, jobs, threads, lock = prepare(arguments(arm, ROOT / (name + '_gpu')))
            try:
                if existing and run != existing[0].parent.resolve():
                    raise RuntimeError('Recovery would fork the existing run')
                if name == TARGET and identity(run) != plan['target_identity']:
                    raise RuntimeError('Transferred arm identity mismatch')
                pending = [j for j in jobs if not completed(run, j)]
                state.update(run=str(run.relative_to(Path.cwd())),
                             jobs=len(jobs), state='running')
                register(args.node, 'gpu', Path(state['run']), state)
                print(json.dumps(dict(arm=name, preserved=len(jobs)-len(pending),
                                      pending=len(pending))), flush=True)
                if pending:
                    archive = evidence / name
                    archive.mkdir()
                    if (run / 'progress.json').exists():
                        shutil.copy2(run / 'progress.json', archive / 'previous_progress.json')
                    for job in pending:
                        for file in [*(run / 'jobs').glob(job['id'] + '.*'),
                                     run / 'logs' / (job['id'] + '.log')]:
                            if file.is_file():
                                shutil.copy2(file, archive / file.name)
                    with GPUFanLease(run / 'fan_events.jsonl', display=':1',
                            xauthority='/run/user/1000/gdm/Xauthority',
                            expected_gpu_count=2 if args.node == 'primary' else 1):
                        schedule(run, jobs, ['0', '1'] if args.node == 'primary' else ['0'], threads)
                if pending or not (run / 'complete.json').exists():
                    merge(run, jobs)
            except BaseException as exc:
                state.update(state='failed', error=repr(exc))
                register(args.node, 'gpu', Path(state['run']) if state['run'] else None, state)
                raise
            finally:
                lock.close()
            state['completed_arms'].append(name)
            state['state'] = 'arm_completed'
            register(args.node, 'gpu', Path(state['run']), state)
        state['state'] = 'completed'
        register(args.node, 'gpu', Path(state['run']), state)


if __name__ == '__main__':
    main()
