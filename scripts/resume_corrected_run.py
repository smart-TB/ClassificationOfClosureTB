"""Resume a correction run using its frozen code, configuration and input bundle."""
import argparse
import hashlib
import json
import os
import inspect
import subprocess
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--display', default=':1')
    parser.add_argument('--xauthority', default='/run/user/1000/gdm/Xauthority')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'run.json').read_text())
    identity = {k: v for k, v in manifest.items() if k != 'input_sha256'}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    if run.name != expected or not manifest.get('runtime_snapshot'):
        raise ValueError('Manifest identity invalid or immutable runtime absent')
    runtime = run / 'runtime'
    sys.path.insert(0, str(runtime / 'src'))
    os.chdir(runtime)
    from tb_outcomes.dl_parallel import completed, file_hash, merge, runtime_signature, schedule, source_signature
    from tb_outcomes.gpu_fans import GPUFanLease

    if source_signature() != manifest['sources'] or runtime_signature() != manifest['packages']:
        raise ValueError('Frozen sources or installed dependencies differ')
    if file_hash(run / 'inputs.pkl') != manifest['input_sha256']:
        raise ValueError('Input bundle changed')
    jobs = manifest['jobs']
    done = sum(completed(run, j) for j in jobs)
    print(json.dumps({'run': str(run), 'verified_jobs': done, 'total': len(jobs)}), flush=True)
    if args.check_only:
        return
    import fcntl

    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest['compute'] == 'cpu':
            if args.cpu_workers < 1:
                raise ValueError('Positive CPU worker count required')
            schedule(run, jobs, [f'cpu{i}' for i in range(args.cpu_workers)], manifest['threads'])
        elif done != len(jobs):
            gpus = args.gpus.split(',')
            if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
                raise ValueError('Distinct physical GPU indices required')
            available = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).split()
            if not set(gpus) <= set(available):
                raise ValueError(f'GPU indices unavailable: {gpus}; available: {available}')
            fan_options = dict(display=args.display, xauthority=args.xauthority)
            # Old runs retain their frozen two-GPU lease; never rewrite a snapshot.
            if 'expected_gpu_count' in inspect.signature(GPUFanLease).parameters:
                fan_options['expected_gpu_count'] = len(available)
            elif len(available) != 2:
                raise ValueError('Frozen run requires its original two-GPU fan controller')
            with GPUFanLease(run / 'fan_events.jsonl', **fan_options):
                schedule(run, jobs, gpus, manifest['threads'])
        merge(run, jobs)


if __name__ == '__main__':
    main()
