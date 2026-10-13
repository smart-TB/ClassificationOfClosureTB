"""Explicit, checksummed recovery overlay; never edits the frozen engine/runtime.

Only the two declared singleton failures use the train-loader overlay. The two
CUDA-interrupted jobs retry the original implementation. Successful old shards
remain immutable. All recovered metadata records the recovery amendment hash.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

ROOT = Path('data/reanalysis/c6_remaining_20260923')
RUN = ROOT / 'random_group_gpu/07de66f4c5992b82303f'
OPS = ROOT / 'singleton_recovery_20261001'
PATCHED = {'ft_transformer__random_undersampling__fold2',
           'tabnet__random_undersampling__fold2'}
RETRY = PATCHED | {'tabnet__none__fold2', 'tabnet__random_undersampling__fold0'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def safe_loader(loader, record=lambda value: None):
    """Keep the sampler/order/RNG; merge only a singleton final training batch."""
    from torch.utils.data import BatchSampler, DataLoader
    base = loader.batch_sampler
    if type(base) is not BatchSampler or base.batch_size < 2 or base.drop_last:
        raise ValueError('Unsupported training batch sampler; refusing implicit changes')
    count = len(base.sampler)
    if count < 2:
        raise ValueError('Training needs at least two samples')
    if count % base.batch_size != 1:
        return loader

    class MergeSingleton(BatchSampler):
        def __init__(self):
            super().__init__(base.sampler, base.batch_size, base.drop_last)

        def __iter__(self):
            previous = None
            for batch in base:
                if previous is not None:
                    if len(batch) == 1:
                        previous = previous + batch
                        continue
                    yield previous
                previous = batch
            if previous is not None:
                yield previous

        def __len__(self):
            return len(base) - 1

    # Constructor uses the same sampler and generator, consuming no random draws.
    options = {key: getattr(loader, key) for key in (
        'num_workers', 'collate_fn', 'pin_memory', 'timeout', 'worker_init_fn',
        'generator', 'persistent_workers', 'pin_memory_device')}
    if loader.num_workers:
        options.update(prefetch_factor=loader.prefetch_factor,
                       multiprocessing_context=loader.multiprocessing_context)
    result = DataLoader(loader.dataset, batch_sampler=MergeSingleton(), **options)
    record(dict(event='singleton_training_batch_merged', timestamp=now(),
                n_train=count, nominal_batch_size=base.batch_size,
                final_batch_size=base.batch_size + 1,
                old_batches=len(base), new_batches=len(result)))
    return result


def install(record):
    from pytorch_tabular.tabular_datamodule import TabularDatamodule
    original = TabularDatamodule.train_dataloader

    def train_loader(self, batch_size=None):
        return safe_loader(original(self, batch_size=batch_size), record)

    TabularDatamodule.train_dataloader = train_loader
    return original


def verify(amendment, expected_hash):
    if sha(amendment) != expected_hash:
        raise RuntimeError('Recovery amendment changed')
    data = json.loads(amendment.read_text())
    if sha(__file__) != data['overlay_sha256']:
        raise RuntimeError('Recovery overlay changed')
    run = Path(data['run'])
    if sha(run / 'run.json') != data['original_manifest_sha256']:
        raise RuntimeError('Original manifest changed')
    from pytorch_tabular.tabular_datamodule import TabularDatamodule
    if sha(inspect.getfile(TabularDatamodule)) != data['datamodule_sha256']:
        raise RuntimeError('Dependency implementation changed')
    return data, run


def worker(amendment, expected_hash, job_id):
    data, run = verify(amendment, expected_hash)
    if job_id not in data['jobs']:
        raise RuntimeError('Job outside the explicit recovery scope')
    if Path.cwd() != run / 'runtime':
        raise RuntimeError('Worker must use the immutable original runtime')
    from tb_outcomes import dl_parallel as engine
    events = []

    def record(event):
        events.append(event)
        print(json.dumps(event), flush=True)
        write(amendment.parent / (job_id + '.batch_events.json'), events)

    write(amendment.parent / (job_id + '.batch_events.json'), events)
    if job_id in data['singleton_patch_jobs']:
        install(record)
    print(json.dumps(dict(event='recovery_worker', job=job_id,
                         amendment_sha256=expected_hash,
                         singleton_overlay=job_id in data['singleton_patch_jobs'])), flush=True)
    engine.worker(run, job_id)
    meta_path = run / 'jobs' / (job_id + '.json')
    meta = json.loads(meta_path.read_text())
    meta['recovery_amendment'] = dict(path=str(amendment), sha256=expected_hash,
                                    singleton_batch_events=events)
    meta['effective_execution_id'] = run.name + '+' + expected_hash[:12]
    write(meta_path, meta)


def prepare_amendment():
    from tb_outcomes import dl_parallel as engine
    from pytorch_tabular.tabular_datamodule import TabularDatamodule
    run = RUN.resolve()
    manifest = json.loads((run / 'run.json').read_text())
    if engine.source_signature() != manifest['sources'] or engine.runtime_signature() != manifest['packages']:
        raise RuntimeError('Scientific runtime mismatch')
    missing = {j['id'] for j in manifest['jobs'] if not engine.completed(run, j)}
    if missing != RETRY:
        raise RuntimeError('Expected exactly the four declared failed jobs: ' + str(missing))
    if sha(run / 'inputs.pkl') != manifest['input_sha256']:
        raise RuntimeError('Original inputs changed')
    # Verify the frozen copy as well as the current engine, before any mutation.
    for path, digest in manifest['sources'].items():
        if sha(run / 'runtime' / path) != digest:
            raise RuntimeError('Frozen source mismatch: ' + path)
    if OPS.exists():
        raise RuntimeError('Recovery directory already exists; do not overwrite amendment')
    OPS.mkdir()
    backups = OPS / 'before'; backups.mkdir()
    for path in [run / 'progress.json', ROOT / 'primary_gpu_status.json']:
        shutil.copy2(path, backups / path.name)
    for job_id in sorted(RETRY):
        for path in [*(run / 'jobs').glob(job_id + '.*'), run / 'logs' / (job_id + '.log')]:
            if path.is_file(): shutil.copy2(path, backups / path.name)
    preserved = {}
    for job in manifest['jobs']:
        if job['id'] not in RETRY:
            for path in (run / 'jobs').glob(job['id'] + '.*'):
                if path.is_file(): preserved[str(path.relative_to(run))] = sha(path)
    data = dict(created_at=now(), run=str(run), jobs=sorted(RETRY),
                singleton_patch_jobs=sorted(PATCHED),
                original_manifest_sha256=sha(run / 'run.json'),
                original_inputs_sha256=sha(run / 'inputs.pkl'),
                overlay_sha256=sha(__file__),
                datamodule_sha256=sha(inspect.getfile(TabularDatamodule)),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                preserved_artifact_hashes=preserved,
                policy='Only merge singleton last TRAIN batch into preceding batch (1024+1). No records dropped or duplicated. Same sampler/order/seed. Validation, calibration, score and outer partitions unchanged. Training dynamics necessarily change for this previously invalid batch; not a claim of numerical equivalence.',
                cuda_retries='Original engine unchanged for two interrupted TabNet jobs; no driver-level fix claimed.',
                reporting='Additive amendment required in C7/C8 and manuscript. Original run manifest stays intact; recovered job metadata carries amendment hash.')
    write(OPS / 'amendment.json', data)
    print('AMENDMENT', str((OPS / 'amendment.json').resolve()), sha(OPS / 'amendment.json'), flush=True)


def coordinator(amendment, expected_hash, continue_campaign):
    from tb_outcomes import dl_parallel as engine
    from tb_outcomes.gpu_fans import GPUFanLease
    from run_remaining_campaign import register
    data, run = verify(amendment, expected_hash)
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != data['boot_id']:
        raise RuntimeError('Host rebooted; inspect before recovery')
    with (ROOT / 'primary_gpu.lock').open('a') as lane, (run / '.lock').open('a') as run_lock:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((run / 'run.json').read_text())
        if engine.source_signature() != manifest['sources'] or engine.runtime_signature() != manifest['packages']:
            raise RuntimeError('Scientific engine changed')
        for path, digest in data['preserved_artifact_hashes'].items():
            if sha(run / path) != digest: raise RuntimeError('Preserved artifact changed: ' + path)
        jobs = manifest['jobs']
        missing = {j['id'] for j in jobs if not engine.completed(run, j)}
        if not missing <= RETRY: raise RuntimeError('Unexpected incomplete jobs')
        state = dict(node='primary', compute='gpu', state='running', pid=os.getpid(),
                     completed_arms=['k27', 'k75'], total_arms=4, arm='random_group',
                     run=str(run.relative_to(Path.cwd())), jobs=len(jobs),
                     recovery_amendment=str(amendment), recovery_sha256=expected_hash)

        def audited_completed(path, job):
            if not engine.completed(path, job): return False
            if job['id'] not in RETRY: return True
            meta = json.loads((path / 'jobs' / (job['id'] + '.json')).read_text())
            return meta.get('recovery_amendment', {}).get('sha256') == expected_hash

        def launch(command, **kwargs):
            if command[:3] != [sys.executable, '-m', 'tb_outcomes.dl_parallel']:
                raise RuntimeError('Unexpected worker command')
            job_id = command[command.index('--job') + 1]
            if job_id not in RETRY: raise RuntimeError('Attempt to rerun a preserved job')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--amendment', str(amendment), '--sha256', expected_hash, '--job', job_id]
            return subprocess.Popen(command, **kwargs)

        # Reuse the frozen scheduler byte-for-byte with explicit launch/completion
        # dependencies. Never modify the shared subprocess module or frozen files.
        context = dict(engine.schedule.__globals__)
        context.update(subprocess=types.SimpleNamespace(Popen=launch,
                       STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
                       completed=audited_completed)
        schedule = types.FunctionType(engine.schedule.__code__, context)
        register('primary', 'gpu', Path(state['run']), state)
        try:
            with GPUFanLease(run / 'fan_events.jsonl', display=':1',
                             xauthority='/run/user/1000/gdm/Xauthority', expected_gpu_count=2):
                schedule(run, jobs, ['0', '1'], manifest['threads'])
            if not all(audited_completed(run, j) for j in jobs):
                raise RuntimeError('Recovery output verification failed')
            for path, digest in data['preserved_artifact_hashes'].items():
                if sha(run / path) != digest: raise RuntimeError('Preserved artifact changed: ' + path)
            engine.merge(run, jobs)
            merged = json.loads((run / 'complete.json').read_text())
            merged.update(recovery_amendment=str(amendment), recovery_sha256=expected_hash,
                          effective_execution_id=run.name + '+' + expected_hash[:12])
            write(run / 'complete.json', merged)
            write(amendment.parent / 'completion.json', dict(timestamp=now(),
                  amendment_sha256=expected_hash, preserved_verified=True, recovered_jobs=sorted(RETRY)))
            state.update(state='arm_completed', completed_arms=['k27', 'k75', 'random_group'])
            register('primary', 'gpu', Path(state['run']), state)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc))
            register('primary', 'gpu', Path(state['run']), state)
            raise
    if continue_campaign:
        resume = Path('scripts/resume_gpu_after_outage.py').resolve()
        expected = '30d820f68494f45dde788262081d16f1f599bec10ea09e651655ce3188a3d7d5'
        if sha(resume) != expected: raise RuntimeError('Original continuation controller changed')
        os.execv(sys.executable, [sys.executable, str(resume), '--node', 'primary',
                                 '--expected-boot-id', data['boot_id']])


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--amendment', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--job')
    parser.add_argument('--continue-campaign', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare_amendment()
    elif args.worker: worker(args.amendment.resolve(), args.sha256, args.job)
    else: coordinator(args.amendment.resolve(), args.sha256, args.continue_campaign)


if __name__ == '__main__': main()
