"""Audited CPU queue transition: preserve frozen sources and running workers.

The addon changes execution only for explicitly listed future GB jobs. Existing
worker processes continue uninterrupted. Resume this controller after failures;
never run two coordinators for the same run.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def process(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, token=fields[19], state=fields[0], ppid=int(fields[1]),
                    cmd=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    p = process(record['pid'])
    return p is not None and p['token'] == record['token'] and p['state'] != 'Z'


def validated_children(parent, run, allowed):
    children = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        rec = process(int(p.name))
        if not rec or rec['ppid'] != parent or rec['state'] == 'Z':
            continue
        cmd = rec['cmd']
        if '--worker' not in cmd or '--job' not in cmd:
            raise RuntimeError(f'Unexpected child of coordinator: {rec}')
        if Path(cmd[cmd.index('--worker')+1]).resolve() != run:
            raise RuntimeError('Coordinator child belongs to a different run')
        job = cmd[cmd.index('--job')+1]
        if job not in allowed:
            raise RuntimeError('Unknown active job')
        rec.update(job=job, optimized=False)
        children.append(rec)
    if len(children) > 4 or len({c['job'] for c in children}) != len(children):
        raise RuntimeError('Invalid active CPU queue')
    return children


def takeover(pid, token, run, allowed, state_path):
    """Stop dispatch only, record live workers, then retire ONLY their parent."""
    p = process(pid)
    if not p or p['token'] != token or '--compute' not in p['cmd']:
        raise RuntimeError('Coordinator identity changed')
    if p['cmd'][p['cmd'].index('--compute')+1] != 'cpu':
        raise RuntimeError('Only CPU coordinator may be replaced')
    if 'tb_outcomes.dl_parallel' not in p['cmd']:
        raise RuntimeError('Unexpected coordinator executable')
    locks = []
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            locks.append(fd.resolve())
        except OSError:
            pass
    if run/'.lock' not in locks:
        raise RuntimeError('Coordinator does not own the requested run lock')
    retired = False
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(50):
            current = process(pid)
            if current and current['token'] == token and current['state'] in ('T', 't'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Coordinator did not stop dispatch')
        children = validated_children(pid, run, allowed)
        atomic(state_path, dict(state='transition_prepared', previous_coordinator=p,
            active=children, failed=[], updated_at=time.time()))
        # SIGTERM invokes the old coordinator's finally block and kills workers.
        # SIGKILL ONLY the verified dispatcher leaves its separate worker sessions alive.
        os.kill(pid, signal.SIGKILL)
        retired = True
        return children
    finally:
        if not retired and alive(p):
            os.kill(pid, signal.SIGCONT)


def verify(run, addon):
    approval = json.loads((addon/'amendment.json').read_text())
    if Path(approval['run']).resolve() != run or sha(run/'run.json') != approval['run_manifest_sha256']:
        raise RuntimeError('Execution amendment belongs to different run')
    for name, expected in approval['files'].items():
        if Path(name).name != name or sha(addon/name) != expected:
            raise RuntimeError(f'Addon integrity failure: {name}')
    manifest = json.loads((run/'run.json').read_text())
    if manifest['compute'] != 'cpu' or manifest['protocol'] != 'nested':
        raise RuntimeError('Nested CPU run required')
    identity = {k:v for k,v in manifest.items() if k != 'input_sha256'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] != run.name:
        raise RuntimeError('Run identity mismatch')
    os.chdir(run/'runtime')
    sys.path.insert(0, str(run/'runtime/src'))
    sys.path.insert(0, str(addon))
    from tb_outcomes.dl_parallel import source_signature, runtime_signature
    if source_signature() != manifest['sources'] or runtime_signature() != manifest['packages']:
        raise RuntimeError('Frozen scientific sources or packages changed')
    if sha(run/'inputs.pkl') != manifest['input_sha256']:
        raise RuntimeError('Input bundle changed')
    if approval['workers_per_gb'] != 4 or approval['cpu_slots'] != 4:
        raise RuntimeError('Unvalidated CPU budget')
    allowed = {j['id'] for j in manifest['jobs'] if j['model'] == 'gradient_boosting'}
    if not set(approval['target_jobs']) <= allowed:
        raise RuntimeError('Amendment targets non-GB jobs')
    return manifest, approval


def optimized_worker(run, addon, job_id):
    manifest, approval = verify(run, addon)
    if job_id not in approval['target_jobs']:
        raise RuntimeError('Job not authorized by execution amendment')
    import gb_parallel
    from tb_outcomes import corrected_worker as cw
    from tb_outcomes import dl_parallel as dl
    from tb_outcomes.models import load_models_config
    from tb_outcomes.preprocess import load_preprocess_config
    from tb_outcomes.imbalance import load_imbalance_config
    original_evaluate = cw.evaluate_nested_job
    original_fit = cw.fit_selected_pipeline

    def evaluate(bundle, meta, job, dev, evaluation, cfg, destination):
        config = load_models_config('configs/models.yaml')
        context = {k:bundle[k].iloc[dev].reset_index(drop=True) for k in ('X','y','strata')}
        context.update(entries={**config.baselines, **config.models}, specs=bundle['specs'],
            pre=load_preprocess_config('configs/preprocess.yaml'),
            imb=load_imbalance_config('configs/imbalance.yaml'),
            seed=bundle['seed'], n_classes=len(bundle['classes']))
        def accelerated(candidates, y, groups, predict, final, cal_cfg, **kwargs):
            return gb_parallel.fit_gradient_boosting(context, groups, bundle['X'].iloc[evaluation],
                candidates, cal_cfg, workers=approval['workers_per_gb'], **kwargs,
                event_callback=lambda event: print(json.dumps({**event,'time':time.time()}), flush=True))
        cw.fit_selected_pipeline = accelerated
        try:
            return original_evaluate(bundle, meta, job, dev, evaluation, cfg, destination)
        finally:
            cw.fit_selected_pipeline = original_fit
    cw.evaluate_nested_job = evaluate
    dl.worker(run, job_id)
    path = run/'jobs'/(job_id+'.json')
    result = json.loads(path.read_text())
    result['execution_amendment'] = dict(path=str(addon/'amendment.json'),
        sha256=sha(addon/'amendment.json'), workers=4, reuse_stages=True)
    atomic(path, result)


def coordinate(run, addon, args):
    manifest, approval = verify(run, addon)
    from tb_outcomes.dl_parallel import completed, gpu_environment, merge
    jobs = {j['id']:j for j in manifest['jobs']}
    targets = set(approval['target_jobs'])
    state_path = addon/'controller_state.json'
    def done(job_id):
        if not completed(run, jobs[job_id]):
            return False
        if job_id in targets:
            meta = json.loads((run/'jobs'/(job_id+'.json')).read_text())
            if meta.get('execution_amendment',{}).get('sha256') != sha(addon/'amendment.json'):
                raise RuntimeError(f'Missing execution provenance: {job_id}')
        return True
    if args.check_only:
        print(json.dumps(dict(verified=True, completed=sum(done(j) for j in jobs),
                             total=len(jobs), targets=len(targets))), flush=True)
        return
    # Additional transition lock prevents two would-be successors freezing one parent.
    with (addon/'.controller.lock').open('a') as transition_lock:
        fcntl.flock(transition_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.takeover_pid:
            if state_path.exists():
                raise RuntimeError('Transition already recorded; resume without takeover')
            active = takeover(args.takeover_pid, args.parent_token, run, jobs, state_path)
            atomic(addon/'takeover.json', json.loads(state_path.read_text()))
        else:
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            active = previous.get('active', [])
        with (run/'.lock').open('a') as lock:
            for attempt in range(50):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if attempt == 49:
                        raise
                    time.sleep(.1)
            successes = {j for j in jobs if done(j)}
            active = [a for a in active if a['job'] not in successes or alive(a)]
            active_ids = {a['job'] for a in active}
            pending = [j for j in jobs if j not in successes and j not in active_ids]
            # Engineering priority: start one optimized GB when a CPU slot opens.
            pending.sort(key=lambda j: 0 if j in targets else 1)
            failures = []
            owned = {}
            draining = False
            def stop_dispatch(signum, frame):
                nonlocal draining
                draining = True
            signal.signal(signal.SIGTERM, stop_dispatch)
            signal.signal(signal.SIGINT, stop_dispatch)
            def progress(state):
                atomic(state_path, dict(state=state, active=active, failed=failures,
                    pending=pending, coordinator_pid=os.getpid(), updated_at=time.time()))
                atomic(run/'progress.json', dict(state=state, updated_at=time.time(),
                    total=len(jobs), completed=len(successes), failed=failures,
                    pending=len(pending), active=[dict(worker=a['slot'], job=a['job'], pid=a['pid'],
                        optimized=a['optimized']) for a in active],
                    execution_amendment=str(addon/'amendment.json')))
            for i,a in enumerate(active):
                a.setdefault('slot', f'cpu{i}')
            progress('running')
            print(json.dumps(dict(event='coordinator_started',adopted=active,targets=len(targets))),flush=True)
            try:
                while pending or active:
                    for a in list(active):
                        if a['pid'] in owned:
                            proc, log = owned[a['pid']]
                            if proc.poll() is None:
                                continue
                            log.close()
                            del owned[a['pid']]
                        elif alive(a):
                            continue
                        if done(a['job']):
                            successes.add(a['job'])
                        else:
                            failures.append(a['job'])
                            if a['optimized']:
                                draining = True
                        active.remove(a)
                        progress('draining' if draining else 'running')
                    if draining and not active:
                        break
                    for slot in range(4):
                        name = f'cpu{slot}'
                        if draining or any(a['slot'] == name for a in active):
                            continue
                        memory = {line.split(':')[0]:int(line.split()[1]) for line in
                                  Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                        gb_allowed = not any(a['optimized'] for a in active) and memory['MemAvailable'] >= 24*1024**2
                        choice = next((j for j in pending if j not in targets or gb_allowed), None)
                        if choice is None:
                            continue
                        pending.remove(choice)
                        optimized = choice in targets
                        # Preserve the original parent/calibration thread budget;
                        # gb_parallel restricts each fitting child to one thread.
                        env = gpu_environment(name, manifest['threads'])
                        if optimized:
                            env['LOKY_MAX_CPU_COUNT'] = '4'
                        env['PYTHONPATH'] = str(addon) + os.pathsep + str(run/'runtime/src')
                        command = ([sys.executable, str(addon/'controller.py'), '--run', str(run),
                                    '--addon', str(addon), '--worker', choice] if optimized else
                                   [sys.executable, '-m', 'tb_outcomes.dl_parallel', '--worker',str(run),'--job',choice])
                        log = (run/'logs'/(choice+'.log')).open('w')
                        try:
                            proc = subprocess.Popen(command, cwd=run/'runtime', env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        except BaseException:
                            log.close()
                            pending.insert(0, choice)
                            raise
                        rec = process(proc.pid)
                        if rec is None:
                            proc.wait()
                            log.close()
                            failures.append(choice)
                            draining = True
                            break
                        rec.update(job=choice, slot=name, optimized=optimized)
                        active.append(rec)
                        owned[proc.pid] = proc,log
                        progress('running')
                        print(json.dumps(dict(event='dispatched',**rec)),flush=True)
                    if active or pending:
                        time.sleep(2)
                if failures or pending:
                    progress('failed' if failures else 'drained')
                    return
                if not all(done(j) for j in jobs):
                    raise RuntimeError('Incomplete audited outputs')
                progress('checking_outputs')
                merge(run, list(jobs.values()))
                atomic(state_path, dict(state='completed',active=[],failed=[],pending=[],updated_at=time.time()))
            except BaseException:
                # Preserve work. Recorded PIDs/tokens allow an audited adoption on resume.
                progress('controller_error')
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--addon', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--takeover-pid', type=int)
    p.add_argument('--parent-token')
    p.add_argument('--worker')
    args = p.parse_args()
    run, addon = args.run.resolve(), args.addon.resolve()
    addon.relative_to(run/'execution_addons')
    if args.takeover_pid and not args.parent_token:
        p.error('Exact parent start token required for takeover')
    if args.worker:
        optimized_worker(run, addon, args.worker)
    else:
        coordinate(run, addon, args)


if __name__ == '__main__':
    main()
