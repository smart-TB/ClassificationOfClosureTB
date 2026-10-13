"""One guarded retry after the active GPU coordinator drains; no model changes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path('data/reanalysis/c6_remaining_20260923')
OPS = ROOT / 'cuda_check_20261002'
RUN = ROOT / 'without_diagnostics_gpu/a3baa54af6f89b3ae40e'
JOB = 'tabnet__random_undersampling__fold1'
CONTROLLER_HASH = '30d820f68494f45dde788262081d16f1f599bec10ea09e651655ce3188a3d7d5'


def eligible(state, progress, missing):
    return (state.get('state') == 'failed' and state.get('pid') == 493915
            and state.get('arm') == 'without_diagnostics'
            and state.get('run') == str(RUN)
            and progress.get('state') == 'failed'
            and progress.get('failed') == [JOB]
            and not progress.get('active') and progress.get('pending') == 0
            and missing == {JOB})


def record(state, **extra):
    p = OPS / 'retry_status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state, pid=os.getpid(), time=time.time(), **extra),indent=2)+'\n')
    tmp.replace(p)


def refresh(stop):
    while not stop.wait(60):
        subprocess.run(['sudo','-n','-v'],capture_output=True,timeout=15)


def main():
    from tb_outcomes.dl_parallel import completed
    receipt = json.loads((OPS/'recovery_receipt.json').read_text())
    boot = receipt['boot_id']
    stop = threading.Event()
    threading.Thread(target=refresh,args=(stop,),daemon=True).start()
    try:
        with (OPS/'retry.lock').open('a') as singleton, (ROOT/'primary_gpu.lock').open('a') as lane:
            fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record('waiting_for_current_coordinator',job=JOB)
            deadline=time.monotonic()+72*3600
            while True:
                if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:
                    raise RuntimeError('Boot changed')
                try:
                    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise RuntimeError('Wait deadline expired')
                    time.sleep(30)
            state=json.loads((ROOT/'primary_gpu_status.json').read_text())
            progress=json.loads((RUN/'progress.json').read_text())
            manifest=json.loads((RUN/'run.json').read_text())
            missing={j['id'] for j in manifest['jobs'] if not completed(RUN,j)}
            if not eligible(state,progress,missing):
                raise RuntimeError('Unexpected remaining failures/state; explicit review required')
            controller=Path('scripts/resume_gpu_after_outage.py').resolve()
            if hashlib.sha256(controller.read_bytes()).hexdigest()!=CONTROLLER_HASH:
                raise RuntimeError('Continuation controller changed')
            if Path('/proc/493915').exists():raise RuntimeError('Old coordinator still present')
            record('launching_original_controller',job=JOB,preserved=29)
        stop.set()
        os.execv(sys.executable,[sys.executable,str(controller),'--node','primary','--expected-boot-id',boot])
    except BaseException as exc:
        record('stopped_for_review',error=repr(exc))
        raise
    finally:stop.set()


if __name__=='__main__':main()
