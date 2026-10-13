"""Exercise real process adoption using only disposable synthetic jobs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def controller():
    spec = importlib.util.spec_from_file_location('gb_controller', Path('scripts/resume_with_gb_optimization.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_takeover_keeps_existing_worker_alive(tmp_path):
    c = controller()
    script = '''
import fcntl,json,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[1])
with (run/'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 worker=subprocess.Popen([sys.executable,'-c',"import sys,time;from pathlib import Path;time.sleep(2);Path(sys.argv[1]).write_text('preserved')",str(run/'done'),'--worker',str(run),'--job','fixture'],start_new_session=True)
 (run/'ready').write_text(str(worker.pid))
 worker.wait()
 time.sleep(20)
'''
    parent = subprocess.Popen([sys.executable,'-c',script,str(tmp_path),'tb_outcomes.dl_parallel','--compute','cpu'])
    try:
        for _ in range(100):
            if (tmp_path/'ready').exists():
                break
            time.sleep(.02)
        assert (tmp_path/'ready').exists()
        token = c.process(parent.pid)['token']
        with pytest.raises(RuntimeError,match='identity'):
            c.takeover(parent.pid,'wrong-token',tmp_path,{'fixture'},tmp_path/'state.json')
        assert parent.poll() is None
        adopted = c.takeover(parent.pid,token,tmp_path,{'fixture'},tmp_path/'state.json')
        assert len(adopted) == 1 and adopted[0]['job'] == 'fixture'
        assert c.alive(adopted[0])
        parent.wait(timeout=2)
        assert parent.returncode == -9
        for _ in range(150):
            if (tmp_path/'done').exists():
                break
            time.sleep(.02)
        assert (tmp_path/'done').read_text() == 'preserved'
        assert json.loads((tmp_path/'state.json').read_text())['active'][0]['pid'] == adopted[0]['pid']
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_start_tokens_prevent_pid_reuse():
    c = controller()
    rec = c.process(os.getpid())
    assert c.alive(rec)
    assert not c.alive({**rec,'token':'different-start-time'})
