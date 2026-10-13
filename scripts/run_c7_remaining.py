"""CPU-only C7 audits of the completed declared sensitivity/ablation arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('data/reanalysis/c7_20261004')
TRAIN = Path('data/reanalysis/c6_remaining_20260923')
ASSIGNED = {'primary': ['individual','municipal','without_diagnostics','random_group',
                         'k27','k75','contemporary'],
            'secondary': ['hiv_uncertain','all_cause']}


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node',required=True,choices=ASSIGNED)
    args=parser.parse_args()
    from tb_outcomes.dl_parallel import file_hash
    declaration=json.loads((TRAIN/'declaration.json').read_text())
    snapshot=ROOT/'analysis_code'; code=json.loads((ROOT/'analysis_code_manifest.json').read_text())
    for name,digest in code.items():
        if file_hash(snapshot/name)!=digest:raise RuntimeError('C7 code changed: '+name)
    ROOT.mkdir(exist_ok=True)
    with (ROOT/(args.node+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(node=args.node,state='running',pid=os.getpid(),started_at=now(),arms={})
        status=ROOT/(args.node+'_status.json')
        def audit(name):
            arm=next(a for a in declaration['arms'] if a['name']==name)
            paths=[]
            for compute in ['cpu','gpu']:
                if not arm[compute+'_models']:continue
                candidates=list((TRAIN/(name+'_'+compute)).glob('*/run.json'))
                if not candidates:
                    imported=ROOT/'imports'/(name+'_'+compute)
                    deadline=time.monotonic()+7200
                    while not (imported/'TRANSFER_COMPLETE').exists():
                        if time.monotonic()>deadline:raise RuntimeError('Transfer incomplete: '+str(imported))
                        time.sleep(10)
                    candidates=list(imported.glob('*/run.json'))
                if len(candidates)!=1:raise RuntimeError('Expected exactly one run: '+name+'_'+compute)
                paths.append(candidates[0].parent)
            directory=ROOT/name;directory.mkdir(exist_ok=False)
            write(directory/'universe.json',arm['pairs'])
            command=[sys.executable,str(snapshot/'scripts/audit_corrected_predictions.py'),
                     '--runs',*[str(p) for p in paths], '--universe',str(directory/'universe.json'),
                     '--selected',str(directory/'selected'),'--out',str(directory/'audit'),'--replicates','2000']
            write(directory/'invocation.json',dict(command=command,started_at=now(),
                  inputs=[dict(path=str(p),manifest_sha256=file_hash(p/'run.json')) for p in paths],
                  analysis_code_manifest_sha256=file_hash(ROOT/'analysis_code_manifest.json')))
            env=dict(os.environ,PYTHONPATH=str((snapshot/'src').resolve()),CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
            with (directory/'audit.log').open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(name+' audit exit '+str(result.returncode))
            manifest=json.loads((directory/'audit/audit_manifest.json').read_text())
            for name2,digest in manifest['files'].items():
                if file_hash(directory/'audit'/name2)!=digest:raise RuntimeError('Audit output hash mismatch')
            return dict(state='completed',finished_at=now(),audit_manifest=str(directory/'audit/audit_manifest.json'),
                        n_expected=manifest['n_expected'],files_verified=len(manifest['files']))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(audit,name):name for name in ASSIGNED[args.node]}
            state['arms']={name:{'state':'queued_or_running'} for name in ASSIGNED[args.node]}
            write(status,state)
            for future in as_completed(futures):
                name=futures[future]
                try:state['arms'][name]=future.result()
                except Exception as exc:state['arms'][name]=dict(state='failed',error=repr(exc))
                state['updated_at']=now();write(status,state)
                print(json.dumps(dict(arm=name,**state['arms'][name])),flush=True)
        state['state']='completed' if all(x['state']=='completed' for x in state['arms'].values()) else 'failed'
        state['finished_at']=now();write(status,state)
        if state['state']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
