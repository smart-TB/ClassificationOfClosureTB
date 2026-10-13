"""Finite panel updater for the currently running C7 jobs; never releases results."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    root = Path('data/reanalysis/c7_20261004')
    panel = Path('docs/manuscrito/colaboracao/codex')
    previous = None
    deadline = time.monotonic() + 21600
    with (root / 'panel_updater.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() < deadline:
            audits = {}
            for node in ['primary', 'secondary']:
                audits.update(read(root / (node + '_status.json'), {}).get('arms', {}))
            comparison = read(root / 'comparison_status.json', {})
            retry = read(root / 'hiv_comparison_retry_receipt.json', {})
            if retry.get('state') == 'completed':
                comparison['hiv_uncertain'] = retry
            shap = read(root / 'attribution/status.json', {'state': 'starting'})
            if shap.get('state') == 'running':
                alive = subprocess.run(['tmux', 'has-session', '-t', 'smarttb-c7-attribution'],
                                       capture_output=True).returncode == 0
                if not alive:
                    shap = read(root / 'attribution/status.json', shap)
                    if shap.get('state') == 'running':
                        shap.update(state='failed', error='Attribution process exited without completion; inspect attribution_run.log')
                        (root / 'attribution/status.json').write_text(json.dumps(shap, indent=2) + '\n')
            tests = read(root / 'full_tests_v2_status.json', {'state': 'starting'})
            snapshot = {'audits_completed': sum(a.get('state') == 'completed' for a in audits.values()),
                        'audits_total': 9,
                        'comparisons_completed': sum(a.get('state') == 'completed' for a in comparison.values()),
                        'comparisons_total': 8,
                        'attribution_folds_completed': len(shap.get('completed_folds', [])),
                        'attribution_folds_total': 5,
                        'attribution_state': shap.get('state'),
                        'tests_state': tests.get('state'),
                        'failures': {**{k: v for k, v in audits.items() if v.get('state') == 'failed'},
                                     **{k: v for k, v in comparison.items() if v.get('state') == 'failed'}},
                        'release_status': 'not_ready'}
            if snapshot != previous:
                now = datetime.now(timezone.utc).isoformat()
                text = ('C7: auditorias {audits_completed}/9; comparacoes {comparisons_completed}/8; '
                        'SHAP {attribution_folds_completed}/5 dobras; suite integral: {tests_state}.').format(**snapshot)
                with Path('data/reanalysis/c6_remaining_20260923/dashboard.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    state = read(panel / 'status.json', {})
                    state.update(updated_at=now, current_task=text,
                                 next_step='Revisar todas as evidencias C7 e testes; liberar artefatos para Claude L5/L6 apenas apos validacao.')
                    state['tasks']['C6'] = 'completed'
                    state['validation']['c7_20261004_progress'] = dict(snapshot, checked_at=now)
                    state['validation']['full_suite_current'] = 'Ver data/reanalysis/c7_20261004/full_tests_v2_status.json; tentativa anterior preservada.'
                    tmp = panel / 'status.c7.tmp'
                    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
                    tmp.replace(panel / 'status.json')
                    with (panel / 'journal.jsonl').open('a') as journal:
                        journal.write(json.dumps(dict(timestamp=now, task_id='C7', event='progress', summary=text,
                                      evidence=['data/reanalysis/c7_20261004',
                                                'docs/manuscrito/colaboracao/codex/C7_EXECUCAO_20261004.md']),
                                      ensure_ascii=False) + '\n')
                previous = snapshot
                print(json.dumps(dict(snapshot, checked_at=now)), flush=True)
            finished = (all(a.get('state') in ['completed', 'failed'] for a in audits.values()) and len(audits) == 9
                        and len(comparison) == 8 and shap.get('state') in ['completed', 'failed']
                        and tests.get('state') in ['passed', 'failed', 'timeout'])
            if finished:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
