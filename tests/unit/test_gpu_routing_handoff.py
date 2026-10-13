"""Boundary guards: changed routing must not duplicate work or replace artifacts."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).parents[2] / 'scripts/rebalance_gpu_campaign.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def declaration():
    return {'arms': [{'name': name, 'gpu_node': node, 'gpu_models': ['ft_transformer', 'tabnet']}
        for node, names in [('primary', ['k27', 'k75', 'random_group']),
                            ('secondary', ['contemporary', mod.TARGET, 'hiv_uncertain', 'all_cause'])]
        for name in names]}


def test_route_preserves_exact_universe_and_one_owner():
    d = declaration()
    p, s = mod.route(d, 'primary'), mod.route(d, 'secondary')
    assert set(p).isdisjoint(s)
    assert set(p + s) == {a['name'] for a in d['arms']}
    assert mod.TARGET in p and mod.TARGET not in s
    assert s == ['contemporary', 'hiv_uncertain', 'all_cause']


def test_changed_original_route_is_rejected():
    d = declaration()
    d['arms'][0]['name'] = 'unexpected'
    with pytest.raises(ValueError):
        mod.route(d, 'primary')


def test_persistent_barrier_stops_legacy_mkdir_even_after_creator_exits(tmp_path):
    p = tmp_path / 'without_diagnostics_gpu'
    receipt = {'plan_sha256': 'a', 'destination': 'primary'}
    mod.reserve_source(p, receipt)
    mod.check_gate(p, receipt)
    with pytest.raises(FileExistsError):
        p.mkdir(parents=True, exist_ok=True)
    assert json.loads(p.read_text()) == receipt


def test_existing_run_directory_is_never_replaced(tmp_path):
    p = tmp_path / 'without_diagnostics_gpu'
    p.mkdir()
    artifact = p / 'existing_result'
    artifact.write_text('preserve')
    with pytest.raises(FileExistsError):
        mod.reserve_source(p, {})
    assert artifact.read_text() == 'preserve'


def test_changed_gate_is_rejected(tmp_path):
    p = tmp_path / 'gate'
    mod.reserve_source(p, {'plan_sha256': 'a'})
    with pytest.raises(ValueError):
        mod.check_gate(p, {'plan_sha256': 'b'})


def test_arguments_preserve_scientific_options():
    arm = {'gpu_models': ['ft_transformer', 'tabnet'],
           'options': {'availability_profile': 'without_diagnostic_tests'}}
    args = mod.arguments(arm, Path('destination'))
    assert args.models == 'ft_transformer,tabnet'
    assert args.protocol == 'nested' and args.max_configs == 3
    assert args.availability_profile == 'without_diagnostic_tests'
    assert args.context_profile == 'baseline' and args.fold_method == 'binpack'
    assert args.outcome_schema == 'sensitive_tb_3class' and not args.sanity


def test_lane_cannot_be_taken_while_original_coordinator_holds_it(tmp_path):
    import fcntl
    path = tmp_path / 'secondary_gpu.lock'
    with path.open('a') as original, path.open('a') as successor:
        fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(successor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(original, fcntl.LOCK_UN)
        fcntl.flock(successor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_missing_gate_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        mod.check_gate(tmp_path / 'missing', {'plan_sha256': 'a'})


@pytest.mark.parametrize('node,old_state,old_arm,expected', [
    ('primary', 'completed', 'random_group', ['without_diagnostics']),
    ('secondary', 'preparing', 'without_diagnostics', ['hiv_uncertain', 'all_cause']),
])
def test_successor_executes_only_its_assigned_future_arms(tmp_path, monkeypatch,
                                                        node, old_state, old_arm, expected):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))
    import run_remaining_campaign as campaign
    import tb_outcomes.dl_parallel as dl
    import tb_outcomes.gpu_fans as fans
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'campaign'
    ops = root / 'ops'
    ops.mkdir(parents=True)
    monkeypatch.setattr(mod, 'ROOT', root)
    monkeypatch.setattr(mod, 'OPS', ops)
    mod.write_json(ops / 'amendment.json', {'test': True})
    receipt = {'plan_sha256': mod.digest(ops / 'amendment.json')}
    mod.write_json(ops / 'source_gate_receipt.json', receipt)
    if node == 'secondary':
        mod.reserve_source(root / 'without_diagnostics_gpu', receipt)
    mod.write_json(root / (node + '_gpu_status.json'), {'state': old_state, 'arm': old_arm})
    d = declaration()
    for arm in d['arms']:
        arm['options'] = {}
    plan = {'assigned': {n: mod.route(d, n) for n in ['primary', 'secondary']},
            'target_identity': {'identical': True}}
    monkeypatch.setattr(mod, 'check_engine', lambda *a: None)
    verified = []
    monkeypatch.setattr(mod, 'check_completed_arm', lambda name: verified.append(name))
    monkeypatch.setattr(mod, 'identity', lambda run: {'identical': True})
    prepared, trained, merged, registrations = [], [], [], []

    def prepare(args):
        name = args.out_root.name.removesuffix('_gpu')
        prepared.append(name)
        run = args.out_root / 'test_run'
        run.mkdir(parents=True)
        return run, [{'id': name}], 4, (run / '.lock').open('a')

    class Lease:
        def __init__(self, *args, **kwargs):
            assert kwargs['expected_gpu_count'] == (2 if node == 'primary' else 1)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dl, 'prepare', prepare)
    monkeypatch.setattr(dl, 'completed', lambda *a: False)
    monkeypatch.setattr(dl, 'schedule', lambda run, jobs, devices, threads: trained.append(jobs[0]['id']))
    monkeypatch.setattr(dl, 'merge', lambda run, jobs: merged.append(jobs[0]['id']))
    monkeypatch.setattr(fans, 'GPUFanLease', Lease)
    monkeypatch.setattr(campaign, 'register', lambda *a: registrations.append(dict(a[-1])))
    mod.execute(node, plan, d)
    assert prepared == trained == merged == expected
    assert verified == (['k27', 'k75', 'random_group'] if node == 'primary' else ['contemporary'])
    assert registrations[-1]['state'] == 'completed'
    assert registrations[-1]['completed_arms'] == plan['assigned'][node]
