from unittest.mock import Mock

import pytest

from tb_outcomes.gpu_fans import GPUFanLease


def test_targets_verified_on_both_gpus(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl")
    lease._settings = Mock(side_effect=["", "1\n1\n", "80\n80\n"])
    lease._set(80)
    assert lease._settings.call_args_list[0].args[0][-1] == "[fan]/GPUTargetFanSpeed=80"


def test_one_unadjusted_fan_blocks_training(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl")
    lease._settings = Mock(side_effect=["", "1\n1\n", "80\n30\n"])
    with pytest.raises(RuntimeError, match="not confirmed"):
        lease._set(80)


def test_worker_error_still_restores_after_processes_end(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl")
    lease._active_compute = Mock(return_value=[])
    lease._set = Mock()
    lease.__exit__(RuntimeError, RuntimeError("worker failed"), None)
    lease._set.assert_called_once_with(30)


def test_never_lowers_fans_with_another_worker_running(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl")
    lease._active_compute = Mock(return_value=["123"])
    lease._set = Mock()
    with pytest.raises(RuntimeError, match="Cannot lower"):
        lease.__exit__(None, None, None)
    lease._set.assert_not_called()


def test_single_gpu_verifies_both_physical_fans(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl", expected_gpu_count=1)
    lease._settings = Mock(side_effect=["", "1\n", "80\n80\n"])
    lease._set(80)


@pytest.mark.parametrize("states,targets", [("1\n", "80\n30\n"), ("0\n", "80\n80\n"), ("", "80\n")])
def test_single_gpu_rejects_unverified_control(tmp_path, states, targets):
    lease = GPUFanLease(tmp_path / "events.jsonl", expected_gpu_count=1)
    lease._settings = Mock(side_effect=["", states, targets])
    with pytest.raises(RuntimeError, match="not confirmed"):
        lease._set(80)


def test_missing_second_gpu_blocks_two_gpu_lease(tmp_path):
    lease = GPUFanLease(tmp_path / "events.jsonl", expected_gpu_count=2)
    lease._settings = Mock(side_effect=["", "1\n", "80\n80\n"])
    with pytest.raises(RuntimeError, match="not confirmed"):
        lease._set(80)


def test_invalid_gpu_count_rejected(tmp_path):
    with pytest.raises(ValueError, match="Positive"):
        GPUFanLease(tmp_path / "events.jsonl", expected_gpu_count=0)
