import json
import pandas as pd
import pytest

from tb_outcomes.dl_parallel import (
    atomic_json, bundle_fingerprint, completed, file_hash, gpu_environment, merge,
)


def test_content_fingerprint_stable_and_sensitive_to_values_and_schema():
    x = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    y = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    assert bundle_fingerprint({"X": x}) == bundle_fingerprint({"X": y})
    y.loc[0, "x"] += 1e-12
    assert bundle_fingerprint({"X": x}) != bundle_fingerprint({"X": y})
    assert bundle_fingerprint({"X": x}) != bundle_fingerprint({"X": x.iloc[::-1]})
    assert bundle_fingerprint({"X": x}) != bundle_fingerprint({"X": x[["c", "x"]]})


def _shard(run, fold):
    job = dict(id=f"category_embedding__none__fold{fold}", model="category_embedding",
               strategy="none", fold=fold)
    path = run / "jobs" / (job["id"] + ".parquet")
    pd.DataFrame({"source_row_id": [str(fold)], "record_pos": [fold], "outer_fold": [fold], "model": [job["model"]],
                  "strategy": ["none"], "proba_0": [0.8]}).to_parquet(path, index=False)
    atomic_json(run / "jobs" / (job["id"] + ".json"), {
        "status": "ok", "run_id": run.name, "job": job, "parquet_sha256": file_hash(path),
        "metric_rows": [{"model": job["model"], "strategy": "none", "outer_fold": fold,
                         "class": "__global__", "f1_macro": 0.5 + fold * 0.2}],
        "manifest_rows": [],
    })
    return job


def test_gpu_environment_isolated(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env = gpu_environment("1", 4)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_resume_requires_success_and_matching_shard(tmp_path):
    (tmp_path / "jobs").mkdir()
    job = _shard(tmp_path, 0)
    assert completed(tmp_path, job)
    path = tmp_path / "jobs" / (job["id"] + ".parquet")
    path.write_bytes(b"truncated")
    assert not completed(tmp_path, job)
    job = _shard(tmp_path, 0)
    meta = tmp_path / "jobs" / (job["id"] + ".json")
    d = json.loads(meta.read_text())
    d["status"] = "error"
    atomic_json(meta, d)
    assert not completed(tmp_path, job)


def test_merge_keeps_both_folds_and_aggregates(tmp_path):
    (tmp_path / "jobs").mkdir()
    jobs = [_shard(tmp_path, i) for i in range(2)]
    atomic_json(tmp_path / "run.json", {"n_expected_predictions_per_pair": 2, "unsupported": []})
    merge(tmp_path, jobs)
    oof = pd.read_parquet(tmp_path / "oof_predictions.parquet")
    assert oof.record_pos.tolist() == [0, 1]
    metrics = pd.read_csv(tmp_path / "benchmark_metrics.csv")
    agg = metrics.loc[metrics.kind == "aggregate"].iloc[0]
    assert abs(agg["mean"] - 0.6) < 1e-12
    assert agg.n_folds == 2
    assert (tmp_path / "complete.json").is_file()


def test_scheduler_assigns_two_isolated_gpus_and_reports_failure(tmp_path, monkeypatch):
    import tb_outcomes.dl_parallel as dl

    (tmp_path / "logs").mkdir()
    jobs = [{"id": str(i)} for i in range(3)]
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.job = command[-1]
            self.pid = 100 + len(calls)
            self.returncode = None
            self.polls = 0
            calls.append((self.job, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

        def poll(self):
            self.polls += 1
            if self.polls >= 2:
                self.returncode = 1 if self.job == "2" else 0
            return self.returncode

    checks = {}
    def complete(run, job):
        checks[job["id"]] = checks.get(job["id"], 0) + 1
        return checks[job["id"]] > 1

    monkeypatch.setattr(dl.subprocess, "Popen", Process)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    monkeypatch.setattr(dl, "completed", complete)
    with pytest.raises(RuntimeError, match="Tarefas falharam"):
        dl.schedule(tmp_path, jobs, ["0", "1"], 4)
    assert calls[:2] == [("0", "0"), ("1", "1")]
    assert calls[2] == ("2", "0")
