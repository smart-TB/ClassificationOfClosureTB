"""Fila de ajustes independentes, um subprocesso por GPU ou worker CPU.

Executar da raiz do projeto: python -m tb_outcomes.dl_parallel --help.
Cada tarefa é uma dobra externa de um par modelo/estratégia. O protocolo nested
separa ajuste, calibração e pontuação interna antes do refit e avaliação externa.
O protocolo defaults é somente a referência legada com calibração OOF.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_fingerprint(bundle):
    """Hash do conteúdo, independente da identidade dos objetos no pickle.

    Pickle preserva referências internas; dois DataFrames iguais reconstruídos
    em processos diferentes podem produzir bytes diferentes.
    """
    import pandas as pd

    h = hashlib.sha256()
    def update(value):
        h.update(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            default=lambda x: x.model_dump(mode="json")).encode())
        h.update(b"\0")
    for key, value in sorted(bundle.items()):
        update(key)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            frame = value.to_frame() if isinstance(value, pd.Series) else value
            update([(c, str(dtype)) for c, dtype in frame.dtypes.items()])
            for col in frame:
                if isinstance(frame[col].dtype, pd.CategoricalDtype):
                    update([col, frame[col].cat.categories.tolist(), frame[col].cat.ordered])
            h.update(pd.util.hash_pandas_object(frame, index=True, categorize=False).values.tobytes())
        else:
            update(value)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def runtime_signature():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "pytorch-lightning", "pytorch-tabular", "numpy", "pandas",
             "scikit-learn", "xgboost", "lightgbm", "catboost")}


def source_signature():
    paths = sorted(Path("configs").glob("*.yaml"))
    paths += [p for p in sorted(Path("src/tb_outcomes").glob("*.py"))
              if p.name not in {"figures.py", "labels.py"}]
    return {str(p): file_hash(p) for p in paths}


def freeze_runtime(run, sources):
    """Immutable per-run code/config copy; later repository edits cannot alter workers."""
    project = Path.cwd().resolve()
    runtime = Path(run) / "runtime"
    for name, expected in sources.items():
        source = (project / name).resolve()
        source.relative_to(project)
        destination = runtime / name
        if destination.exists():
            if file_hash(destination) != expected:
                raise RuntimeError(f"Frozen runtime changed: {name}")
            continue
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Source changed while freezing runtime: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return runtime.resolve()


def gpu_environment(gpu, threads):
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu) if str(gpu).isdigit() else "",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               LOKY_MAX_CPU_COUNT=str(threads), PYTHONUNBUFFERED="1")
    return env


def completed(run, job):
    """Só retomar sucesso com shard íntegro; erro e resultado parcial são reexecutados."""
    meta = run / "jobs" / (job["id"] + ".json")
    if not meta.exists():
        return False
    try:
        d = json.loads(meta.read_text())
        parquet = run / "jobs" / (job["id"] + ".parquet")
        inner = run / "jobs" / (job["id"] + ".inner.csv")
        inner_ok = (not d.get("inner_scores_sha256") or
                    (inner.is_file() and file_hash(inner) == d["inner_scores_sha256"]))
        return (inner_ok and d["status"] == "ok" and d["job"] == job and
                d["run_id"] == run.name and parquet.is_file() and
                d["parquet_sha256"] == file_hash(parquet))
    except (KeyError, ValueError, OSError):
        return False


def worker(run, job_id):
    # CUDA_VISIBLE_DEVICES já foi fixado pelo subprocess.Popen ANTES destes imports.
    import logging
    import numpy as np
    import pandas as pd
    import torch
    from tb_outcomes.calibration import load_calibration_config
    from tb_outcomes.dl_config import DLConfig
    from tb_outcomes.executor import PairResult, _evaluate_split, make_sklearn_fold_factory
    from tb_outcomes.imbalance import load_imbalance_config
    from tb_outcomes.models import build_registry, load_models_config
    from tb_outcomes.preprocess import load_preprocess_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((run / "run.json").read_text())
    if source_signature() != manifest["sources"] or runtime_signature() != manifest["packages"]:
        raise RuntimeError("Código/configuração/dependências mudaram durante a execução")
    use_gpu = manifest.get("compute", "gpu") == "gpu"
    if use_gpu and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Cada worker exige exatamente uma GPU CUDA visível")
    job = next(j for j in manifest["jobs"] if j["id"] == job_id)
    if file_hash(run / "inputs.pkl") != manifest["input_sha256"]:
        raise RuntimeError("Bundle de entradas alterado")
    # Arquivo privado criado pelo próprio coordenador; não carregar pickles externos.
    with (run / "inputs.pkl").open("rb") as fh:
        b = pickle.load(fh)
    cfg = DLConfig.model_validate(manifest["dl_config"])
    registry = build_registry(load_models_config("configs/models.yaml"), cfg)
    factory = make_sklearn_fold_factory(
        b["X"], b["y"], b["strata"], registry,
        load_imbalance_config("configs/imbalance.yaml"),
        load_preprocess_config("configs/preprocess.yaml"), b["specs"], b["seed"], cfg)
    if manifest["mode"] == "spatial":
        test = b["outer"].to_numpy() == job["fold"]
        train = ~test
    else:
        years = b["years"].to_numpy(dtype=int)
        train, test = years < b["test_year"], years == b["test_year"]
    train_idx, test_idx = np.flatnonzero(train), np.flatnonzero(test)
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Partição vazia")
    started = time.time()
    print(json.dumps({"event": "start", "job": job, "pid": os.getpid(),
                      "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                      "gpu_name": torch.cuda.get_device_name(0) if use_gpu else None,
                      "time": started}), flush=True)
    if manifest.get("protocol") == "nested":
        from tb_outcomes.corrected_worker import evaluate_nested_job

        result = evaluate_nested_job(b, manifest, job, train_idx, test_idx, cfg, run)
    else:
        result = PairResult(job["model"], job["strategy"])
        _evaluate_split(
            result, job["model"], job["strategy"], b["y"], b["clusters"], train_idx, test_idx,
            len(b["classes"]), b["classes"], factory(job["model"], job["strategy"]),
            registry[job["model"]].capabilities,
            load_calibration_config("configs/calibration.yaml"), b["n_inner"], job["fold"])
    frame = pd.DataFrame(result.oof_rows)
    if len(frame) != len(test_idx) or frame.record_pos.duplicated().any():
        raise RuntimeError("Predições incompletas ou duplicadas")
    frame["source_row_id"] = b["record_ids"].iloc[frame.record_pos].to_numpy()
    parquet = run / "jobs" / (job_id + ".parquet")
    tmp = parquet.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(parquet)
    atomic_json(run / "jobs" / (job_id + ".json"), {
        "status": "ok", "run_id": run.name, "job": job,
        "parquet_sha256": file_hash(parquet), "n_predictions": len(frame),
        "inner_scores_sha256": (file_hash(run / "jobs" / (job_id + ".inner.csv"))
                                 if manifest.get("protocol") == "nested" else None),
        "metric_rows": result.metric_rows, "manifest_rows": result.manifest_rows,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "pid": os.getpid(),
        "started_at": started, "finished_at": time.time(),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if use_gpu else 0,
    })


def schedule(run, jobs, gpus, threads):
    """Fila dinâmica: a próxima dobra vai para a primeira GPU livre."""
    pending = [j for j in jobs if not completed(run, j)]
    if pending:
        (run / "complete.json").unlink(missing_ok=True)
    active = {}
    failed = []
    runtime = run / "runtime"
    successes = len(jobs) - len(pending)
    def progress(state):
        atomic_json(run / "progress.json", {
            "state": state, "updated_at": time.time(), "total": len(jobs),
            "completed": successes, "failed": failed.copy(), "pending": len(pending),
            "active": [{"worker": device, "job": job["id"], "pid": process.pid}
                       for device, (process, _, job) in active.items()],
        })
    progress("running")
    print(f"{len(jobs) - len(pending)} tarefas retomadas; {len(pending)} pendentes", flush=True)
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                log = (run / "logs" / (job["id"] + ".log")).open("w")
                try:
                    environment = gpu_environment(gpu, threads)
                    environment["PYTHONPATH"] = str(runtime / "src")
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "tb_outcomes.dl_parallel", "--worker",
                         str(run), "--job", job["id"]],
                        env=environment, cwd=runtime, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[gpu] = (proc, log, job)
                progress("running")
                print(f"Worker {gpu}: {job['id']} (pid={proc.pid})", flush=True)
            for gpu, (proc, log, job) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close()
                ok = proc.returncode == 0 and completed(run, job)
                if not ok:
                    failed.append(job["id"])
                else:
                    successes += 1
                print(f"Worker {gpu}: {job['id']} -> {'ok' if ok else 'ERRO'}", flush=True)
                del active[gpu]
                progress("running")
            if active:
                time.sleep(1)
    finally:
        for proc, log, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        if active:
            active.clear()
            progress("interrupted")
    if failed:
        progress("failed")
        raise RuntimeError(f"Tarefas falharam (ver logs; reexecutar retoma sucessos): {failed}")
    progress("checking_outputs")


def merge(run, jobs):
    """Consolida parquet em streaming, sem milhões de dicts simultâneos em RAM."""
    import pandas as pd
    import pyarrow.parquet as pq
    from tb_outcomes.executor import PairResult, aggregate_pair

    (run / "complete.json").unlink(missing_ok=True)
    pairs = {}
    run_manifest = json.loads((run / "run.json").read_text())
    expected = run_manifest["n_expected_predictions_per_pair"]
    current_pair, seen = None, set()
    tmp = run / "oof_predictions.parquet.tmp"
    writer = None
    try:
        for job in jobs:
            if not completed(run, job):
                raise RuntimeError(f"Shard ausente/inválido: {job['id']}")
            table = pq.read_table(run / "jobs" / (job["id"] + ".parquet"))
            key = (job["model"], job["strategy"])
            if current_pair != key:
                if current_pair is not None and len(seen) != expected:
                    raise RuntimeError(f"Incomplete prediction coverage: {current_pair}")
                current_pair, seen = key, set()
            ids = table.column("source_row_id").to_pylist()
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise RuntimeError(f"Duplicated prediction IDs: {key}")
            seen.update(ids)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            meta = json.loads((run / "jobs" / (job["id"] + ".json")).read_text())
            key = (job["model"], job["strategy"])
            pair = pairs.setdefault(key, PairResult(*key))
            pair.metric_rows.extend(meta["metric_rows"])
            pair.manifest_rows.extend(meta["manifest_rows"])
        if current_pair is not None and len(seen) != expected:
            raise RuntimeError(f"Incomplete prediction coverage: {current_pair}")
    finally:
        if writer:
            writer.close()
    tmp.replace(run / "oof_predictions.parquet")
    metrics, manifests, statuses = [], [], []
    for pair in pairs.values():
        metrics.extend({**r, "kind": "per_fold"} for r in pair.metric_rows)
        metrics.extend({**r, "kind": "aggregate"} for r in aggregate_pair(pair))
        manifests.extend(pair.manifest_rows)
        statuses.append({"model": pair.model, "strategy": pair.strategy, "status": "ok"})
    statuses.extend(run_manifest["unsupported"])
    pd.DataFrame(metrics).to_csv(run / "benchmark_metrics.csv", index=False)
    atomic_json(run / "calibration_manifest.json", {"manifest": manifests, "status": statuses})
    atomic_json(run / "complete.json", {"jobs": len(jobs), "finished_at": time.time()})
    atomic_json(run / "progress.json", {"state": "completed", "updated_at": time.time(),
                "total": len(jobs), "completed": len(jobs), "failed": [], "pending": 0, "active": []})


def prepare(args):
    from tb_outcomes.cli import _load_benchmark_inputs
    from tb_outcomes.config import load_config
    from tb_outcomes.dl_config import load_dl_config
    from tb_outcomes.executor import load_executor_config
    from tb_outcomes.models import DL_MODELS, build_registry, load_models_config

    cfg = load_executor_config("configs/executor.yaml")
    if args.fold_method:
        cfg = cfg.model_copy(update={"fold_method": args.fold_method})
    if args.protocol == "defaults" and cfg.fold_method != "binpack":
        raise ValueError("Legacy defaults supports binpack only; use nested for fold sensitivity")
    if args.outcome_schema:
        cfg = cfg.model_copy(update={"outcome_schema": args.outcome_schema})
    dl = load_dl_config(args.dl_config)
    if args.sanity:
        # Piloto de engenharia, nunca resultado científico.
        cfg = cfg.model_copy(update={"n_outer_folds": 2,
                                     "sanity_subsample": args.sanity_size,
                                     "n_inner_folds": 3 if args.protocol == "nested" else 2})
        dl = dl.model_copy(update={"max_epochs": args.sanity_epochs})
    allowed = DL_MODELS if args.compute == "gpu" else set(cfg.models) - DL_MODELS
    models = args.models.split(",") if args.models else sorted(allowed)
    strategies = args.strategies.split(",") if args.strategies else ["none", *cfg.strategies]
    if (not models or not set(models) <= allowed or len(set(models)) != len(models)):
        raise ValueError("Models must be distinct and compatible with --compute")
    if (not strategies or not set(strategies) <= {"none", "random_oversampling", "random_undersampling", "local_cost_sensitive"}
            or len(set(strategies)) != len(strategies)):
        raise ValueError("Invalid imbalance strategies")
    registry = build_registry(load_models_config("configs/models.yaml"), dl)
    unsupported = [{"model": m, "strategy": s, "status": "not_run_incompatible",
                    "reason": "adapter does not support this strategy"}
                   for m in models for s in strategies
                   if s not in registry[m].capabilities.supported_imbalance]
    sources = source_signature()
    inputs = _load_benchmark_inputs(cfg, args.sanity, k_override=args.k,
                                    feature_scope=args.scope, with_year=True,
                                    with_record_id=True,
                                    context_profile=args.context_profile,
                                    availability_profile=args.availability_profile,
                                    population_role="temporal" if args.mode == "temporal"
                                    else "development")
    X, y, strata, clusters, outer, classes, specs, labels, years, record_ids = inputs
    if args.mode == "temporal" and years.isna().any():
        raise ValueError("Ano ausente impede partição temporal")
    test_year = int(load_config("configs/analysis_decisions.yaml").study.temporal_test_period)
    if args.mode == "temporal" and (not (years < test_year).any() or not (years == test_year).any()):
        raise ValueError("Treino ou teste temporal vazio")
    bundle = dict(X=X, y=y, strata=strata, clusters=clusters, outer=outer, classes=classes,
                  specs=specs, labels=labels, years=years, record_ids=record_ids,
                  test_year=test_year,
                  seed=cfg.random_state, n_inner=cfg.n_inner_folds, fold_method=cfg.fold_method)
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fd, input_tmp = tempfile.mkstemp(prefix="inputs-", suffix=".pkl", dir=root)
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(bundle, fh, protocol=5)
        folds = sorted(int(f) for f in outer.unique()) if args.mode == "spatial" else [f"holdout_{test_year}"]
        jobs = [dict(id=f"{m}__{s}__fold{f}", model=m, strategy=s, fold=f)
                for s in strategies for m in models for f in folds
                if s in registry[m].capabilities.supported_imbalance]
        if not jobs:
            raise ValueError("No compatible model/strategy pair to execute")
        metadata = dict(mode=args.mode, k=args.k or cfg.k, scope=args.scope, sanity=args.sanity,
                        context_profile=args.context_profile, outcome_schema=cfg.outcome_schema,
                        availability_profile=args.availability_profile,
                        protocol=args.protocol, runtime_snapshot=True,
                        max_configs=1 if args.sanity else args.max_configs,
                        compute=args.compute, threads=args.threads, unsupported=unsupported,
                        n_expected_predictions_per_pair=(len(X) if args.mode == "spatial"
                                                         else int(years.eq(test_year).sum())),
                        harmonized_sha256=file_hash("data/harmonized.parquet"),
                        input_content_sha256=bundle_fingerprint(bundle), sources=sources,
                        dl_config=dl.model_dump(), packages=runtime_signature(), jobs=jobs)
        scientific = {key: value for key, value in metadata.items()
                      if key not in {"compute", "threads", "unsupported", "jobs"}}
        metadata["experiment_id"] = hashlib.sha256(
            json.dumps(scientific, sort_keys=True).encode()).hexdigest()[:20]
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
        run = root / digest
        run.mkdir(exist_ok=True)
        # Linux flock evita dois coordenadores escrevendo na mesma execução.
        import fcntl
        lock = (run / ".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock.close()
            raise
        for folder in ("jobs", "logs"):
            (run / folder).mkdir(exist_ok=True)
        freeze_runtime(run, sources)
        if (run / "inputs.pkl").exists():
            previous = json.loads((run / "run.json").read_text())
            if (file_hash(run / "inputs.pkl") != previous["input_sha256"] or
                    previous["input_content_sha256"] != metadata["input_content_sha256"]):
                raise RuntimeError("Bundle existente inválido")
            metadata["input_sha256"] = previous["input_sha256"]
        else:
            metadata["input_sha256"] = file_hash(input_tmp)
            Path(input_tmp).replace(run / "inputs.pkl")
        atomic_json(run / "run.json", metadata)
        return run, jobs, args.threads, lock
    finally:
        Path(input_tmp).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1", help="Índices físicos distintos (ordem PCI)")
    p.add_argument("--compute", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--cpu-workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--protocol", choices=["nested", "defaults"], default="nested")
    p.add_argument("--max-configs", type=int, default=3, help="Declared configurations per model/strategy")
    p.add_argument("--mode", choices=["spatial", "temporal"], default="spatial")
    p.add_argument("--models", help="Lista separada por vírgula; padrão: quatro DL")
    p.add_argument("--strategies", help="Lista separada por vírgula; padrão: compatíveis do executor")
    p.add_argument("--k", type=int)
    p.add_argument("--fold-method", choices=["binpack", "random_group"])
    p.add_argument("--scope", choices=["combined", "individual", "municipal", "clinical_baseline"], default="combined")
    p.add_argument("--context-profile", choices=["baseline", "contemporary_sensitivity"], default="baseline")
    p.add_argument("--availability-profile", choices=["primary", "hiv_uncertain", "without_diagnostic_tests"],
                   default="primary")
    p.add_argument("--outcome-schema", choices=["sensitive_tb_3class", "all_cause_3class"])
    p.add_argument("--dl-config", type=Path, default=Path("configs/deep_learning.yaml"))
    p.add_argument("--out-root", type=Path, default=Path("data/dl_parallel"))
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--sanity-size", type=int, default=800, help="Engineering pilot sample size only")
    p.add_argument("--sanity-epochs", type=int, default=1, help="Engineering pilot epochs only")
    p.add_argument("--display", help="Display owning NVIDIA controls, e.g. :1")
    p.add_argument("--xauthority", help="X authority file of the authenticated desktop owner")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--job", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.worker, args.job)
        return
    gpus = args.gpus.split(",")
    if not gpus or any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        p.error("--gpus exige índices inteiros distintos")
    if args.cpu_workers < 1 or args.threads < 1 or args.max_configs < 1:
        p.error("Worker, thread and configuration counts must be positive")
    if args.sanity_size < 100 or args.sanity_epochs < 1:
        p.error("Pilot requires at least 100 records and one epoch")
    if not args.sanity and (args.sanity_size != 800 or args.sanity_epochs != 1):
        p.error("Pilot size/epochs require --sanity")
    if args.compute == "gpu":
        available = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
        if not set(gpus) <= set(available):
            p.error(f"GPUs indisponíveis: {gpus}; disponíveis: {available}")
    else:
        gpus = [f"cpu{i}" for i in range(args.cpu_workers)]
    run, jobs, threads, lock = prepare(args)
    print(f"Execução: {run}", flush=True)
    try:
        from tb_outcomes.gpu_fans import GPUFanLease

        if args.compute == "cpu":
            schedule(run, jobs, gpus, threads)
        elif any(not completed(run, j) for j in jobs):
            with GPUFanLease(run / "fan_events.jsonl", display=args.display,
                             xauthority=args.xauthority, expected_gpu_count=len(available)):
                schedule(run, jobs, gpus, threads)
        merge(run, jobs)
    finally:
        lock.close()
    print(f"Concluído: {run}", flush=True)


if __name__ == "__main__":
    main()
