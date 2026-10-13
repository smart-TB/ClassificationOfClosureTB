"""Microbenchmark sintético de runtime, sem avaliar ou selecionar por desfecho real.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/profile_dl_runtime.py \
    --models category_embedding,tabnet --out data/gpu_review/profile_gpu0.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/profile_dl_runtime.py \
    --models tab_transformer,ft_transformer --out data/gpu_review/profile_gpu1.json
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--variants", default="baseline,optimized,bf16,workers2")
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from tb_outcomes.dl_config import DLConfig
    from tb_outcomes.models import build_dl_adapter, load_models_config

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Usar CUDA_VISIBLE_DEVICES para isolar uma GPU")
    entries = load_models_config("configs/models.yaml").models
    original = Path("docs/manuscrito/backups/gpu_20260919/src/tb_outcomes/models.py").resolve()
    old = None
    if "baseline" in args.variants.split(","):
        spec = importlib.util.spec_from_file_location("tb_original_models", original)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
    rng = np.random.RandomState(42)
    X = pd.DataFrame({f"num_{i}": rng.normal(size=args.n) for i in range(35)})
    for i in range(25):
        X[f"cat_{i}"] = rng.choice(["A", "B", "C", "D", "E", "F"], args.n)
    X.iloc[::17, 0] = np.nan
    y = rng.choice([0, 1, 2], args.n, p=[0.76, 0.19, 0.05])
    xt = X.iloc[:4096].copy()
    rows = []
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.ones((256, 256), device="cuda").matmul(torch.ones((256, 256), device="cuda"))
    torch.cuda.synchronize()
    for name in args.models.split(","):
        for variant in args.variants.split(","):
            torch.set_num_threads(32 if variant == "baseline" else 4)
            torch.set_float32_matmul_precision("highest")
            cfg = DLConfig(max_epochs=args.epochs, early_stopping_patience=8,
                           validation_split=0.15, batch_size=1024, seed=42,
                           precision="bf16-mixed" if variant == "bf16" else "32-true",
                           num_workers=2 if variant == "workers2" else 0,
                           inference_batch_size=4096)
            ad = (old.build_dl_adapter(name, entries[name], cfg) if variant == "baseline"
                  else build_dl_adapter(name, entries[name], cfg))
            row = dict(model=name, variant=variant, n_train=args.n, n_predict=len(xt),
                       epochs=args.epochs, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       gpu_name=torch.cuda.get_device_name(0), started_at=time.time())
            print(json.dumps({"event": "start", **row}), flush=True)
            torch.cuda.reset_peak_memory_stats()
            cwd = Path.cwd()
            try:
                with tempfile.TemporaryDirectory(prefix="tb-profile-") as tmp:
                    # O adapter antigo grava diretórios relativos: isolar também o baseline.
                    os.chdir(tmp)
                    start = time.perf_counter()
                    ad.fit(X, y)
                    torch.cuda.synchronize()
                    row["fit_seconds"] = time.perf_counter() - start
                    start = time.perf_counter()
                    probabilities = ad.predict_raw_scores(xt)
                    torch.cuda.synchronize()
                    row["predict_seconds"] = time.perf_counter() - start
                    assert probabilities.shape == (len(xt), 3)
                    assert np.isfinite(probabilities).all()
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=0.02)
                    row["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
                    row["status"] = "ok"
            except Exception:
                row["status"] = "error"
                row["error"] = traceback.format_exc()
            finally:
                os.chdir(cwd)
                if hasattr(ad, "close"):
                    ad.close()
                del ad
                gc.collect()
                torch.cuda.empty_cache()
            row["finished_at"] = time.time()
            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
