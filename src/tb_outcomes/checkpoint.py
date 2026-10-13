"""Checkpoint por par do benchmark (SP4g): grava/retoma cada model×strategy.

A unidade é o par model×strategy (uma linha de status, uma agregação). Invariante:
o .meta.json só existe quando o shard está COMPLETO — o parquet de OOF é escrito
primeiro e o meta por último, via rename atômico. Assim, matar o processo no meio
nunca deixa um par "concluído pela metade" que o resume trataria como pronto.

Sem dependência de `executor` (opera sobre listas/dicts puros) — evita ciclo de import.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tb_outcomes.run_identity import file_sha256


def _slug(model: str, strategy: str) -> str:
    return f"{model}__{strategy}"


def pair_paths(checkpoint_dir, model: str, strategy: str) -> tuple[Path, Path]:
    d = Path(checkpoint_dir)
    base = _slug(model, strategy)
    return d / f"{base}.parquet", d / f"{base}.meta.json"


def pair_checkpoint_exists(checkpoint_dir, model: str, strategy: str) -> bool:
    oof_path, meta_path = pair_paths(checkpoint_dir, model, strategy)
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("checkpoint_version") != 2 or meta["status_row"]["status"] == "error":
            return False
        return (not meta["has_oof"] or
                (oof_path.is_file() and file_sha256(oof_path) == meta["parquet_sha256"]))
    except (OSError, ValueError, KeyError):
        return False


def write_pair_checkpoint(checkpoint_dir, model: str, strategy: str, oof_rows,
                          metric_rows, manifest_rows, aggregate_rows, status_row) -> None:
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    oof_path, meta_path = pair_paths(checkpoint_dir, model, strategy)
    if oof_rows:
        tmp_oof = oof_path.with_suffix(".parquet.tmp")
        pd.DataFrame(oof_rows).to_parquet(tmp_oof, index=False)
        tmp_oof.replace(oof_path)
    meta = {
        "checkpoint_version": 2,
        "parquet_sha256": file_sha256(oof_path) if oof_rows else None,
        "model": model, "strategy": strategy, "status_row": status_row,
        "metric_rows": metric_rows, "manifest_rows": manifest_rows,
        "aggregate_rows": aggregate_rows, "has_oof": bool(oof_rows),
    }
    tmp = meta_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, default=str)
    tmp.replace(meta_path)  # rename atômico: o meta só aparece completo


def load_pair_checkpoint(checkpoint_dir, model: str, strategy: str) -> dict:
    if not pair_checkpoint_exists(checkpoint_dir, model, strategy):
        raise ValueError("Checkpoint absent, historical, failed or corrupted")
    oof_path, meta_path = pair_paths(checkpoint_dir, model, strategy)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    oof_rows: list = []
    if meta.get("has_oof") and oof_path.exists():
        oof_rows = pd.read_parquet(oof_path).to_dict("records")
    return {
        "oof_rows": oof_rows,
        "metric_rows": meta["metric_rows"],
        "manifest_rows": meta["manifest_rows"],
        "aggregate_rows": meta["aggregate_rows"],
        "status_row": meta["status_row"],
    }
