"""Contratos de artefato: escrita atômica e manifesto de execução.

Escrita atômica evita que uma interrupção deixe um artefato truncado que depois
seria lido como válido. Todo artefato entra no run_manifest.json com caminho,
SHA-256, tamanho e etapa geradora (briefing §23).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tb_outcomes.provenance import sha256_file

logger = logging.getLogger(__name__)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Grava em arquivo temporário no mesmo diretório e move com os.replace."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"esperado bytes, recebido {type(data).__name__}")
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destino.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, destino)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    if not isinstance(text, str):
        raise TypeError(f"esperado str, recebido {type(text).__name__}")
    atomic_write_bytes(path, text.encode(encoding))


def write_json(obj: object, path: Path) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Parquet via arquivo temporário; pyarrow precisa de um caminho real."""
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, destino)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_frame_hash(df: pd.DataFrame) -> str:
    """Hash do conteúdo do DataFrame, para compor o run_id."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


@dataclass
class RunManifest:
    """Inventário de tudo que uma execução produziu."""

    run_id: str
    artifacts: list[dict] = field(default_factory=list)

    def record(self, path: Path, step: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"artefato inexistente não pode ser registrado: {p}")
        self.artifacts.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "step": step,
            }
        )
        logger.info("Artefato registrado (%s): %s", step, p)

    def save(self, path: Path) -> None:
        write_json({"run_id": self.run_id, "artifacts": self.artifacts}, path)
