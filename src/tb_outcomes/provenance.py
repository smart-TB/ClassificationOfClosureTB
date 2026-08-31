"""Proveniência: checksums, estado do git, ambiente, hardware e run_id.

Sem git, um resultado não fica amarrado a uma versão do código. Registramos a
limitação explicitamente em vez de escondê-la: git_state() detecta e devolve
available=False. Quando o repositório for inicializado, o campo passa a ser
preenchido sozinho, sem mudança de código.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BIBLIOTECAS_RASTREADAS = [
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "pytorch-lightning",
    "pytorch_tabular",
    "shap",
    "geopandas",
    "pyproj",
    "pyarrow",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 de um arquivo, lido em blocos para suportar Parquet grande."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for bloco in iter(lambda: fh.read(chunk_size), b""):
            h.update(bloco)
    return h.hexdigest()


def git_state() -> dict:
    """Estado do repositório, se houver. Nunca levanta exceção."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commit.returncode != 0:
            return {"available": False, "reason": "not_a_repository"}
        sujo = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": True,
            "commit": commit.stdout.strip(),
            "dirty": bool(sujo.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "reason": f"git_unavailable: {e}"}


def environment() -> dict:
    """Versões de Python e das bibliotecas do briefing §25.1."""
    env: dict = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in BIBLIOTECAS_RASTREADAS:
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    return env


def hardware() -> dict:
    """CPU, RAM e GPU (briefing §25.2)."""
    hw: dict = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "os": platform.platform(),
    }
    try:
        paginas = os.sysconf("SC_PHYS_PAGES")
        tamanho = os.sysconf("SC_PAGE_SIZE")
        hw["ram_total_gb"] = round(paginas * tamanho / 1024**3, 1)
    except (ValueError, OSError):
        hw["ram_total_gb"] = 0.0
    try:
        import torch

        hw["cuda_available"] = torch.cuda.is_available()
        hw["gpus"] = [
            {
                "name": torch.cuda.get_device_properties(i).name,
                "vram_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 1
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:  # torch ausente ou driver indisponível não é erro de proveniência
        hw["cuda_available"] = False
        hw["gpus"] = []
    return hw


def make_run_id(track: str, config_hash: str, data_hash: str) -> str:
    """Identificador único de execução. Uma execução nunca sobrescreve outra."""
    agora = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git = git_state()
    marca = f"g{git['commit'][:7]}" if git["available"] else "nogit"
    return f"{track}_{agora}_{marca}_{config_hash}_{data_hash}"
