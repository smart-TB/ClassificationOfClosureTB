"""Orçamento de treino DL pré-registrado (SP4f; Opção A, sem busca)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DLConfig(BaseModel):
    max_epochs: int
    early_stopping_patience: int
    validation_split: float
    batch_size: int
    seed: int
    num_workers: int = 0  # dataloader paralelo (SP4g); 0 = padrão single-process


def load_dl_config(path: str | Path) -> DLConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return DLConfig.model_validate(yaml.safe_load(fh))
