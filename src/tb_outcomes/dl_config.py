"""Orçamento de treino DL pré-registrado (SP4f; Opção A, sem busca)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(ge=0)
    validation_split: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=1)
    seed: int
    num_workers: int = Field(default=0, ge=0)
    precision: Literal["32-true", "bf16-mixed", "16-mixed"] = "32-true"
    matmul_precision: Literal["highest", "high", "medium"] = "highest"
    torch_num_threads: int = Field(default=4, gt=0)
    inference_batch_size: int | None = Field(default=None, gt=1)
    # Spawn evita fork de workers depois da inicialização do CUDA.
    prefetch_factor: int = Field(default=2, gt=0)
    checkpoint_root: str | None = None
    learning_rate: float | None = Field(default=None, gt=0)

    def dataloader_kwargs(self) -> dict:
        if self.num_workers == 0:
            return {}
        return {"multiprocessing_context": "spawn", "persistent_workers": True,
                "prefetch_factor": self.prefetch_factor}


def load_dl_config(path: str | Path) -> DLConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return DLConfig.model_validate(yaml.safe_load(fh))
