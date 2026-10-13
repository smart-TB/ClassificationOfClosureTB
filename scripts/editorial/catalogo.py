"""Porta única de leitura dos resultados liberados para o L5.

Todo número, tabela e figura do manuscrito corrigido entra por aqui. O carregador recusa
qualquer caminho que não esteja em `release_manifest.json`, confere o SHA-256 antes da
primeira leitura e registra o que foi lido, de modo que a proveniência de cada saída é o
conjunto de artefatos efetivamente consultados e não uma lista escrita à mão.

O objetivo é tornar impossível o erro que o plano de correção mais repete: um consumidor
editorial cair, por caminho padrão, nas tabelas históricas de `data/`. Não há fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
MANIFESTO = RAIZ / "docs/manuscrito/colaboracao/release_manifest.json"
PRONTO = "ready_for_reporting"

C23 = "data/reanalysis/c7_20260923"
C04 = "data/reanalysis/c7_20261004"
REL = f"{C04}/reporting_release"


class ArtefatoNaoLiberado(RuntimeError):
    """Leitura de um caminho ausente do catálogo, ou catálogo não liberado."""


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


class Catalogo:
    def __init__(self, manifesto: Path = MANIFESTO, raiz: Path = RAIZ) -> None:
        self.raiz = raiz
        self.doc = json.loads(Path(manifesto).read_text(encoding="utf-8"))
        if self.doc.get("status") != PRONTO:
            raise ArtefatoNaoLiberado(
                f"catálogo com status {self.doc.get('status')!r}; L5 exige {PRONTO!r}")
        self.indice = {a["path"]: a for a in self.doc["artifacts"]}
        self.run_id = self.doc["run_id"]
        self._conferidos: set[str] = set()
        self.usados: dict[str, dict] = {}

    # -- acesso ------------------------------------------------------------------
    def meta(self, rel: str) -> dict:
        if rel not in self.indice:
            raise ArtefatoNaoLiberado(f"{rel} não consta no catálogo liberado")
        return self.indice[rel]

    def caminho(self, rel: str) -> Path:
        a = self.meta(rel)
        p = self.raiz / rel
        if rel not in self._conferidos:
            if not p.is_file():
                raise ArtefatoNaoLiberado(f"{rel}: arquivo catalogado não existe")
            real = _sha256(p)
            if real != a["sha256"]:
                raise ArtefatoNaoLiberado(
                    f"{rel}: SHA-256 diverge (catálogo {a['sha256']}, arquivo {real})")
            self._conferidos.add(rel)
        self.usados[rel] = a
        return p

    def csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.caminho(rel), **kw)

    def json(self, rel: str):
        return json.loads(self.caminho(rel).read_text(encoding="utf-8"))

    def parquet(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_parquet(self.caminho(rel), **kw)

    def texto(self, rel: str) -> str:
        return self.caminho(rel).read_text(encoding="utf-8")

    # -- atalhos de localização --------------------------------------------------
    @staticmethod
    def auditoria(braco: str) -> str:
        """Diretório da auditoria de um braço. Principal e clínico usam a revisão v2."""
        if braco in ("primary", "clinical"):
            return f"{C23}/{braco}_audit_v2"
        if braco == "temporal":
            return f"{C23}/temporal_audit"
        return f"{C04}/{braco}/audit"

    @staticmethod
    def selecionado(braco: str) -> str:
        if braco in ("primary", "clinical", "temporal"):
            return f"{C23}/{braco}_selected"
        return f"{C04}/{braco}/selected"

    def proveniencia(self) -> list[dict]:
        """Artefatos consultados até aqui, com os campos de rastreio do catálogo."""
        campos = ("path", "sha256", "arm", "population", "decision_rule", "aggregation",
                  "source_experiment_id")
        return [{c: a.get(c) for c in campos} for _, a in sorted(self.usados.items())]
