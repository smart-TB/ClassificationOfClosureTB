"""Captura a lista de municípios do IBGE como artefato bruto estático.

Roda uma vez, do lado da aquisição (privado). O pacote publicado lê o CSV
resultante e nunca toca a rede, o que torna a harmonização reprodutível e o
teste de equivalência determinístico.

Uso: poetry run python scripts/snapshot_municipios_ibge.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
DESTINO = Path("data/municipios_ibge.csv")


def main() -> int:
    logger.info("Baixando lista de municípios: %s", URL)
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()

    linhas = []
    for m in resp.json():
        micro = m.get("microrregiao")
        meso = micro.get("mesorregiao") if micro else None
        uf = meso.get("UF") if meso else None
        regiao = uf.get("regiao") if uf else None
        if not uf or not uf.get("sigla") or not regiao or not regiao.get("nome"):
            # Mesma condição do bot (proc10): estrutura incompleta é pulada.
            logger.warning(
                "Estrutura incompleta, pulando: %s (ID %s)", m.get("nome"), m.get("id")
            )
            continue
        linhas.append(
            {
                "ID_MUNICIP": str(m["id"]),
                "NOME_MUNIC": m["nome"],
                "SIGLA_UF": uf["sigla"],
                "REGIAO": regiao["nome"],
            }
        )

    df = pd.DataFrame(linhas)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DESTINO, index=False)
    logger.info("Salvos %d municípios em %s", len(df), DESTINO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
