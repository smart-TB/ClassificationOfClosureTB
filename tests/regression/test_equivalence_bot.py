"""data.py deve reproduzir exatamente o parquet que o bot gera.

Este é o portão da fronteira entre o pipeline de aquisição privado (etapas 1-7) e
o pacote publicado (harmonização, etapas 8-14 portadas). As duas implementações
coexistem; este teste prova que não divergiram.

Requer os artefatos brutos reais em data/. Pula com mensagem clara quando
ausentes — eles não são versionados (microdado).
"""
from pathlib import Path

import pandas as pd
import pytest

from tb_outcomes.config import load_config
from tb_outcomes.data import harmonize, load_raw_inputs

DATA = Path("data")
REFERENCIA = DATA / "sinnan_tratado.parquet"
CONFIG = Path("configs/analysis_decisions.yaml")

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def referencia() -> pd.DataFrame:
    if not REFERENCIA.exists():
        pytest.skip(
            f"{REFERENCIA} ausente (microdado, não versionado). "
            "Rode o pipeline de aquisição para gerá-lo."
        )
    return pd.read_parquet(REFERENCIA)


@pytest.fixture(scope="module")
def portado(tmp_path_factory) -> pd.DataFrame:
    if not (DATA / "sinnan.parquet").exists():
        pytest.skip("brutos ausentes; rode o pipeline de aquisição (etapas 1-7).")
    df = harmonize(load_raw_inputs(DATA), load_config(CONFIG))

    # Persiste e relê antes de comparar. A referência do bot é um ARQUIVO
    # parquet, não um frame em memória, e o pyarrow normaliza os nulos de
    # colunas object (NaN -> None) na ida e volta. Comparar um frame em memória
    # com um arquivo compararia coisas diferentes e dispararia o
    # 'Mismatched null-like values nan and None' do pandas — que hoje passa com
    # aviso e numa versão futura vira erro. O que precisa ser idêntico é o que
    # os dois pipelines PERSISTEM.
    caminho = tmp_path_factory.mktemp("port") / "portado.parquet"
    df.to_parquet(caminho, index=False)
    return pd.read_parquet(caminho)


def test_same_shape(referencia, portado):
    assert portado.shape == referencia.shape


def test_same_columns_in_same_order(referencia, portado):
    assert list(portado.columns) == list(referencia.columns)


def test_same_dtypes(referencia, portado):
    diferentes = {
        c: (str(referencia[c].dtype), str(portado[c].dtype))
        for c in referencia.columns
        if c in portado.columns and str(referencia[c].dtype) != str(portado[c].dtype)
    }
    assert not diferentes, f"dtypes divergentes: {diferentes}"


def test_frames_are_identical(referencia, portado):
    # O bot ordena por (ID_MUNIC_ANALISE, ANOMES_DIAG) em proc13 e grava com
    # index=False: a ordem do arquivo não é a de entrada. Reindexamos os dois
    # lados para comparar conteúdo, não permutação.
    chaves = ["ID_MUNIC_ANALISE", "DT_DIAG", "DT_NOTIFIC", "SITUA_ENCE"]
    a = referencia.sort_values(chaves, kind="stable").reset_index(drop=True)
    b = portado.sort_values(chaves, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a, check_dtype=True, check_exact=True)
