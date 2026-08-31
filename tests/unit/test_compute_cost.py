"""Tempo e custo computacional por modelo (ESPECIFICACAO §6)."""
import pandas as pd

from tb_outcomes.compute_cost import collect, parse_log, summarize_by_model

LOG_COM_TEMPO = """\
2026-08-01 20:47:50,583 [INFO] [benchmark] 1/15 início lightgbm × random_undersampling
2026-08-01 20:50:00,949 [INFO] [benchmark] 1/15 fim lightgbm × random_undersampling: ok em 130.4s
2026-08-01 20:50:02,119 [INFO] [benchmark] 2/15 início lightgbm × random_oversampling
2026-08-01 21:20:03,043 [INFO] [benchmark] 2/15 fim lightgbm × random_oversampling: ok em 1800.9s
"""

LOG_SEM_TEMPO = """\
2026-08-03 22:55:38,122 [INFO] [temporal] 1/33 início majority_class × random_undersampling
2026-08-03 22:56:31,897 [INFO] [temporal] 1/33 fim majority_class × random_undersampling: ok
"""


def _escreve(tmp_path, nome, conteudo):
    p = tmp_path / nome
    p.write_text(conteudo, encoding="utf-8")
    return p


def test_le_o_tempo_explicito_do_log(tmp_path):
    d = parse_log(_escreve(tmp_path, "a.log", LOG_COM_TEMPO), arm="ablacao")
    assert len(d) == 2
    linha = d.set_index("strategy").loc["random_oversampling"]
    assert linha["segundos"] == 1800.9
    assert linha["fonte"] == "log_explicito"
    assert linha["model"] == "lightgbm"


def test_calcula_por_timestamp_quando_o_log_nao_traz_o_tempo(tmp_path):
    d = parse_log(_escreve(tmp_path, "b.log", LOG_SEM_TEMPO), arm="temporal")
    assert len(d) == 1
    # 22:55:38 -> 22:56:31 = 53 s
    assert d.iloc[0]["segundos"] == 53.0
    assert d.iloc[0]["fonte"] == "diferenca_de_timestamp"


def test_o_tempo_do_log_tem_precedencia_sobre_o_timestamp(tmp_path):
    # o par tem início e fim COM tempo; o valor do log manda, não a diferença
    d = parse_log(_escreve(tmp_path, "c.log", LOG_COM_TEMPO), arm="x")
    assert (d["fonte"] == "log_explicito").all()
    assert d.set_index("strategy").loc["random_undersampling", "segundos"] == 130.4


def test_reexecucao_mantem_a_ultima(tmp_path):
    """Um par refeito (retomada após queda de energia, backfill) aparece duas vezes; a
    execução vigente é a última, que é a que produziu o artefato."""
    log = (
        "2026-07-27 10:00:00,000 [INFO] [benchmark] 1/2 fim ft × random_oversampling: ok em 100.0s\n"
        "2026-08-01 10:00:00,000 [INFO] [benchmark] 1/2 fim ft × random_oversampling: ok em 999.0s\n"
    )
    d = collect({"sweep": _escreve(tmp_path, "d.log", log)})
    assert len(d) == 1
    assert d.iloc[0]["segundos"] == 999.0


def test_fim_orfao_sem_inicio_nao_inventa_duracao(tmp_path):
    log = "2026-08-03 22:56:31,897 [INFO] [temporal] 1/33 fim m × s: ok\n"
    assert parse_log(_escreve(tmp_path, "e.log", log), arm="t").empty


def test_resumo_por_modelo_soma_e_ordena(tmp_path):
    d = collect({"ablacao": _escreve(tmp_path, "f.log", LOG_COM_TEMPO)})
    g = summarize_by_model(d)
    assert list(g["model"]) == ["lightgbm"]
    assert g.iloc[0]["n_pares"] == 2
    assert g.iloc[0]["total_horas"] == round((130.4 + 1800.9) / 3600, 3)
    assert g.iloc[0]["pct_do_total"] == 100.0


def test_log_inexistente_nao_derruba(tmp_path):
    d = collect({"nao_existe": tmp_path / "ausente.log"})
    assert isinstance(d, pd.DataFrame) and d.empty
