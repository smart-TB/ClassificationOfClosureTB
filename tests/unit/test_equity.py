"""Equidade e subgrupos (ESPECIFICACAO §3.2).

O que estes testes protegem: (a) a aritmética das métricas por grupo, (b) o intervalo de
Wilson, e sobretudo (c) a **supressão de células pequenas** — que é requisito da §3.2 e
regra §20.3 do briefing (<30). Uma métrica suprimida tem de sair NaN e marcada, nunca
sair como número calculado sobre 3 pacientes.
"""
import numpy as np
import pandas as pd
import pytest

from tb_outcomes.equity import (
    IGNORED_LEVELS,
    MIN_CELL,
    age_bands,
    binary_rates,
    build_subgroup_frame,
    quantile_bands,
    subgroup_metrics,
    wilson_ci,
)


def test_wilson_cobre_a_proporcao_e_respeita_os_limites():
    lo, hi = wilson_ci(30, 100)
    assert lo < 0.30 < hi
    assert 0.0 <= lo and hi <= 1.0
    # amostra minúscula -> intervalo largo, mas ainda dentro de [0,1]
    lo, hi = wilson_ci(1, 2)
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_sem_denominador_e_indefinido():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_binary_rates_aritmetica():
    # 10 positivos: 8 detectados; 90 negativos: 81 corretos, 9 falsos positivos
    y = np.array([1] * 10 + [0] * 90)
    p = np.array([1] * 8 + [0] * 2 + [0] * 81 + [1] * 9)
    r = binary_rates(y, p)
    assert r["tp"] == 8 and r["fn"] == 2 and r["fp"] == 9 and r["tn"] == 81
    assert r["sensibilidade"] == pytest.approx(0.8)
    assert r["especificidade"] == pytest.approx(81 / 90)
    assert r["ppv"] == pytest.approx(8 / 17)
    # o falso negativo é o complemento da sensibilidade — é o que a §3.2 compara
    assert r["fn_rate"] == pytest.approx(0.2)
    assert r["fn_rate"] + r["sensibilidade"] == pytest.approx(1.0)


def test_faixa_etaria_e_ordenada_e_cobre_os_extremos():
    b = age_bands(pd.Series([0, 14, 15, 30, 64, 65, 120, np.nan]))
    assert b.isna().sum() == 1  # ausente continua ausente, não vira faixa
    assert b.cat.ordered
    assert b.iloc[0] != b.iloc[2]
    assert str(b.iloc[-2]).startswith("65")


def test_quintis_de_vulnerabilidade_sao_cinco_e_ordenados():
    b = quantile_bands(pd.Series(np.linspace(0, 1, 1000)), n=5, prefix="Q")
    assert b.nunique() == 5 and b.cat.ordered
    assert b.value_counts().min() >= 190  # quintis equilibrados


def _frame(n_por_grupo, taxa_evento=0.3, seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for grupo, n in n_por_grupo.items():
        y = rng.binomial(1, taxa_evento, n)
        p = np.clip(y * 0.6 + rng.uniform(0, 0.4, n), 0, 1)
        linhas.append(pd.DataFrame({"grupo": grupo, "y": y, "proba": p, "pred": (p > 0.5).astype(int)}))
    return pd.concat(linhas, ignore_index=True)


def test_celula_pequena_e_suprimida_e_marcada():
    df = _frame({"grande": 4000, "minusculo": 12})
    out = subgroup_metrics(df, axis="teste", group_col="grupo", y_col="y",
                           pred_col="pred", proba_col="proba")
    g = out[out["grupo"] == "grande"].iloc[0]
    m = out[out["grupo"] == "minusculo"].iloc[0]

    assert not g["suprimido"] and np.isfinite(g["sensibilidade"])
    assert m["suprimido"]
    # o valor NÃO pode vazar: suprimido significa ausente, não "calculado e escondido"
    for col in ("sensibilidade", "especificidade", "ppv", "fn_rate", "ece"):
        assert pd.isna(m[col]), col
    # o n do grupo continua reportado — a supressão é da métrica, não da existência
    assert m["n"] == 12


def test_supressao_e_por_denominador_de_cada_metrica():
    # grupo grande no total, mas com pouquíssimos eventos: sensibilidade suprimida,
    # especificidade não (o denominador dela é o dos negativos, que é grande).
    df = _frame({"raro": 3000}, taxa_evento=0.002, seed=3)
    assert df.y.sum() < MIN_CELL
    out = subgroup_metrics(df, axis="teste", group_col="grupo", y_col="y",
                           pred_col="pred", proba_col="proba").iloc[0]
    assert pd.isna(out["sensibilidade"]) and pd.isna(out["fn_rate"])
    assert np.isfinite(out["especificidade"])


def test_nivel_ignorado_vira_ausente_e_nao_grupo():
    """'Ignorado' é ausência de informação, não uma população: comparar iniquidade contra
    'pessoas cuja raça não foi registrada' é artefato, e o nível é grande o bastante
    (21,7% em CS_ESCOL_N) para o artefato parecer resultado."""
    X = pd.DataFrame({
        "CS_SEXO": ["M", "F", "I", "M"],
        "CS_RACA": ["1", "9", "4", "2"],
        "CS_ESCOL_N": ["9", "3", "5", "9"],
        "REGIAO": ["Norte", "Sul", "Norte", "Sul"],
        "IDADE": [30, 40, 50, 60],
        "IPEA_PCT_VULNERAVEIS_POBREZA": [0.1, 0.4, 0.6, 0.9],
    })
    sub = build_subgroup_frame(X, pd.Series([0, 0, 1, 1]))

    assert sub["sexo"].tolist()[:2] == ["M", "F"]
    assert pd.isna(sub["sexo"].iloc[2])          # 'I' de CS_SEXO
    assert pd.isna(sub["raca_cor"].iloc[1])      # 9 de CS_RACA
    assert pd.isna(sub["escolaridade"].iloc[0])  # 9 de CS_ESCOL_N
    assert pd.isna(sub["escolaridade"].iloc[3])
    # o registro sai só DAQUELE eixo — segue contando nos demais
    assert sub["regiao"].iloc[1] == "Sul" and sub["raca_cor"].iloc[0] == "1"
    # eixos sem nível Ignorado declarado ficam intactos
    assert sub["regiao"].notna().all()
    assert sub["faixa_etaria"].notna().all()


def test_rotulos_do_dicionario_sinan():
    from tb_outcomes.equity import label_for

    assert label_for("raca_cor", "4") == "Parda"
    assert label_for("raca_cor", "5") == "Indígena"
    assert label_for("sexo", "M") == "Masculino"
    assert label_for("escolaridade", "0") == "Analfabeto"
    assert label_for("escolaridade", "6") == "Médio completo"
    # o DBF pode trazer escolaridade com dois dígitos
    assert label_for("escolaridade", "06") == "Médio completo"
    # 10 é marcador de idade, não nível educacional — o rótulo tem de dizer isso
    assert "idade < 7" in label_for("escolaridade", "10")
    # eixos sem dicionário devolvem o próprio valor
    assert label_for("regiao", "Norte") == "Norte"
    assert label_for("faixa_etaria", "65+") == "65+"


def test_codigo_fora_do_dicionario_aparece_marcado():
    # CS_RACA tem 1 registro com código 6, inexistente no dicionário (válidos 1-5 e 9).
    # Sujeira do dado tem de aparecer na tabela, não sumir dela.
    from tb_outcomes.equity import label_for

    assert label_for("raca_cor", "6") == "6 (código fora do dicionário)"


def test_tabela_carrega_codigo_e_rotulo():
    df = _frame({"1": 500, "4": 500}).rename(columns={"grupo": "raca_cor"})
    out = subgroup_metrics(df, axis="raca_cor", group_col="raca_cor", y_col="y",
                           pred_col="pred", proba_col="proba")
    assert set(out["grupo"]) == {"1", "4"}  # o código segue sendo o identificador
    assert set(out["grupo_rotulo"]) == {"Branca", "Parda"}


def test_o_codigo_do_ignorado_depende_da_coluna():
    # 9 nos categóricos numerados, 'I' em CS_SEXO — um `!= 9` global erraria o sexo
    assert "I" in IGNORED_LEVELS["CS_SEXO"]
    assert IGNORED_LEVELS["CS_RACA"] == {"9"}
    assert "REGIAO" not in IGNORED_LEVELS


def test_auc_separa_discriminacao_de_ponto_de_operacao():
    """Dois grupos com a MESMA ordenação de risco mas prevalências diferentes: a
    sensibilidade a limiar fixo difere, a AUC não. É a distinção que o eixo de escolaridade
    exigiu — sem ela, baixa prevalência é lida como iniquidade."""
    rng = np.random.default_rng(7)

    def grupo(nome, n, taxa):
        y = rng.binomial(1, taxa, n)
        # mesmo poder de ordenação nos dois grupos: proba = sinal + ruído idêntico
        p = np.clip(0.25 + 0.45 * y + rng.normal(0, 0.12, n), 0.001, 0.999)
        return pd.DataFrame({"grupo": nome, "y": y, "proba": p,
                             "pred": (p > 0.5).astype(int)})

    df = pd.concat([grupo("comum", 8000, 0.30), grupo("raro", 8000, 0.05)],
                   ignore_index=True)
    out = subgroup_metrics(df, axis="teste", group_col="grupo", y_col="y",
                           pred_col="pred", proba_col="proba").set_index("grupo")

    assert not out["suprimido"].any()
    # a AUC é praticamente igual — o modelo ordena igual nos dois
    assert abs(out.loc["comum", "auc"] - out.loc["raro", "auc"]) < 0.05
    # a prevalência é reportada, para o leitor ver a origem da diferença
    assert out.loc["comum", "prevalencia"] > 5 * out.loc["raro", "prevalencia"]


def test_auc_indefinida_em_grupo_de_uma_classe_so():
    df = pd.DataFrame({"grupo": "unico", "y": [0] * 200, "proba": np.linspace(0, 1, 200),
                       "pred": [0] * 200})
    out = subgroup_metrics(df, axis="teste", group_col="grupo", y_col="y",
                           pred_col="pred", proba_col="proba").iloc[0]
    assert pd.isna(out["auc"])


def test_grupos_ausentes_nao_viram_categoria():
    df = _frame({"a": 500, "b": 500})
    df.loc[df.index[:50], "grupo"] = np.nan
    out = subgroup_metrics(df, axis="teste", group_col="grupo", y_col="y",
                           pred_col="pred", proba_col="proba")
    assert not out["grupo"].isna().any()
    assert set(out["grupo"]) == {"a", "b"}
