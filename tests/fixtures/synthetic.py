"""Gerador de dado sintético com o schema do SINAN-TB bruto.

Nenhum dado real. Cobre deliberadamente os casos que o snapshot real não cobre
(código 6) e os que quebrariam o pipeline em silêncio (duas eras de codificação,
divergência residência×notificação, datas impossíveis, município só no Censo 2022).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Municípios fictícios, mas com geocódigos de 7 dígitos plausíveis.
MUNICIPIOS = [
    {"ID_MUNICIP": "1100015", "NOME_MUNIC": "Alfa", "SIGLA_UF": "RO", "REGIAO": "Norte"},
    {"ID_MUNICIP": "2300100", "NOME_MUNIC": "Beta", "SIGLA_UF": "CE", "REGIAO": "Nordeste"},
    {"ID_MUNICIP": "3500105", "NOME_MUNIC": "Gama", "SIGLA_UF": "SP", "REGIAO": "Sudeste"},
    {"ID_MUNICIP": "4100103", "NOME_MUNIC": "Delta", "SIGLA_UF": "PR", "REGIAO": "Sul"},
    # Criado após 2010: só existe no Censo 2022 (caso do backfill proibido).
    {"ID_MUNICIP": "5100102", "NOME_MUNIC": "Epsilon", "SIGLA_UF": "MT", "REGIAO": "Centro-Oeste"},
]
MUNICIPIO_NOVO = "5100102"
ANOS = list(range(2015, 2026))


# Códigos de UF do IBGE correspondentes aos municípios acima.
ESTADOS = [
    {"codigo_uf": 11, "latitude": -10.83, "longitude": -63.34, "regiao": "Norte"},
    {"codigo_uf": 23, "latitude": -5.20, "longitude": -39.53, "regiao": "Nordeste"},
    {"codigo_uf": 35, "latitude": -22.19, "longitude": -48.79, "regiao": "Sudeste"},
    {"codigo_uf": 41, "latitude": -24.89, "longitude": -51.55, "regiao": "Sul"},
    {"codigo_uf": 51, "latitude": -12.64, "longitude": -55.42, "regiao": "Centro-Oeste"},
]


@dataclass
class SyntheticRaw:
    sinan: pd.DataFrame
    municipios: pd.DataFrame
    municipios_latlong: pd.DataFrame
    estados: pd.DataFrame
    ibge: pd.DataFrame
    ipea: pd.DataFrame
    cnes_prof: pd.DataFrame
    cnes_estab: pd.DataFrame


def _sinan(rng: np.random.Generator, n: int) -> pd.DataFrame:
    ids = [m["ID_MUNICIP"] for m in MUNICIPIOS]
    anos = rng.choice(ANOS, size=n)
    meses = rng.integers(1, 13, size=n)
    dias = rng.integers(1, 29, size=n)

    resi = rng.choice(ids, size=n)
    # ~15% com notificação em município diferente da residência.
    noti = np.where(rng.random(n) < 0.15, rng.choice(ids, size=n), resi)

    situa = []
    for ano in anos:
        r = rng.random()
        if r < 0.06:
            # Sem encerramento: '0' até 2017, vazio de 2018 em diante.
            situa.append("0" if ano <= 2017 else "")
        else:
            # Desfecho dominado por 'cura' (código 1), como no dado real. Antes a escolha
            # era uniforme sobre 1–10, deixando as 3 classes quase iguais (~70 cada); com o
            # cap de oversampling 1:3 isso torna o random_oversampling um no-op idêntico a
            # 'none' (a minoria já fica > maioria/3) e derruba o audit de imbalance. Enviesar
            # para 'cura' recoloca a minoria < maioria/3 — o oversampling volta a diferir —
            # e é mais fiel ao real. Todos os códigos 1–10 seguem com peso > 0 para exercitar
            # a codificação SITUA_ENCE.
            situa.append(str(rng.choice(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                p=[0.60, 0.13, 0.09, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.06],
            )))

    dt_notific = [f"{a}-{m:02d}-{d:02d}" for a, m, d in zip(anos, meses, dias)]
    dt_encerra = [
        "1899-12-30" if rng.random() < 0.02 else f"{a + 1}-{m:02d}-{d:02d}"
        for a, m, d in zip(anos, meses, dias)
    ]

    df = pd.DataFrame(
        {
            "TP_NOT": ["2"] * n,
            "DT_NOTIFIC": dt_notific,
            "NU_ANO": [str(a) for a in anos],
            "DT_DIAG": dt_notific,
            "ANO_NASC": [str(a - int(i)) for a, i in zip(anos, rng.integers(1, 90, size=n))],
            "CS_SEXO": rng.choice(["M", "F"], size=n),
            "CS_RACA": rng.choice(["1", "2", "4", "9"], size=n),
            "ID_MUNICIP": noti,
            "ID_MN_RESI": resi,
            "SG_UF_NOT": [m[:2] for m in noti],
            "HIV": rng.choice(["1", "2", "3", "4", "9"], size=n),
            "AGRAVAIDS": rng.choice(["1", "2", "9"], size=n),
            # Texto livre: precisa sobreviver à tipagem (caso AGRAVOUTDE).
            "AGRAVOUTDE": rng.choice(["", "HIPERTENSAO", "DIABETES", "286.4"], size=n),
            "SITUA_ENCE": situa,
            "DT_ENCERRA": dt_encerra,
            "ANT_RETRO": rng.choice(["1", "2", "9"], size=n),
        }
    )
    df = _completa_schema_sinan(df, rng, n)

    # Duplicatas exatas para a auditoria do §10.
    df = pd.concat([df, df.head(5)], ignore_index=True)
    return df


# Domínios observados no SINAN real, por coluna. A fixture precisa ter o MESMO
# schema do bruto (spec §5): o portão de features.py compara as 168 colunas e
# falha se a fixture for menor — foi assim que a versão mínima foi pega.
DOMINIOS_SINAN = {
    "ID_AGRAVO": ["A169", "A144", "A16."],
    "ID_REGIONA": ["1", "2", "3"],
    "NU_IDADE_N": ["4030", "4045", "4062"],
    "CS_GESTANT": ["1", "2", "5", "6", "9"],
    "CS_ESCOL_N": ["1", "2", "3", "4", "9", "10"],
    "SG_UF": ["11", "23", "35", "41", "51"],
    "ID_RG_RESI": ["1", "2", "3"],
    "ID_PAIS": ["1"],
    "NDUPLIC_N": ["0", "1", "9"],
    "IN_VINCULA": ["0", "1", "9"],
    "MIGRADO_W": ["0", "9"],
    "ID_OCUPA_N": ["999992", "622005", "517330"],
    "TRATAMENTO": ["1", "2", "3", "4", "5"],
    "INSTITUCIO": ["1", "2", "3", "9"],
    "RAIOX_TORA": ["1", "2", "3", "4", "9"],
    "TESTE_TUBE": ["1", "2", "3", "9"],
    "FORMA": ["1", "2", "3"],
    "EXTRAPU1_N": ["1", "2", "3", "9"],
    "EXTRAPU2_N": ["1", "2", "3", "9"],
    "EXTRAPUL_O": ["", "PLEURAL", "GANGLIONAR"],
    "AGRAVALCOO": ["1", "2", "9"],
    "AGRAVDIABE": ["1", "2", "9"],
    "AGRAVDOENC": ["1", "2", "9"],
    "AGRAVOUTRA": ["1", "2", "9"],
    "AGRAVDROGA": ["1", "2", "9"],
    "AGRAVTABAC": ["1", "2", "9"],
    "BACILOSC_E": ["1", "2", "3", "4"],
    "BACILOS_E2": ["1", "2", "3", "9"],
    "BACILOSC_O": ["1", "2", "3", "9"],
    "CULTURA_ES": ["1", "2", "3", "4"],
    "CULTURA_OU": ["1", "2", "3", "4", "9"],
    "HISTOPATOL": ["1", "2", "3", "4"],
    "RIFAMPICIN": ["1", "2", "9"],
    "ISONIAZIDA": ["1", "2", "9"],
    "ETAMBUTOL": ["1", "2", "9"],
    "ESTREPTOMI": ["1", "2", "9"],
    "PIRAZINAMI": ["1", "2", "9"],
    "ETIONAMIDA": ["1", "2", "9"],
    "OUTRAS": ["1", "2", "9"],
    "OUTRAS_DES": ["", "LEVOFLOXACINA"],
    "TRAT_SUPER": ["1", "2", "9"],
    "NU_CONTATO": ["0", "1", "2", "5"],
    "DOENCA_TRA": ["1", "2", "9"],
    "SG_UF_AT": ["11", "35"],
    "ID_MUNIC_A": ["1100015", "3500105"],
    "SG_UF_2": ["11", "35"],
    "ID_MUNIC_2": ["1100015", "3500105"],
    "BACILOSC_1": ["1", "2", "3", "4"],
    "BACILOSC_2": ["1", "2", "3", "4"],
    "BACILOSC_3": ["1", "2", "3", "4"],
    "BACILOSC_4": ["1", "2", "3", "4"],
    "BACILOSC_5": ["1", "2", "3", "4"],
    "BACILOSC_6": ["1", "2", "3", "4"],
    "TRATSUP_AT": ["1", "2", "9"],
    "NU_COMU_EX": ["0", "1", "3"],
    "SITUA_9_M": ["1", "2", "3"],
    "SITUA_12_M": ["1", "2", "3"],
    "TPUNINOT": ["1", "2", "3"],
    "POP_LIBER": ["1", "2", "9"],
    "POP_RUA": ["1", "2", "9"],
    "POP_SAUDE": ["1", "2", "9"],
    "POP_IMIG": ["1", "2", "9"],
    "BENEF_GOV": ["1", "2", "9"],
    "TEST_MOLEC": ["1", "2", "3", "4", "5"],
    "TEST_SENSI": ["1", "2", "3", "4"],
    "BAC_APOS_6": ["1", "2", "3", "4"],
    "TRANSF": ["0", "1", "2", "9"],
    "UF_TRANSF": ["11", "35"],
    "MUN_TRANSF": ["1100015", "3500105"],
    "CS_FLXRET": [""],
    "FLXRECEBI": [""],
}

DATAS_SINAN = [
    "DT_DIGITA",
    "DT_TRANSUS",
    "DT_TRANSDM",
    "DT_TRANSSM",
    "DT_TRANSRM",
    "DT_TRANSRS",
    "DT_TRANSSE",
    "DT_INIC_TR",
    "DT_NOTI_AT",
    "DT_MUDANCA",
]


def _completa_schema_sinan(df: pd.DataFrame, rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Preenche as colunas do SINAN que os campos significativos não cobrem.

    Sem isto a fixture teria menos colunas que a coorte real e o portão de
    completude falharia — corretamente.
    """
    total = len(df)
    for col, dominio in DOMINIOS_SINAN.items():
        if col not in df.columns:
            df[col] = rng.choice(dominio, size=total)
    for col in DATAS_SINAN:
        if col not in df.columns:
            df[col] = df["DT_NOTIFIC"]
    return df


# Indicadores contextuais reais. A fixture precisa reproduzir o schema inteiro
# (spec §5); com menos colunas, o portão de completude falha — corretamente.
IBGE_NUMERICOS = [
    "IBGE_PESSOAL_OCUPADO",
    "IBGE_SALARIO_MEDIO_MENSAL",
    "IBGE_ESTAB_DE_SAUDE_SUS",
    "IBGE_MORTALIDADE_INFANTIL",
    "IBGE_NUMERO_DE_ESTAB_DE_ENSINO_FUNDAMENTAL",
    "IBGE_NUMERO_DE_ESTAB_DE_ENSINO_MEDIO",
    "IBGE_ARBORIZACAO_DE_VIAS_PUBLICAS",
    "IBGE_ESGOTAMENTO_SANITARIO_ADEQUADO",
    "IBGE_URBANIZACAO_DE_VIAS_PUBLICAS",
    "IBGE_POPULACAO_OCUPADA",
    "IBGE_POPULACAO_C_RENDIMENTO_1",
    "IBGE_TAXA_DE_ESCOLARIZACAO_6",
    "IBGE_IDEB_ANOS_INICIAIS_REDE",
    "IBGE_IDEB_ANOS_FINAIS_REDE",
    "IBGE_AREA_URBANIZADA",
    "IBGE_DENSIDADE_DEMOGRAFICA",
]

IPEA_INDICADORES = [
    "IPEA_IDHM",
    "IPEA_PCT_ADENSAMENTO_DOMICILIAR",
    "IPEA_PCT_AGUA_ESGOTO_INADEQUADO",
    "IPEA_PCT_VULNERAVEIS_POBREZA",
    "IPEA_PCT_EXTREMAMENTE_POBRES",
    "IPEA_INDICE_GINI",
    "IPEA_RENDA_PER_CAPITA",
    "IPEA_TAXA_ANALFABETISMO_15MAIS",
    "IPEA_TAXA_DESEMPREGO_18MAIS",
    "IPEA_ESPERANCA_VIDA",
    "IPEA_MORTALIDADE_ATE_1ANO",
    "IPEA_PCT_PAREDE_INADEQUADA",
    "IPEA_PCT_AGUA_ENCANADA",
    "IPEA_PCT_COLETA_LIXO",
    "IPEA_PCT_TRABALHO_FORMAL",
    "IPEA_RAZAO_DEPENDENCIA",
    "IPEA_TAXA_ENVELHECIMENTO",
    "IPEA_POPULACAO_TOTAL",
    "IPEA_INDICE_THEIL",
]


def _contexto(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    linhas_ibge: list[dict] = []
    linhas_ipea: list[dict] = []
    prof: list[dict] = []
    estab: list[dict] = []
    for m in MUNICIPIOS:
        mid = m["ID_MUNICIP"]
        for ano in ANOS:
            linha_ibge = {
                "ID_MUNICIP": int(mid),
                "ANO_DIAG": ano,
                "IBGE_PIB_PER_CAPITA": f"{rng.uniform(8000, 40000):.2f}",
                # Sentinela textual do IBGE.
                "IBGE_INTERNACOES_POR_DIARREIA_SUS": (
                    "-" if rng.random() < 0.1 else f"{rng.uniform(1, 100):.1f}"
                ),
                "IBGE_BIOMA": "Cerrado",  # categórica: deve continuar texto
                "IBGE_SISTEMA_COSTEIRO_MARINHO": "Não",
            }
            for c in IBGE_NUMERICOS:
                linha_ibge[c] = round(float(rng.uniform(1, 1000)), 2)
            linhas_ibge.append(linha_ibge)
            # Município novo só tem Censo 2022: antes disso, ausente.
            ausente = mid == MUNICIPIO_NOVO and ano < 2022
            censo = None if ausente else (2010 if ano < 2022 else 2022)
            linha_ipea = {"ID_MUNICIP": int(mid), "ANO_DIAG": ano}
            for c in IPEA_INDICADORES:
                linha_ipea[c] = None if ausente else round(float(rng.uniform(0.1, 100)), 3)
                linha_ipea[f"{c}_ANO_CENSO"] = censo
            linhas_ipea.append(linha_ipea)
            for mes in range(1, 13):
                anomes = f"{ano}-{mes:02d}"
                prof.append(
                    {
                        "ID_MUNICIP": mid[:6],
                        "data": anomes,
                        "QT_PROFISSIONAIS_TOTAL": float(rng.integers(10, 5000)),
                    }
                )
                estab.append(
                    {
                        "ID_MUNICIP": mid[:6],
                        "data": anomes,
                        "QT_ESTABELECIMENTOS_TOTAL": float(rng.integers(1, 300)),
                    }
                )
    return (
        pd.DataFrame(linhas_ibge),
        pd.DataFrame(linhas_ipea),
        pd.DataFrame(prof),
        pd.DataFrame(estab),
    )


def make_synthetic_raw(n: int = 2000, seed: int = 42) -> SyntheticRaw:
    """Gera o conjunto bruto sintético completo, determinístico por semente."""
    rng = np.random.default_rng(seed)
    sinan = _sinan(rng, n)
    ibge, ipea, prof, estab = _contexto(rng)
    municipios = pd.DataFrame(MUNICIPIOS)
    latlong = pd.DataFrame(
        {
            "GEOCODIGO_MUNICIPIO": [int(m["ID_MUNICIP"]) for m in MUNICIPIOS],
            "LATITUDE": [-8.76, -3.71, -23.55, -25.42, -15.60],
            "LONGITUDE": [-63.90, -38.54, -46.63, -49.27, -56.10],
        }
    )
    estados = pd.DataFrame(ESTADOS)
    return SyntheticRaw(sinan, municipios, latlong, estados, ibge, ipea, prof, estab)
