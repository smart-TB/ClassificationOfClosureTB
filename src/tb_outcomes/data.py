"""Harmonização do SINAN-TB com contexto municipal (IBGE/IPEA/CNES).

Port fiel das etapas 8–14 de bot_data_SINNAN_IBGE_CNES.py, que permanece
intacto e privado. O teste de regressão em tests/regression/ exige que este
módulo reproduza exatamente o parquet do bot; qualquer melhoria é mudança
deliberada, com o teste atualizado no mesmo commit.

Invariantes que este módulo mantém:
  - ausência nunca vira zero (Int64/<NA>);
  - toda geografia deriva de ID_MUNIC_ANALISE;
  - nenhum valor futuro (ffill sim, bfill não).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Fora do intervalo plausível para uma coorte do SINAN-TB. Datas fora disto são
# REPORTADAS, nunca excluídas aqui: exclusão é decisão de coorte (briefing §9.1).
DATA_MINIMA_PLAUSIVEL = pd.Timestamp("2000-01-01")
DATA_MAXIMA_PLAUSIVEL = pd.Timestamp("2030-12-31")

COLUNAS_DATA = ["DT_NOTIFIC", "DT_DIAG", "DT_ENCERRA", "DT_INIC_TR", "DT_DIGITA"]

# Marcadores textuais que representam ausência, não um valor observado.
AUSENTES_TEXTUAIS = {"", "nan", "NaN", "None", "<NA>", "NA", "-"}

# Sentinelas do IBGE que representam ausência, não zero.
SENTINELAS_CONTEXTO = {"", "-", "..", "...", "nan", "NaN", "None", "X"}

COLUNA_MUNICIPIO_RESOLVIDO = "ID_MUNIC_ANALISE"


def normalize_missing(s: pd.Series) -> pd.Series:
    """Aplica strip e converte marcadores textuais de ausência em <NA>."""
    limpa = s.astype("string").str.strip()
    return limpa.mask(limpa.isin(AUSENTES_TEXTUAIS))


def is_all_numeric(s: pd.Series) -> bool:
    """True se TODOS os valores presentes forem numéricos.

    A versão original do bot comparava 'isna() + notna() == len', soma
    trivialmente verdadeira; na prática respondia 'existe algum número aqui?' e
    convertia colunas de texto livre, destruindo o conteúdo.
    """
    presentes = s.dropna()
    if presentes.empty:
        return False
    return bool(pd.to_numeric(presentes, errors="coerce").notna().all())


def coerce_numeric_column(s: pd.Series) -> pd.Series:
    """Int64 quando todos os valores são inteiros; float64 quando há decimal."""
    numerica = pd.to_numeric(s, errors="coerce")
    presentes = numerica.dropna()
    if (presentes % 1 == 0).all():
        return numerica.astype("Int64")
    return numerica.astype("float64")


def treat_sinan(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem preservando ausência. Port de proc08_tratamento_SINNAN_01."""
    df = df.copy()
    df["ANO_DIAG"] = df["DT_DIAG"].astype("string").str[:4]

    for col in df.columns:
        df[col] = normalize_missing(df[col])

    for coluna in df.columns:
        if coluna.startswith("ANO_NASC"):
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").astype("Int64")
            continue
        if coluna.startswith(("DT", "ANO", "NU_IDADE_N")):
            continue
        if is_all_numeric(df[coluna]):
            df[coluna] = coerce_numeric_column(df[coluna])

    return df


def fix_municipality_codes(df: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Completa códigos de 6 dígitos e repara códigos inválidos.

    Port de proc10: mapeia código incompleto -> completo, preenche cada coluna
    inválida com a outra e, em último caso, recorre a ID_MUNIC_A.
    """
    df = df.copy()
    completo = {m[:6]: m for m in municipios["ID_MUNICIP"]}
    validos = set(municipios["ID_MUNICIP"])

    for coluna in ("ID_MUNICIP", "ID_MN_RESI"):
        df[coluna] = (
            df[coluna].astype(str).str.replace(r"\.0$", "", regex=True).replace(completo)
        )

    mask = (df["ID_MUNICIP"].str.strip() == "0") | (~df["ID_MUNICIP"].isin(validos))
    df.loc[mask, "ID_MUNICIP"] = df.loc[mask, "ID_MN_RESI"]
    mask = (df["ID_MN_RESI"].str.strip() == "0") | (~df["ID_MN_RESI"].isin(validos))
    df.loc[mask, "ID_MN_RESI"] = df.loc[mask, "ID_MUNICIP"]

    # Último recurso: município de atendimento, quando ambos ficaram em "0".
    zeros = (df["ID_MUNICIP"] == "0") & (df["ID_MN_RESI"] == "0")
    if "ID_MUNIC_A" in df.columns and zeros.any():
        atendimento = df.loc[zeros, "ID_MUNIC_A"].astype(str).str.strip()
        recuperavel = ~atendimento.isin(["", "0", "nan", "None", "<NA>"])
        alvos = atendimento[recuperavel]
        df.loc[alvos.index, "ID_MUNICIP"] = alvos
        df.loc[alvos.index, "ID_MN_RESI"] = alvos
        logger.info("%d registros recuperados via ID_MUNIC_A.", len(alvos))

    return df


def resolve_municipality(
    df: pd.DataFrame, key: str, fallback: str, municipios: pd.DataFrame
) -> pd.DataFrame:
    """Resolve, de uma vez, o município que governa TODA a geografia da análise.

    É o ponto único onde a geografia é decidida. Antes, coordenadas vinham da
    residência e indicadores da notificação; como divergem em ~10% dos casos,
    ~130 mil registros tinham contexto de um município diferente do que definia
    seu cluster espacial, quebrando o bloqueio espacial.
    """
    df = df.copy()
    validos = set(municipios["ID_MUNICIP"])
    principal = df[key].where(df[key].isin(validos))
    reserva = df[fallback].where(df[fallback].isin(validos))
    df[COLUNA_MUNICIPIO_RESOLVIDO] = principal.fillna(reserva)

    sem_municipio = int(df[COLUNA_MUNICIPIO_RESOLVIDO].isna().sum())
    if sem_municipio:
        logger.warning("%d registros sem município válido.", sem_municipio)
    return df


def attach_names(df: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """NOME_MUNIC/SIGLA_UF/REGIAO a partir da coluna resolvida."""
    df = df.copy()
    for coluna in ("NOME_MUNIC", "SIGLA_UF", "REGIAO"):
        mapa = municipios.set_index("ID_MUNICIP")[coluna]
        df[coluna] = df[COLUNA_MUNICIPIO_RESOLVIDO].map(mapa)
    return df


def attach_coordinates(df: pd.DataFrame, latlong: pd.DataFrame) -> pd.DataFrame:
    """LAT_MUNIC/LONG_MUNIC a partir da coluna resolvida."""
    df = df.copy()
    cod = pd.to_numeric(df[COLUNA_MUNICIPIO_RESOLVIDO], errors="coerce").astype("Int64")
    df["LAT_MUNIC"] = cod.map(latlong.set_index("GEOCODIGO_MUNICIPIO")["LATITUDE"])
    df["LONG_MUNIC"] = cod.map(latlong.set_index("GEOCODIGO_MUNICIPIO")["LONGITUDE"])

    sem_coord = int(df["LAT_MUNIC"].isna().sum())
    if sem_coord:
        logger.warning("%d registros sem coordenada municipal.", sem_coord)
    return df


CENTROIDES_REGIAO = {
    "Norte": {"latitude": -3.4168, "longitude": -62.2159},
    "Nordeste": {"latitude": -8.4703, "longitude": -37.9980},
    "Centro-Oeste": {"latitude": -15.5989, "longitude": -52.5625},
    "Sudeste": {"latitude": -20.3175, "longitude": -44.3968},
    "Sul": {"latitude": -27.5712, "longitude": -51.8339},
}


def load_estados(path: Path) -> pd.DataFrame:
    """Lê data/estados.json: código da UF, lat/long e região."""
    import json

    with Path(path).open("r", encoding="utf-8-sig") as fh:
        dados = json.load(fh)
    return pd.DataFrame(
        [
            {
                "codigo_uf": item["codigo_uf"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "regiao": item["regiao"],
            }
            for item in dados
        ]
    )


def attach_state_coordinates(df: pd.DataFrame, estados: pd.DataFrame) -> pd.DataFrame:
    """LAT_UF/LONG_UF a partir de SG_UF_NOT (UF de NOTIFICAÇÃO).

    ATENÇÃO — incoerência conhecida, preservada por fidelidade ao pipeline de
    aquisição. Ver a nota em attach_region_coordinates.
    """
    df = df.copy()
    cod_uf = pd.to_numeric(df["SG_UF_NOT"], errors="coerce").astype("Int64")
    df["LAT_UF"] = cod_uf.map(estados.set_index("codigo_uf")["latitude"])
    df["LONG_UF"] = cod_uf.map(estados.set_index("codigo_uf")["longitude"])
    return df


def attach_region_coordinates(df: pd.DataFrame, estados: pd.DataFrame) -> pd.DataFrame:
    """LAT_REG/LONG_REG a partir do centróide da região de SG_UF_NOT.

    ATENÇÃO — incoerência conhecida, preservada por fidelidade ao pipeline de
    aquisição: LAT_UF, LONG_UF, LAT_REG e LONG_REG derivam da UF de NOTIFICAÇÃO
    (SG_UF_NOT), enquanto NOME_MUNIC/SIGLA_UF/REGIAO e as coordenadas municipais
    derivam do município de análise (residência, por padrão). Num registro cuja
    residência e notificação estão em UFs diferentes, REGIAO e LAT_REG discordam.

    São centróides grosseiros, e o cluster espacial usa LAT_MUNIC/LONG_MUNIC —
    então o impacto é menor que o do defeito municipal já corrigido. Ainda assim
    é a mesma classe de defeito. Unificá-las quebra a equivalência com o pipeline
    de aquisição e exige atualizar o teste de regressão no mesmo commit: é
    decisão do PI, não do código.
    """
    df = df.copy()
    cod_uf = pd.to_numeric(df["SG_UF_NOT"], errors="coerce").astype("Int64")
    regiao = cod_uf.map(estados.set_index("codigo_uf")["regiao"])
    df["LAT_REG"] = regiao.map({k: v["latitude"] for k, v in CENTROIDES_REGIAO.items()})
    df["LONG_REG"] = regiao.map({k: v["longitude"] for k, v in CENTROIDES_REGIAO.items()})
    return df


def municipality_audit(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Auditoria da chave municipal (briefing §8.5)."""
    total = len(df)
    sem = int(df[COLUNA_MUNICIPIO_RESOLVIDO].isna().sum())
    divergentes = int((df["ID_MN_RESI"] != df["ID_MUNICIP"]).sum())
    return pd.DataFrame(
        [
            {"metrica": "registros_total", "valor": total},
            {"metrica": "chave_municipio_analise", "valor": key},
            {"metrica": "registros_com_municipio_valido", "valor": total - sem},
            {"metrica": "registros_sem_municipio_valido", "valor": sem},
            {"metrica": "registros_residencia_diferente_notificacao", "valor": divergentes},
        ]
    )


def attach_cnes(df: pd.DataFrame, prof: pd.DataFrame, estab: pd.DataFrame) -> pd.DataFrame:
    """Junta CNES por (município de análise, ano-mês) e propaga só o passado.

    O bot original usava ffill().bfill(); o bfill dava a um município a
    capacidade instalada de um mês FUTURO. Aqui há apenas ffill: meses
    anteriores ao primeiro registro permanecem ausentes, que é a descrição
    honesta do que se sabe (briefing §8.4).
    """
    df = df.copy()
    prof = prof.rename(columns={"ID_MUNICIP": "ID_MUNICIP_6", "data": "ANOMES_DIAG"})
    estab = estab.rename(columns={"ID_MUNICIP": "ID_MUNICIP_6", "data": "ANOMES_DIAG"})

    df["ID_MUNICIP_6"] = df[COLUNA_MUNICIPIO_RESOLVIDO].astype(str).str[:6]
    df["ANOMES_DIAG"] = df["DT_DIAG"].astype(str).str[:7]

    df = df.merge(prof, on=["ID_MUNICIP_6", "ANOMES_DIAG"], how="left")
    df = df.merge(estab, on=["ID_MUNICIP_6", "ANOMES_DIAG"], how="left")

    df = df.sort_values(by=[COLUNA_MUNICIPIO_RESOLVIDO, "ANOMES_DIAG"])
    for coluna in ("QT_ESTABELECIMENTOS_TOTAL", "QT_PROFISSIONAIS_TOTAL"):
        antes = int(df[coluna].isna().sum())
        df[coluna] = df.groupby(COLUNA_MUNICIPIO_RESOLVIDO)[coluna].transform(
            lambda g: g.ffill()
        )
        logger.info(
            "%s: %d ausentes após merge, %d após ffill.",
            coluna,
            antes,
            int(df[coluna].isna().sum()),
        )

    return df.drop(columns=["ID_MUNICIP_6", "ANOMES_DIAG"])


def attach_ibge_ipea(
    df: pd.DataFrame, ibge: pd.DataFrame, ipea: pd.DataFrame
) -> pd.DataFrame:
    """Junta IBGE e IPEA por (município de análise, ano) e tipa o contexto.

    validate='m:1' faz o merge FALHAR se a fonte tiver mais de uma linha por
    (município, ano), em vez de inflar a coorte em silêncio.
    """
    df = df.copy()
    df[COLUNA_MUNICIPIO_RESOLVIDO] = pd.to_numeric(
        df[COLUNA_MUNICIPIO_RESOLVIDO], errors="coerce"
    ).astype("Int64")
    df["ANO_DIAG"] = pd.to_numeric(df["ANO_DIAG"], errors="coerce").astype("Int64")

    for fonte_original in (ibge, ipea):
        fonte = fonte_original.rename(columns={"ID_MUNICIP": COLUNA_MUNICIPIO_RESOLVIDO})
        fonte[COLUNA_MUNICIPIO_RESOLVIDO] = pd.to_numeric(
            fonte[COLUNA_MUNICIPIO_RESOLVIDO], errors="coerce"
        ).astype("Int64")
        fonte["ANO_DIAG"] = pd.to_numeric(fonte["ANO_DIAG"], errors="coerce").astype("Int64")
        df = df.merge(
            fonte, on=[COLUNA_MUNICIPIO_RESOLVIDO, "ANO_DIAG"], how="left", validate="m:1"
        )

    convertidas = []
    for col in df.columns:
        if not col.startswith(("IBGE_", "IPEA_")):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        limpa = df[col].astype("string").str.strip()
        limpa = limpa.mask(limpa.isin(SENTINELAS_CONTEXTO))
        presentes = limpa.dropna()
        if presentes.empty:
            # Coluna inteiramente sentinela ('-', '..'): é ausência, não texto.
            #
            # DESVIO DECLARADO em relação ao bot, que aqui dava `continue` e
            # deixava o '-' sobreviver como valor literal — entrando na análise
            # como se fosse uma categoria. Não afeta a equivalência: no snapshot
            # atual nenhuma coluna contextual é inteiramente sentinela (as únicas
            # que permanecem texto, IBGE_BIOMA e IBGE_SISTEMA_COSTEIRO_MARINHO,
            # são categóricas legítimas e têm zero sentinelas).
            df[col] = limpa
            continue
        if pd.to_numeric(presentes, errors="coerce").notna().all():
            df[col] = pd.to_numeric(limpa, errors="coerce")
            convertidas.append(col)

    logger.info(
        "Colunas contextuais convertidas para numérico (%d): %s", len(convertidas), convertidas
    )
    return df


@dataclass
class RawInputs:
    """Artefatos brutos entregues pelo pipeline de aquisição (privado)."""

    sinan: pd.DataFrame
    municipios: pd.DataFrame
    municipios_latlong: pd.DataFrame
    estados: pd.DataFrame
    ibge: pd.DataFrame
    ipea: pd.DataFrame
    cnes_prof: pd.DataFrame
    cnes_estab: pd.DataFrame


def load_raw_inputs(data_dir: Path) -> RawInputs:
    """Lê os artefatos brutos (etapas 1–7) mais o snapshot de municípios."""
    d = Path(data_dir)
    return RawInputs(
        sinan=pd.read_parquet(d / "sinnan.parquet"),
        municipios=pd.read_csv(d / "municipios_ibge.csv", dtype={"ID_MUNICIP": str}),
        municipios_latlong=pd.read_csv(d / "municipios_lat_long.csv"),
        estados=load_estados(d / "estados.json"),
        ibge=pd.read_csv(d / "indicadores_IBGE.csv", low_memory=False),
        ipea=pd.read_csv(d / "indicadores_IPEA.csv", low_memory=False),
        cnes_prof=pd.read_parquet(d / "profissionais_CNES.parquet"),
        cnes_estab=pd.read_parquet(d / "estabelecimentos_CNES.parquet"),
    )


def raw_manifest(data_dir: Path) -> pd.DataFrame:
    """Inventário dos brutos: caminho, SHA-256 e tamanho (briefing §8.2).

    Como o pipeline de aquisição não é publicado, este manifesto é o que permite
    a um terceiro saber QUAIS arquivos obter do DATASUS e com que checksum.
    """
    from tb_outcomes.provenance import sha256_file

    arquivos = [
        "sinnan.parquet",
        "municipios_ibge.csv",
        "municipios_lat_long.csv",
        "indicadores_IBGE.csv",
        "indicadores_IPEA.csv",
        "profissionais_CNES.parquet",
        "estabelecimentos_CNES.parquet",
    ]
    linhas = []
    for nome in arquivos:
        p = Path(data_dir) / nome
        if not p.exists():
            raise FileNotFoundError(f"artefato bruto ausente: {p}")
        linhas.append(
            {
                "file": nome,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "source": "DATASUS/IBGE/IPEA/CNES via pipeline de aquisição (privado)",
            }
        )
    return pd.DataFrame(linhas)


def harmonize(raw: RawInputs, cfg) -> pd.DataFrame:
    """Pipeline de harmonização: proc08 → proc14, na ordem do bot."""
    logger.info("Harmonizando %d registros brutos.", len(raw.sinan))
    df = treat_sinan(raw.sinan)
    df = fix_municipality_codes(df, raw.municipios)
    df = resolve_municipality(
        df, cfg.geography.municipality_key, cfg.geography.fallback_key, raw.municipios
    )
    df = attach_names(df, raw.municipios)
    # Ordem idêntica à do proc12 do bot: UF, município, região. A ordem das
    # colunas no parquet final depende disto, e o teste de equivalência a compara.
    df = attach_state_coordinates(df, raw.estados)
    df = attach_coordinates(df, raw.municipios_latlong)
    df = attach_region_coordinates(df, raw.estados)
    df = attach_cnes(df, raw.cnes_prof, raw.cnes_estab)
    df = attach_ibge_ipea(df, raw.ibge, raw.ipea)
    logger.info("Harmonização concluída: %d linhas, %d colunas.", len(df), df.shape[1])
    return df


# Versão da lógica de harmonização. Compõe a chave do cache: se este módulo mudar
# de forma que altere a saída, INCREMENTE aqui para invalidar caches antigos. É o
# mesmo mecanismo declarado do git — o cache não detecta mudança de código sozinho,
# então a versão é a declaração explícita de que a lógica mudou.
HARMONIZATION_VERSION = 1

_CACHE_PARQUET = "harmonized.parquet"
_CACHE_META = "harmonized_meta.json"

# Arquivos brutos cuja mudança deve invalidar o cache.
_ARQUIVOS_BRUTOS = [
    "sinnan.parquet",
    "municipios_ibge.csv",
    "municipios_lat_long.csv",
    "estados.json",
    "indicadores_IBGE.csv",
    "indicadores_IPEA.csv",
    "profissionais_CNES.parquet",
    "estabelecimentos_CNES.parquet",
]


def sinan_column_names(data_dir: Path) -> list[str]:
    """Nomes das colunas do SINAN bruto, lidos do schema (sem carregar os dados)."""
    import pyarrow.parquet as pq

    return list(pq.read_schema(Path(data_dir) / "sinnan.parquet").names)


def harmonization_key(data_dir: Path, cfg) -> str:
    """Chave que identifica esta harmonização.

    Muda quando muda: qualquer arquivo bruto (SHA-256), a chave de geografia da
    config, ou a versão da lógica. É o que torna o cache seguro — servir um
    parquet velho após o bruto mudar seria pior que não ter cache.
    """
    from tb_outcomes.provenance import sha256_file

    partes = [f"v{HARMONIZATION_VERSION}", f"geo={cfg.geography.municipality_key}"]
    d = Path(data_dir)
    for nome in _ARQUIVOS_BRUTOS:
        p = d / nome
        partes.append(f"{nome}={sha256_file(p) if p.exists() else 'MISSING'}")
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:16]


def load_or_build_harmonized(data_dir: Path, cfg, *, rebuild: bool = False) -> tuple[pd.DataFrame, bool]:
    """Devolve a coorte harmonizada, do cache se fresco, senão reconstrói.

    Retorna (df, foi_construido). O cache é validado pela chave: se a gravada não
    bater com a atual, o parquet é ignorado e refeito. Evita re-harmonizar 1,3 M
    de registros (~8 min) a cada comando.
    """
    import json

    d = Path(data_dir)
    parquet = d / _CACHE_PARQUET
    meta = d / _CACHE_META
    chave = harmonization_key(d, cfg)

    if not rebuild and parquet.exists() and meta.exists():
        try:
            gravada = json.loads(meta.read_text(encoding="utf-8")).get("key")
        except (json.JSONDecodeError, OSError):
            gravada = None
        if gravada == chave:
            logger.info("Coorte harmonizada reusada do cache (%s).", parquet)
            return pd.read_parquet(parquet), False
        logger.info("Cache de harmonização obsoleto (chave mudou); reconstruindo.")

    df = harmonize(load_raw_inputs(d), cfg)
    from tb_outcomes.artifacts import write_parquet

    write_parquet(df, parquet)
    meta_obj = {
        "key": chave,
        "harmonization_version": HARMONIZATION_VERSION,
        "municipality_key": cfg.geography.municipality_key,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
    }
    from tb_outcomes.artifacts import write_json

    write_json(meta_obj, meta)
    logger.info("Coorte harmonizada materializada em %s (%d linhas).", parquet, len(df))
    return df, True


def quality_report(df: pd.DataFrame) -> dict:
    """Relatório de qualidade. Reporta, não exclui."""
    datas_impossiveis = {}
    for col in COLUNAS_DATA:
        if col not in df.columns:
            continue
        d = pd.to_datetime(df[col], errors="coerce")
        fora = ((d < DATA_MINIMA_PLAUSIVEL) | (d > DATA_MAXIMA_PLAUSIVEL)) & d.notna()
        datas_impossiveis[col] = int(fora.sum())

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "residence_notification_divergence": int((df["ID_MN_RESI"] != df["ID_MUNICIP"]).sum()),
        "rows_without_municipality": int(df[COLUNA_MUNICIPIO_RESOLVIDO].isna().sum()),
        "impossible_dates": datas_impossiveis,
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }


def quality_report_markdown(rep: dict) -> str:
    """Versão legível do relatório de qualidade (briefing §23.2)."""
    linhas = [
        "# Relatório de qualidade dos dados",
        "",
        f"- Linhas: {rep['n_rows']:,}",
        f"- Colunas: {rep['n_columns']}",
        f"- Residência != notificação: {rep['residence_notification_divergence']:,}",
        f"- Sem município válido: {rep['rows_without_municipality']:,}",
        "",
        "## Datas fora do intervalo plausível",
        "",
        "Reportadas, não excluídas: exclusão é decisão de coorte (briefing §9.1).",
        "",
        "| Coluna | N |",
        "|---|---:|",
    ]
    linhas += [f"| {c} | {n:,} |" for c, n in rep["impossible_dates"].items()]
    linhas += ["", "## Ausência por coluna", "", "| Coluna | N ausente |", "|---|---:|"]
    linhas += [
        f"| {c} | {n:,} |"
        for c, n in sorted(rep["missing_by_column"].items(), key=lambda kv: -kv[1])
        if n > 0
    ]
    return "\n".join(linhas) + "\n"
