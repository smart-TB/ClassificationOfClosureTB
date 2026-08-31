"""Derivação de features e auditoria de ausência.

Toda feature nasce aqui, a partir de configs/features.yaml, que classifica cada
coluna da coorte harmonizada pelo momento em que a informação existe. A
classificação é ancorada na descrição do próprio dicionário oficial (SINAN NET
5.0), não em suposição: cada variável carrega a citação que a justifica.

O momento de predição é a NOTIFICAÇÃO, conjunto único. Nada que só exista depois
entra no X — nem exame cujo resultado demora, nem campo de seguimento, nem o
encerramento.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class FeatureConfigError(Exception):
    """A configuração de features não cobre a coorte, ou deixa o desfecho entrar."""


# Colunas que SÃO o desfecho, ou das quais o desfecho é trivialmente recuperável.
# Nenhuma feature do X pode derivar de qualquer uma delas — uma feature construída
# a partir da data de encerramento (p.ex. tempo até o desfecho) codifica o próprio
# alvo e produz vazamento. Este conjunto é a fronteira que o verifica.
COLUNAS_DE_DESFECHO = frozenset(
    {"SITUA_ENCE", "DT_ENCERRA", "TRANSF", "UF_TRANSF", "MUN_TRANSF"}
)


class Availability:
    """Momento em que a informação passa a existir."""

    AT_NOTIFICATION = "at_notification"
    POST_BASELINE = "post_baseline"  # existe, mas só depois da notificação
    FOLLOWUP = "followup"  # preenchido durante o acompanhamento
    OUTCOME = "outcome"  # é o desfecho, ou deriva dele
    INTERNAL = "internal"  # controle do sistema, não é informação clínica
    UNKNOWN = "unknown"  # indefinido: bloqueia o conjunto primário

    ELEGIVEIS = {AT_NOTIFICATION}


class FeatureSpec(BaseModel):
    raw_name: str
    harmonized_name: str
    form_field: int | None = None
    type: str
    source: str
    availability: str
    dictionary_evidence: str = ""
    in_notification_set: bool
    # Colunas de origem. Vazio = coluna bruta, que não deriva de nada.
    # Toda derivada DEVE declarar sua procedência: é o que permite verificar a
    # cadeia em vez de confiar numa lista de nomes a remover.
    derived_from: list[str] = []
    # O código 0 desta coluna é ausência, não categoria.
    #
    # Até 2017 o SINAN gravava "não preenchido" como 0; a partir da v5 da ficha,
    # como campo vazio. O 0 não consta no dicionário destes campos e é quase
    # perfeitamente colinear com "notificado antes de 2018" — sem este mapeamento,
    # o modelo aprenderia a data da notificação disfarçada de característica clínica.
    zero_means_missing: bool = False
    sensitivity_only: str | None = None
    protected: bool = False
    contextual: bool = False
    clinical_review: str = "pending"
    clinical_approved_by: str | None = None
    clinical_approved_date: str | None = None

    @model_validator(mode="after")
    def _valida(self) -> FeatureSpec:
        if self.in_notification_set and self.availability not in Availability.ELEGIVEIS:
            raise ValueError(
                f"{self.raw_name}: availability='{self.availability}' não pode estar no "
                f"conjunto primário. Só '{Availability.AT_NOTIFICATION}' entra."
            )
        if self.in_notification_set:
            do_desfecho = COLUNAS_DE_DESFECHO & ({self.raw_name} | set(self.derived_from))
            if do_desfecho:
                raise ValueError(
                    f"{self.raw_name}: deriva do desfecho ({sorted(do_desfecho)}) e não pode "
                    f"estar no X — uma feature construída a partir do desfecho o vaza para "
                    f"dentro do modelo."
                )
        if not self.dictionary_evidence and self.clinical_review != "pending":
            raise ValueError(
                f"{self.raw_name}: dictionary_evidence vazio exige clinical_review='pending'."
            )
        return self


def load_feature_config(path: Path) -> list[FeatureSpec]:
    with Path(path).open("r", encoding="utf-8") as fh:
        bruto = yaml.safe_load(fh)
    return [FeatureSpec.model_validate(item) for item in bruto["features"]]


def assert_covers_all_columns(specs: list[FeatureSpec], columns: list[str]) -> None:
    """Falha se alguma coluna da coorte não estiver classificada, ou vice-versa.

    É o portão que impede uma variável de ser esquecida em silêncio — o modo mais
    fácil de um campo pós-baseline entrar sem ninguém decidir que entraria.
    """
    declaradas = {s.raw_name for s in specs}
    presentes = set(columns)

    faltando = sorted(presentes - declaradas)
    sobrando = sorted(declaradas - presentes)

    problemas = []
    if faltando:
        problemas.append(f"colunas da coorte sem classificação ({len(faltando)}): {faltando}")
    if sobrando:
        problemas.append(
            f"classificações sem coluna correspondente ({len(sobrando)}): {sobrando}"
        )
    if problemas:
        raise FeatureConfigError("; ".join(problemas))


def assert_no_outcome_in_derivation(specs: list[FeatureSpec]) -> None:
    """Falha se alguma feature do X derivar do desfecho.

    Segunda barreira, redundante com o validador do FeatureSpec — de propósito. O
    validador roda na construção; este roda sobre a config carregada e sobre
    qualquer lista montada em runtime, inclusive depois de alguém mexer num campo.

    Redundância aqui é barata. Uma lista de nomes a remover falha em silêncio (um
    typo, um esquecimento); declarar a procedência de cada feature e verificar a
    cadeia inverte o ônus — o default passa a ser bloquear, e incluir exige
    justificar.
    """
    vazando = {}
    for s in specs:
        if not s.in_notification_set:
            continue
        do_desfecho = COLUNAS_DE_DESFECHO & ({s.raw_name} | set(s.derived_from))
        if do_desfecho:
            vazando[s.raw_name] = sorted(do_desfecho)
    if vazando:
        raise FeatureConfigError(
            f"features do conjunto primário derivam do desfecho: {vazando}. "
            f"Nenhuma coluna derivada de {sorted(COLUNAS_DE_DESFECHO)} pode entrar no X."
        )


# ---------------------------------------------------------------------------
# Auditoria do tempo até o encerramento
# ---------------------------------------------------------------------------

DURACAO_MAXIMA_PLAUSIVEL_DIAS = 3650


def closure_timing_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo da notificação ao encerramento, por desfecho.

    Sustenta a rejeição da janela de 30 dias: 67,2% dos óbitos por TB encerram em
    menos de 30 dias (mediana 13), contra 2,7% das curas. Uma feature que só
    existe quando o exame teve tempo de voltar seria um indicador de
    sobrevivência ao primeiro mês — quase o próprio desfecho.
    """
    notif = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    encerr = pd.to_datetime(df["DT_ENCERRA"], errors="coerce")
    dias = (encerr - notif).dt.days

    valido = dias.notna() & (dias >= 0) & (dias < DURACAO_MAXIMA_PLAUSIVEL_DIAS)
    d = pd.DataFrame({"SITUA_ENCE": df["SITUA_ENCE"], "dias": dias})[valido]

    linhas = []
    for codigo, grupo in d.groupby("SITUA_ENCE"):
        linhas.append(
            {
                "SITUA_ENCE": codigo,
                "n": len(grupo),
                "median_days": float(grupo.dias.median()),
                "pct_closed_lt_30d": round(float((grupo.dias < 30).mean() * 100), 1),
                "pct_closed_lt_60d": round(float((grupo.dias < 60).mean() * 100), 1),
                "pct_closed_lt_90d": round(float((grupo.dias < 90).mean() * 100), 1),
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# HIV: regra derivada e colapso dos níveis que o desfecho reescreve
# ---------------------------------------------------------------------------

# Códigos do campo 40 (HIV) e 43 (Cultura), dicionário SINAN NET 5.0.
HIV_POSITIVO = 1
HIV_NEGATIVO = 2
HIV_EM_ANDAMENTO = 3
HIV_NAO_REALIZADO = 4
AIDS_SIM = 1

# Nível único para "não disponível na notificação". Recebe tanto 'Em andamento'
# quanto 'Não realizado', que a regra do campo 62 torna intercambiáveis.
NIVEL_NAO_DISPONIVEL = HIV_NAO_REALIZADO

# Desfechos que disparam a reescrita 3 -> 4 (dicionário, campo 62).
DESFECHOS_QUE_REESCREVEM = [1, 2, 3, 4, 6, 10]

COLUNAS_CONTAMINADAS = ["HIV", "CULTURA_ES"]

FAIXAS_DURACAO = [(0, 7), (8, 14), (15, 30), (31, 60), (61, 120), (121, 365), (366, 3650)]


def collapse_pending_levels(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Funde 'Em andamento' e 'Não realizado' num nível só.

    O SINAN reescreve 3 -> 4 quando o encerramento é 1, 2, 3, 4, 6 ou 10. Isso faz
    o valor da feature depender do desfecho: 'Em andamento' sobrevive em 4,1% das
    transferências e só em 1,2% das curas. Como a regra é exatamente 3 -> 4,
    tornar f(3) = f(4) neutraliza o efeito.

    O que este colapso NÃO faz: tornar a ausência independente do desfecho. Medido,
    a diferença entre desfechos piora de 2,37 para 6,51 pp — porque a ausência
    restante reflete morte precoce e marginalização de linha de base, não a regra.
    Ver spec §3.1.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].replace({HIV_EM_ANDAMENTO: NIVEL_NAO_DISPONIVEL})
    return df


def derive_hiv_rule(df: pd.DataFrame) -> pd.Series:
    """HIV positivo por sorologia OU AIDS registrada (briefing §7.4).

    Nunca substitui os campos originais: devolve uma Series nova.

    Nota: o próprio SINAN já preenche HIV=1 automaticamente quando AIDS=1 e
    desabilita o campo (dicionário, campo 40). Esta regra alcança apenas o
    resíduo — ~1.183 registros no snapshot atual, de 138.098 com AIDS.

    'Em andamento' e 'Não realizado' NÃO são negativos: viram <NA>.
    """
    hiv = df["HIV"]
    aids = df["AGRAVAIDS"]

    positivo = (hiv == HIV_POSITIVO) | (aids == AIDS_SIM)
    negativo = (hiv == HIV_NEGATIVO) & (aids != AIDS_SIM)

    resultado = pd.Series(pd.NA, index=df.index, dtype="Int64", name="hiv_pos_model")
    resultado[negativo] = 0
    resultado[positivo] = 1
    return resultado


def hiv_rule_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Quanto a regra derivada muda (briefing §7.4)."""
    derivada = derive_hiv_rule(df)
    sorologia_pos = int((df["HIV"] == HIV_POSITIVO).sum())
    apos = int((derivada == 1).sum())
    alterados = int(((derivada == 1) & (df["HIV"] != HIV_POSITIVO)).sum())
    return pd.DataFrame(
        [
            {"metrica": "positivos_sorologia_original", "valor": sorologia_pos},
            {"metrica": "positivos_apos_regra", "valor": apos},
            {"metrica": "registros_alterados_pela_regra", "valor": alterados},
            {"metrica": "pct_alterados", "valor": round(alterados / max(len(df), 1) * 100, 4)},
        ]
    )


def outcome_contamination_audit(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Taxa de 'Em andamento' por desfecho, antes do colapso.

    Se a taxa for sistematicamente maior nos desfechos fora da lista da regra, a
    feature está carregando informação do desfecho.
    """
    linhas = []
    for col in columns:
        if col not in df.columns:
            continue
        for codigo, grupo in df.groupby(df["SITUA_ENCE"].fillna(-1)):
            linhas.append(
                {
                    "column": col,
                    "SITUA_ENCE": int(codigo),
                    "n": len(grupo),
                    "pct_pending": round(float((grupo[col] == HIV_EM_ANDAMENTO).mean() * 100), 2),
                    "affected_by_rule": int(codigo) in DESFECHOS_QUE_REESCREVEM,
                }
            )
    return pd.DataFrame(linhas)


def hiv_availability_by_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Disponibilidade da sorologia por desfecho e tempo até o encerramento.

    Discriminador entre dois mecanismos de ausência (spec §3.1):
      - artefato de registro: a disponibilidade CRESCE com o tempo (houve tempo de
        testar);
      - linha de base: a diferença entre desfechos persiste A TEMPO IGUAL.

    Medido no snapshot: a tempo igual (0-7 dias), óbito por TB tem 62,8% e cura
    81,3%. Os 18,5 pp que o tempo não explica são linha de base — quem morre já
    chegou menos vinculado ao serviço. É por isso que HIV entra no conjunto
    primário apesar do resíduo de artefato.
    """
    notif = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
    encerr = pd.to_datetime(df["DT_ENCERRA"], errors="coerce")
    dias = (encerr - notif).dt.days

    valido = dias.notna() & (dias >= 0) & (dias < DURACAO_MAXIMA_PLAUSIVEL_DIAS)
    d = pd.DataFrame(
        {
            "SITUA_ENCE": df["SITUA_ENCE"],
            "dias": dias,
            "disponivel": df["HIV"].isin([HIV_POSITIVO, HIV_NEGATIVO]),
        }
    )[valido]

    linhas = []
    for codigo, grupo in d.groupby("SITUA_ENCE"):
        for lo, hi in FAIXAS_DURACAO:
            faixa = grupo[grupo.dias.between(lo, hi)]
            if faixa.empty:
                continue
            linhas.append(
                {
                    "SITUA_ENCE": int(codigo),
                    "duration_bin": f"{lo}-{hi}",
                    "n": len(faixa),
                    "pct_available": round(float(faixa.disponivel.mean() * 100), 1),
                }
            )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Derivações de features do momento da notificação.
#
# Não entram: TEMPO_TRATAMENTO e FLAG_ENCERRA, que derivam de DT_ENCERRA e
# portanto do desfecho (ver COLUNAS_DE_DESFECHO); e ANT_RETRO_BIN, HISTO_*,
# CULT_E_*, TEST_SENSI_*, MULTI_DROGA_RESISTENTE, DIAGNOSTICO_LAB_CONFIRMADO,
# que dependem de exames cujo resultado é posterior à notificação.
# ---------------------------------------------------------------------------

IDADE_MINIMA, IDADE_MAXIMA = 0, 120

# Códigos SINAN dos campos Sim/Não/Ignorado. Além de 9 (Ignorado), a base traz
# 0 e 3 em algumas colunas — o original já os tratava como ausentes.
AGRAVO_SIM, AGRAVO_NAO = 1, 2

POPULACOES_ESPECIAIS = ["POP_LIBER", "POP_RUA", "POP_SAUDE", "POP_IMIG"]

# AGRAVOUTRA fora: é texto livre (valores 'X', digitação não codificada), não um
# campo Sim/Não. O _BIN dele saía 100% vazio.
AGRAVOS = [
    "AGRAVAIDS",
    "AGRAVALCOO",
    "AGRAVDIABE",
    "AGRAVDOENC",
    "AGRAVDROGA",
    "AGRAVTABAC",
]

# Derivadas que SÃO features do X. Os *_BIN dos agravos e populações são
# calculados (as interações dependem deles) mas NÃO entram aqui: o categórico
# bruto correspondente já cobre a informação, preservando o código 9 (ignorado).
DERIVED_FEATURES = [
    "IDADE",
    "POP_ESPECIAIS_BIN",
    "IDADE_X_DIABETES",
    "IDADE_X_AIDS",
]

# ---------------------------------------------------------------------------
# Escopos da ablação territorial (ESPECIFICACAO §3.3)
# ---------------------------------------------------------------------------

FEATURE_SCOPES = ("individual", "municipal", "combined", "clinical_baseline")

# Baseline clínico simples da §3.8: idade + HIV + álcool/drogas + situação de rua. É o
# comparador que decide se o modelo de 72 variáveis vale a pena — um serviço não precisa de
# aprendizado de máquina para montar uma lista com quatro fatores de risco conhecidos.
# Usa `hiv_pos_model` (a regra AIDS⇒HIV da §2.5) em vez do `HIV` cru, que é o indicador
# canônico do projeto.
CLINICAL_BASELINE_FEATURES = ("IDADE", "hiv_pos_model", "AGRAVALCOO", "AGRAVDROGA", "POP_RUA")

# Derivadas e engenheiradas que não têm FeatureSpec próprio. Todas descrevem a
# PESSOA notificada (idade, interações com agravos, regra AIDS⇒HIV), então são
# individuais. Listadas à mão de propósito: uma feature nova só entra na ablação
# depois de alguém decidir de que lado ela cai.
_DERIVED_INDIVIDUAL = frozenset(DERIVED_FEATURES) | {"hiv_pos_model"}


def classify_feature_scope(columns, specs) -> dict:
    """Rotula cada coluna do X como 'individual' ou 'municipal'.

    O rótulo vem do `contextual` do FeatureSpec — que marca o que descreve o
    MUNICÍPIO (IBGE, IPEA, CNES e a geografia derivada) contra o que descreve a
    pessoa — e NÃO de heurística sobre o nome da coluna. Coluna sem spec e fora
    de `_DERIVED_INDIVIDUAL` levanta erro em vez de cair num braço por omissão:
    engolir uma feature nova enviesaria a ablação em silêncio, que é exatamente
    o defeito que a análise existe para excluir.
    """
    por_nome = {s.raw_name: s for s in specs}
    escopo, desconhecidas = {}, []
    for c in columns:
        spec = por_nome.get(c)
        if spec is not None:
            escopo[c] = "municipal" if spec.contextual else "individual"
        elif c in _DERIVED_INDIVIDUAL:
            escopo[c] = "individual"
        else:
            desconhecidas.append(c)
    if desconhecidas:
        raise ValueError(
            f"{len(desconhecidas)} coluna(s) sem escopo declarado para a ablação "
            f"territorial: {sorted(desconhecidas)}. Declare um FeatureSpec (com "
            "'contextual') ou inclua em _DERIVED_INDIVIDUAL."
        )
    return escopo


def select_feature_scope(X: pd.DataFrame, specs, scope: str) -> pd.DataFrame:
    """Recorta o X para um braço da ablação, preservando a ordem das colunas."""
    if scope not in FEATURE_SCOPES:
        raise ValueError(f"escopo desconhecido: {scope!r}; use um de {FEATURE_SCOPES}")
    if scope == "combined":
        return X
    if scope == "clinical_baseline":
        # subconjunto NOMEADO, não uma partição: falha se alguma das quatro variáveis
        # clínicas sumir do X, em vez de devolver um baseline silenciosamente menor.
        faltando = [c for c in CLINICAL_BASELINE_FEATURES if c not in X.columns]
        if faltando:
            raise ValueError(f"baseline clínico sem as colunas {faltando} no X")
        return X[list(CLINICAL_BASELINE_FEATURES)].copy()
    escopo = classify_feature_scope(list(X.columns), specs)
    cols = [c for c in X.columns if escopo[c] == scope]
    if not cols:
        raise ValueError(f"o braço {scope!r} ficou sem nenhuma coluna")
    return X[cols].copy()


def derive_age(df: pd.DataFrame) -> pd.Series:
    """Idade = ano da notificação − ano de nascimento.

    Ausente permanece ausente. O original usava `df[df["IDADE"] != -1]`, que
    depende de a ausência já ter virado -1 em algum ponto anterior — aqui não
    existe sentinela numérica.
    """
    idade = pd.to_numeric(df["NU_ANO"], errors="coerce") - pd.to_numeric(
        df["ANO_NASC"], errors="coerce"
    )
    fora = (idade < IDADE_MINIMA) | (idade > IDADE_MAXIMA)
    return idade.mask(fora).astype("Int64").rename("IDADE")


def normalize_agravos(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Campos Sim/Não/Ignorado em binário.

    Códigos: 1 Sim, 2 Não, 9 Ignorado (e 0/3 observados na base). Tudo que não é
    1 nem 2 vira <NA>, nunca 0 — não saber se o paciente é diabético é diferente
    de saber que não é.
    """
    df = df.copy()
    for col in colunas:
        if col not in df.columns:
            continue
        b = pd.Series(pd.NA, index=df.index, dtype="Int64")
        b[df[col] == AGRAVO_SIM] = 1
        b[df[col] == AGRAVO_NAO] = 0
        df[f"{col}_BIN"] = b
    return df


def normalize_pop_especiais(df: pd.DataFrame) -> pd.DataFrame:
    """Populações especiais (campo 33), as quatro SEPARADAS + um combinado.

    O original chamava keep_only_combined=True e drop_original=True: colapsava as
    quatro num binário e descartava as originais. Isso inviabiliza o briefing
    §20.1, que exige desempenho por situação de rua e por privação de liberdade
    separadamente. O combinado é adicional, nunca substituto.

    Semântica do combinado, igual à original: 1 se qualquer uma é Sim; 0 se
    nenhuma é Sim e ao menos uma é Não; <NA> se todas são ausentes.
    """
    df = normalize_agravos(df, POPULACOES_ESPECIAIS)

    bins = [f"{c}_BIN" for c in POPULACOES_ESPECIAIS if f"{c}_BIN" in df.columns]
    if bins:
        combinado = df[bins].max(axis=1, skipna=True)
        todas_ausentes = df[bins].isna().all(axis=1)
        df["POP_ESPECIAIS_BIN"] = combinado.mask(todas_ausentes).astype("Int64")
    return df


def derive_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """IDADE × comorbidade. Reproduz as células 83-84 do notebook original."""
    df = df.copy()
    if {"IDADE", "AGRAVDIABE_BIN"} <= set(df.columns):
        df["IDADE_X_DIABETES"] = df["IDADE"] * df["AGRAVDIABE_BIN"]
    if {"IDADE", "AGRAVAIDS_BIN"} <= set(df.columns):
        df["IDADE_X_AIDS"] = df["IDADE"] * df["AGRAVAIDS_BIN"]
    return df


def apply_all_derivations(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as derivações do conjunto da notificação.

    Nenhuma delas lê DT_ENCERRA ou SITUA_ENCE — nem se estiverem no DataFrame.
    """
    df = df.copy()
    df["IDADE"] = derive_age(df)
    df = normalize_agravos(df, AGRAVOS)
    df = normalize_pop_especiais(df)
    df = derive_interactions(df)
    return df


# ---------------------------------------------------------------------------
# Construção do conjunto de features
# ---------------------------------------------------------------------------


def collapse_era_zeros(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Mapeia o código 0 para <NA> nas colunas marcadas `zero_means_missing`.

    Existem DOIS fenômenos distintos que o snapshot mistura, e só um deles é
    resolvível aqui:

    1. **Codificação da ausência.** Até 2017 o SINAN grava "não preenchido" como
       0; da v5 em diante, como campo vazio. É notação: a variável é válida e o
       mapeamento 0 -> <NA> elimina o degrau artificial de 2018.
    2. **Ausência estrutural.** O campo saiu da ficha — a coluna sobrevive no DBF
       mas a pergunta não é mais feita (ID_OCUPA_N: 10 valores em 808.889
       notificações pós-2018). Nenhum tratamento de ausência recupera isso; a
       variável é removida do X, não mapeada.

    Este função trata apenas o caso 1. O caso 2 sai pela classificação.
    """
    df = df.copy()
    alvos = [s.raw_name for s in specs if s.zero_means_missing and s.raw_name in df.columns]
    for col in alvos:
        df[col] = df[col].mask(df[col] == 0)
    if alvos:
        logger.info("Código 0 mapeado para <NA> em %d colunas: %s", len(alvos), alvos)
    return df


def build_notification_feature_set(df: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Monta o X do momento da notificação.

    Só entra o que a config marcou in_notification_set. O FeatureSpec já proíbe,
    na validação, que qualquer coisa que não seja 'at_notification' receba essa
    marca — então este filtro é a segunda barreira, não a primeira.

    Terceira barreira: assert_no_outcome_in_derivation, que falha se alguma feature
    derivar de DT_ENCERRA ou SITUA_ENCE. Três barreiras para o mesmo defeito porque
    ele já passou uma vez, em silêncio, e custou o benchmark inteiro.
    """
    assert_no_outcome_in_derivation(specs)

    # O código 0 vira <NA> ANTES das derivações: caso contrário normalize_agravos
    # e derive_age operariam sobre um 0 que significa ausência.
    limpo = collapse_era_zeros(df, specs)

    # As derivadas nascem antes da seleção; nenhuma delas lê o desfecho.
    enriquecido = apply_all_derivations(limpo)

    escolhidas = [s.raw_name for s in specs if s.in_notification_set]
    presentes = [c for c in escolhidas if c in enriquecido.columns]

    faltando = sorted(set(escolhidas) - set(presentes))
    if faltando:
        logger.warning(
            "%d features declaradas não estão na coorte recebida: %s", len(faltando), faltando
        )

    # Só as derivadas que SÃO features entram no X. Os *_BIN dos agravos e das
    # populações são calculados (as interações dependem deles), mas ficam FORA:
    # o categórico bruto correspondente já está no X e preserva o código 9
    # (ignorado) como nível distinto, que o _BIN colapsaria em <NA>. Ter os dois
    # seria colinearidade perfeita para modelo linear (decisão do PI, opção 'a').
    derivadas = [c for c in DERIVED_FEATURES if c in enriquecido.columns and c not in df.columns]
    X = enriquecido[presentes + derivadas].copy()
    X = collapse_pending_levels(X, [c for c in COLUNAS_CONTAMINADAS if c in X.columns])

    if {"HIV", "AGRAVAIDS"} <= set(limpo.columns):
        X["hiv_pos_model"] = derive_hiv_rule(limpo)

    logger.info("Conjunto de features da notificação: %d colunas.", X.shape[1])
    return X


def feature_availability_table(specs: list[FeatureSpec]) -> pd.DataFrame:
    """data/feature_availability.csv (briefing §7)."""
    return pd.DataFrame([s.model_dump() for s in specs])


def feature_dictionary_table(specs: list[FeatureSpec]) -> pd.DataFrame:
    """data/feature_dictionary.csv (briefing §11.1)."""
    return pd.DataFrame(
        [
            {
                "raw_name": s.raw_name,
                "harmonized_name": s.harmonized_name,
                "form_field": s.form_field,
                "type": s.type,
                "source": s.source,
                "availability": s.availability,
                "dictionary_evidence": s.dictionary_evidence,
                "in_notification_set": s.in_notification_set,
                "derived_from": ",".join(s.derived_from),
                "protected": s.protected,
                "contextual": s.contextual,
            }
            for s in specs
        ]
    )


# ---------------------------------------------------------------------------
# Ausência informativa (briefing §11.3)
# ---------------------------------------------------------------------------

CODIGO_CURA = 1
CODIGO_OBITO_TB = 3


def missingness_report(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Ausência por coluna, opcionalmente estratificada (briefing §11.3)."""
    if by is None:
        return pd.DataFrame(
            [
                {
                    "column": c,
                    "n": len(df),
                    "n_missing": int(df[c].isna().sum()),
                    "pct_missing": round(float(df[c].isna().mean() * 100), 2),
                }
                for c in df.columns
            ]
        )

    linhas = []
    alvo = [c for c in df.columns if c not in by]
    for chaves, grupo in df.groupby(by):
        chaves = chaves if isinstance(chaves, tuple) else (chaves,)
        for c in alvo:
            linha = dict(zip(by, chaves))
            linha.update(
                {
                    "column": c,
                    "n": len(grupo),
                    "n_missing": int(grupo[c].isna().sum()),
                    "pct_missing": round(float(grupo[c].isna().mean() * 100), 2),
                }
            )
            linhas.append(linha)
    return pd.DataFrame(linhas)


def missingness_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta '<col>_is_missing' para cada coluna com ausência.

    A ausência é informação: vira feature própria em vez de ser apagada. O valor
    original permanece ausente — a imputação é do preprocess, dentro da dobra.

    Ressalva que precisa acompanhar estes indicadores no SHAP: para exames, a
    ausência mistura marginalização de linha de base com morte precoce. Não são
    achado clínico. Ver spec §3.1.
    """
    X = X.copy()
    for c in [c for c in X.columns if X[c].isna().any()]:
        X[f"{c}_is_missing"] = X[c].isna().astype("int8")
    return X


def early_death_missingness_audit(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Variáveis cuja ausência pode ser consequência de morte precoce (briefing §11.3).

    Se um campo está muito mais ausente nos óbitos do que nas curas, sua ausência
    pode refletir que o paciente morreu antes de o dado ser coletado — e o modelo
    aprenderia isso como se fosse preditivo.

    Cuidado ao ler: diferença grande NÃO prova artefato. Para o HIV, medimos que a
    maior parte do gap persiste a tempo igual (18,5 pp entre óbito e cura na faixa
    de 0-7 dias), o que aponta linha de base, não registro. Este relatório
    sinaliza candidatos; o discriminador é hiv_availability_by_duration.
    """
    obitos = df[df[outcome_col] == CODIGO_OBITO_TB]
    curas = df[df[outcome_col] == CODIGO_CURA]

    linhas = []
    for c in [c for c in df.columns if c != outcome_col]:
        pct_obito = float(obitos[c].isna().mean() * 100) if len(obitos) else 0.0
        pct_cura = float(curas[c].isna().mean() * 100) if len(curas) else 0.0
        linhas.append(
            {
                "column": c,
                "pct_missing_death": round(pct_obito, 2),
                "pct_missing_cure": round(pct_cura, 2),
                "abs_difference": round(abs(pct_obito - pct_cura), 2),
            }
        )
    return pd.DataFrame(linhas).sort_values("abs_difference", ascending=False)


# ---------------------------------------------------------------------------
# Colinearidade (briefing §11.4)
# ---------------------------------------------------------------------------


def correlation_report(X: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Pearson em formato longo, sem diagonal nem pares repetidos."""
    numericas = X.select_dtypes(include=[np.number])
    corr = numericas.corr()

    linhas = []
    colunas = list(corr.columns)
    for i, a in enumerate(colunas):
        for b in colunas[i + 1 :]:
            linhas.append({"column_a": a, "column_b": b, "correlation": float(corr.loc[a, b])})
    return pd.DataFrame(linhas)


def vif_report(X: pd.DataFrame, max_missing_fraction: float = 0.30) -> pd.DataFrame:
    """VIF por coluna. Constante ou não computável => NA, jamais 0.

    O manuscrito reporta 'VIF exatamente 0,000' para algumas variáveis. Zero não
    é um VIF possível — o mínimo teórico é 1. Era coluna constante devolvendo um
    valor degenerado; rastreamos até a coluna `NumeroCasos`, constante igual a 1,
    já removida no sub-projeto 1. Aqui isso é NA com motivo declarado.

    Método (briefing §11.4 exige que seja documentado):

    VIF precisa de uma matriz sem ausentes. Caso completo sobre todas as colunas
    numéricas é inviável num registro de saúde real: no snapshot atual, **zero**
    das 1.326.538 linhas está completa nas 94 colunas do X — basta uma variável
    rara faltar para a linha cair. Um `dropna()` ingênuo devolve um relatório
    inteiramente NA, que passa despercebido porque tecnicamente 'não tem zeros'.

    Então: colunas com mais de `max_missing_fraction` de ausência são excluídas do
    cálculo (com motivo declarado), e o VIF é computado por caso completo sobre as
    restantes. A coluna `n_rows_used` expõe quantas linhas sustentaram cada
    estimativa — sem isso, um resultado vazio é indistinguível de um resultado.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    numericas = X.select_dtypes(include=[np.number]).astype("float64")

    linhas = []
    constantes = [c for c in numericas.columns if numericas[c].nunique(dropna=True) <= 1]

    ausencia = numericas.drop(columns=constantes).isna().mean()
    muito_ausentes = sorted(ausencia[ausencia > max_missing_fraction].index)
    for c in muito_ausentes:
        linhas.append(
            {
                "column": c,
                "vif": np.nan,
                "reason": f"too_much_missing ({ausencia[c] * 100:.1f}%)",
                "n_rows_used": 0,
            }
        )

    computaveis = numericas.drop(columns=constantes + muito_ausentes).dropna()
    n_usadas = len(computaveis)

    for c in constantes:
        linhas.append({"column": c, "vif": np.nan, "reason": "constant", "n_rows_used": 0})

    if computaveis.shape[1] == 1:
        # VIF mede colinearidade COM AS OUTRAS variáveis. Sozinha, uma coluna não
        # tem com quem ser colinear: R² = 0 e VIF = 1 por definição. Devolver NA
        # aqui seria tão degenerado quanto o zero que este relatório corrige.
        linhas.append(
            {
                "column": computaveis.columns[0],
                "vif": 1.0,
                "reason": "single_column_no_collinearity",
                "n_rows_used": n_usadas,
            }
        )
    elif computaveis.shape[1] >= 2 and n_usadas > computaveis.shape[1]:
        matriz = computaveis.to_numpy()
        for i, c in enumerate(computaveis.columns):
            try:
                v = float(variance_inflation_factor(matriz, i))
                if not np.isfinite(v):
                    linhas.append(
                        {
                            "column": c,
                            "vif": np.nan,
                            "reason": "perfect_collinearity",
                            "n_rows_used": n_usadas,
                        }
                    )
                else:
                    linhas.append(
                        {"column": c, "vif": round(v, 4), "reason": "ok", "n_rows_used": n_usadas}
                    )
            except Exception as e:  # matriz singular
                linhas.append(
                    {
                        "column": c,
                        "vif": np.nan,
                        "reason": f"not_computable: {e}",
                        "n_rows_used": n_usadas,
                    }
                )
    else:
        for c in computaveis.columns:
            linhas.append(
                {
                    "column": c,
                    "vif": np.nan,
                    "reason": "insufficient_complete_cases",
                    "n_rows_used": n_usadas,
                }
            )

    logger.info(
        "VIF: %d colunas computadas sobre %d linhas de caso completo; "
        "%d constantes, %d com ausência acima de %.0f%%.",
        sum(1 for r_ in linhas if r_["reason"] == "ok"),
        n_usadas,
        len(constantes),
        len(muito_ausentes),
        max_missing_fraction * 100,
    )
    return pd.DataFrame(linhas)
