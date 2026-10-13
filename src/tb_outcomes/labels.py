"""Rótulos em inglês para o manuscrito.

O pipeline trabalha com códigos do SINAN e nomes internos em português. O artigo é em
inglês, então a tradução vive aqui, na camada de apresentação, e não nos módulos de
análise: mudar o idioma de saída não pode alterar a semântica do dado.

Códigos sem tradução declarada são mantidos como estão, nunca inventados. `SITUA` marca
os que vêm do dicionário SINAN NET 5.0; fontes adicionais estão documentadas em R13.
"""
from __future__ import annotations

# --- algoritmos: identificador interno -> nome de publicação -----------------
ALGORITMOS = {
    "lightgbm": "LightGBM",
    "hist_gradient_boost": "Histogram gradient boosting",
    "gradient_boosting": "Gradient boosting",
    "random_forest": "Random forest",
    "catboost": "CatBoost",
    "extra_trees": "Extremely randomised trees",
    "xgboost": "XGBoost",
    "adaboost": "AdaBoost",
    "decision_tree": "Decision tree",
    "logistic_regression": "Logistic regression",
    "logistic_plain": "Logistic regression (unpenalised)",
    "ridge_classifier": "Ridge classifier",
    "lda": "Linear discriminant analysis",
    "qda": "Quadratic discriminant analysis",
    "gaussian_nb": "Gaussian naive Bayes",
    "multinomial_nb": "Multinomial naive Bayes",
    "bernoulli_nb": "Bernoulli naive Bayes",
    "complement_nb": "Complement naive Bayes",
    "knn": "k-nearest neighbours",
    "linear_svc": "Linear support vector classifier",
    "rbf_svm": "Support vector classifier (RBF kernel)",
    "mlp": "Multilayer perceptron",
    "tabnet": "TabNet",
    "ft_transformer": "FT-Transformer",
    "tab_transformer": "TabTransformer",
    "category_embedding": "Category embedding network",
    "majority_class": "Majority-class baseline",
    "stratified_random": "Stratified random baseline",
}

# --- desfechos ---------------------------------------------------------------
DESFECHOS = {
    "cure": "Cure",
    "tb_death": "TB death",
    "treatment_interruption": "Treatment interruption",
}

# --- eixos de subgrupo -------------------------------------------------------
EIXOS = {
    "regiao": "Region",
    "sexo": "Sex",
    "raca_cor": "Race/colour",
    "escolaridade": "Schooling",
    "faixa_etaria": "Age band",
    "vulnerabilidade_municipal": "Municipal vulnerability",
    "dobra_espacial": "Spatial fold",
    "completude_notificacao": "Notification completeness",
}

# --- níveis, por variável ----------------------------------------------------
# SITUA = dicionário SINAN NET 5.0, agravo tuberculose.
NIVEIS = {
    "Sex": {"Masculino": "Male", "Feminino": "Female", "Ignorado": "Unknown"},
    "Race/colour": {"Branca": "White", "Preta": "Black", "Amarela": "Asian",
                    "Parda": "Mixed", "Indígena": "Indigenous", "Ignorado": "Unknown"},
    "Region": {"Norte": "North", "Nordeste": "Northeast", "Centro-Oeste": "Central-West",
               "Sudeste": "Southeast", "Sul": "South"},
    "Schooling": {
        "Analfabeto": "Illiterate",
        "1ª a 4ª série incompleta do Fundamental": "Primary, incomplete (years 1-4)",
        "4ª série completa do Fundamental": "Primary, complete (year 4)",
        "5ª a 8ª série incompleta do Fundamental": "Primary, incomplete (years 5-8)",
        "Fundamental completo": "Primary, complete",
        "Médio incompleto": "Secondary, incomplete",
        "Médio completo": "Secondary, complete",
        "Superior incompleto": "Higher education, incomplete",
        "Superior completo": "Higher education, complete",
        "Ignorado": "Unknown",
        "Não se aplica (idade < 7 anos)": "Not applicable (age under 7)",
    },
}

# R13 supplies field-specific dictionaries through codigo_sinan below.
# HIV is outside the R13 tables and remains undeclared in this legacy list.
SEM_DICIONARIO = ("HIV status",)

# quintis de vulnerabilidade: Q1 = menos vulnerável
QUINTIS = {f"Q{i}": f"Q{i}" for i in range(1, 6)}

# Motivos de exclusão da elegibilidade. As CHAVES vêm de `cohort.apply_eligibility`; a lista
# cresce quando um critério é acrescentado ao protocolo. Fonte única para a figura do funil,
# a tabela do corpo e a tabela suplementar — as três traziam o mesmo dicionário copiado, e
# um critério novo ficava faltando em duas delas sem aviso.
MOTIVOS_EXCLUSAO = {
    "municipio_irrecuperavel": "Unrecoverable municipality code",
    "idade_impossivel": "Impossible age",
    "fora_da_janela_declarada": "Outside the declared window",
    "notificacao_pos_obito": "Post-mortem notification",
    "seguimento_insuficiente": "Insufficient potential follow-up",
}
MOTIVOS_EXCLUSAO_PT = {
    "municipio_irrecuperavel": "Código de município irrecuperável",
    "idade_impossivel": "Idade impossível",
    "fora_da_janela_declarada": "Fora da janela declarada",
    "notificacao_pos_obito": "Notificação pós-óbito",
    "seguimento_insuficiente": "Tempo potencial de acompanhamento insuficiente",
}

# A transição do funil em que cada motivo age. `declarada` é recalculada só pela janela de
# ano; todos os demais critérios agem entre `declarada` e `elegivel`.
MOTIVO_TRANSICAO = {"fora_da_janela_declarada": "recuperada->declarada"}
TRANSICAO_PADRAO = "declarada->elegivel"


def algoritmo(x) -> str:
    return ALGORITMOS.get(str(x), str(x))


def desfecho(x) -> str:
    return DESFECHOS.get(str(x), str(x))


def eixo(x) -> str:
    return EIXOS.get(str(x), str(x))


def nivel(variavel: str, x) -> str:
    """Traduz um nível dentro de uma variável; devolve o original se não houver mapa."""
    s = str(x)
    tabela = NIVEIS.get(variavel, {})
    if s in tabela:
        return tabela[s]
    if s in QUINTIS:
        return QUINTIS[s]
    if s == "Missing":
        return "Missing"
    return s


def nivel_por_eixo(eixo_ingles: str, x) -> str:
    """Como `nivel`, mas recebendo o eixo já traduzido (Region, Sex, ...)."""
    return nivel(eixo_ingles, x)


def motivo_exclusao(chave, pt: bool = False) -> str:
    """Rótulo legível de um motivo de exclusão, tolerante a critérios ainda não mapeados.

    Um critério novo no protocolo não pode quebrar a figura nem aparecer como chave crua na
    tabela: sem entrada no dicionário, `notificacao_pos_obito` vira "Notificacao pos obito",
    que é legível e sinaliza que falta traduzir.
    """
    s = str(chave)
    tabela = MOTIVOS_EXCLUSAO_PT if pt else MOTIVOS_EXCLUSAO
    if s in tabela:
        return tabela[s]
    return s.replace("_", " ").capitalize()


def transicao_do_motivo(chave) -> str:
    """Em que transição do funil o motivo age."""
    return MOTIVO_TRANSICAO.get(str(chave), TRANSICAO_PADRAO)


# ---------------------------------------------------------------------------
# Versão em português (divulgação nacional). O inglês continua sendo o idioma
# do manuscrito submetido; isto serve à versão pt-BR.
# ---------------------------------------------------------------------------

DESFECHOS_PT = {
    "cure": "Cura",
    "tb_death": "Óbito por TB",
    "treatment_interruption": "Interrupção do tratamento",
}

EIXOS_PT = {
    "regiao": "Região",
    "sexo": "Sexo",
    "raca_cor": "Raça/cor",
    "escolaridade": "Escolaridade",
    "faixa_etaria": "Faixa etária",
    "vulnerabilidade_municipal": "Vulnerabilidade municipal",
    "dobra_espacial": "Dobra espacial",
    "completude_notificacao": "Completude da notificação",
}

ESTRATEGIAS_PT = {
    "none": "Sem balanceamento",
    "local_cost_sensitive": "Sensível ao custo",
    "random_oversampling": "Sobreamostragem",
    "random_undersampling": "Subamostragem",
}

ESTRATEGIAS_EN = {
    "none": "No balancing",
    "local_cost_sensitive": "Cost-sensitive",
    "random_oversampling": "Oversampling",
    "random_undersampling": "Undersampling",
}


def desfecho_pt(x) -> str:
    return DESFECHOS_PT.get(str(x), str(x))


def eixo_pt(x) -> str:
    return EIXOS_PT.get(str(x), str(x))


def estrategia(x, pt: bool = False) -> str:
    tab = ESTRATEGIAS_PT if pt else ESTRATEGIAS_EN
    return tab.get(str(x), str(x))

# ---------------------------------------------------------------------------
# Vocabulário de proveniência (contrato de execução conjunta).
#
# Três distinções que o manuscrito confundia e que a revisão exigiu separar: a forma de
# AGREGAÇÃO (média entre dobras externas contra cálculo sobre predições agrupadas), a REGRA
# de decisão (arg max contra limiares por classe) e o PAPEL do número (estimativa produzida
# pelo procedimento de seleção contra comparação exploratória entre famílias). Números iguais
# sob rótulos diferentes deixaram de bater entre seções justamente por isso.

AGREGACAO = {
    "mean_across_folds": "mean across outer folds",
    "pooled_oof": "computed on pooled out-of-fold predictions",
    "single_split": "single held-out partition",
}
AGREGACAO_PT = {
    "mean_across_folds": "média entre as dobras externas",
    "pooled_oof": "calculado sobre as predições fora da dobra agrupadas",
    "single_split": "partição única retida",
}

REGRA_DECISAO = {
    "pred_argmax": "arg max of the calibrated probabilities",
    "pred_policy": "per-class decision thresholds (secondary policy)",
    "ranking_only": "ranking only; no label assigned",
}
REGRA_DECISAO_PT = {
    "pred_argmax": "arg max das probabilidades calibradas",
    "pred_policy": "limiares de decisão por classe (política secundária)",
    "ranking_only": "apenas ordenamento; sem atribuição de rótulo",
}

PAPEL_RESULTADO = {
    "selection_procedure": "estimate produced by the selection procedure",
    "exploratory_ranking": "exploratory comparison among families",
    "sensitivity": "sensitivity analysis",
}
PAPEL_RESULTADO_PT = {
    "selection_procedure": "estimativa produzida pelo procedimento de seleção",
    "exploratory_ranking": "comparação exploratória entre famílias",
    "sensitivity": "análise de sensibilidade",
}


def _traduz(tabela: dict, tabela_pt: dict, chave, pt: bool) -> str:
    """Rótulo legível; chave desconhecida degrada para si mesma, sem quebrar."""
    s = str(chave)
    return (tabela_pt if pt else tabela).get(s, s.replace("_", " "))


def agregacao(chave, pt: bool = False) -> str:
    return _traduz(AGREGACAO, AGREGACAO_PT, chave, pt)


def regra_decisao(chave, pt: bool = False) -> str:
    return _traduz(REGRA_DECISAO, REGRA_DECISAO_PT, chave, pt)


def papel_resultado(chave, pt: bool = False) -> str:
    return _traduz(PAPEL_RESULTADO, PAPEL_RESULTADO_PT, chave, pt)


def proveniencia(*, arm=None, population=None, decision_rule=None, aggregation=None,
                 role=None, pt: bool = False) -> str:
    """Frase de proveniência para a legenda de uma tabela ou figura.

    Campos ausentes são OMITIDOS em vez de preenchidos por omissão: uma legenda que afirma
    uma regra de decisão que ninguém declarou é pior que uma legenda incompleta.
    """
    partes = []
    if arm:
        partes.append(f"{'braço' if pt else 'arm'}: {arm}")
    if population:
        partes.append(f"{'população' if pt else 'population'}: {population}")
    if decision_rule:
        partes.append(f"{'regra' if pt else 'decision rule'}: "
                      f"{regra_decisao(decision_rule, pt)}")
    if aggregation:
        partes.append(f"{'agregação' if pt else 'aggregation'}: "
                      f"{agregacao(aggregation, pt)}")
    if role:
        partes.append(f"{'papel' if pt else 'role'}: {papel_resultado(role, pt)}")
    return "; ".join(partes)


# ---------------------------------------------------------------------------
# Reanálise corrigida (L5): braços, códigos de raça/cor e preditores do conjunto principal.
#
# As chaves são os identificadores dos artefatos liberados. Um identificador sem tradução
# declarada degrada para si mesmo, como no restante do módulo: rótulo inventado é pior que
# código cru. Os 49 preditores abaixo são exatamente os de `primary_feature_dictionary.csv`;
# `tests/editorial/test_c7_rotulos.py` falha se o dicionário liberado trouxer um nome ausente.

BRACOS = {
    "primary": "Primary (49 predictors)",
    "temporal": "Temporal evaluation (2024)",
    "clinical": "Four-factor clinical comparator",
    "individual": "Individual predictors only",
    "municipal": "Municipal predictors only",
    "contemporary": "Contemporaneous municipal context",
    "without_diagnostics": "Without diagnostic tests",
    "hiv_uncertain": "HIV serology added",
    "k27": "27 spatial clusters",
    "k75": "75 spatial clusters",
    "random_group": "Random municipality groups",
    "all_cause": "Death from any cause",
}
BRACOS_PT = {
    "primary": "Principal (49 preditores)",
    "temporal": "Avaliação temporal (2024)",
    "clinical": "Comparador clínico de quatro fatores",
    "individual": "Apenas preditores individuais",
    "municipal": "Apenas preditores municipais",
    "contemporary": "Contexto municipal contemporâneo",
    "without_diagnostics": "Sem exames diagnósticos",
    "hiv_uncertain": "Com sorologia para HIV",
    "k27": "27 agrupamentos espaciais",
    "k75": "75 agrupamentos espaciais",
    "random_group": "Grupos aleatórios de municípios",
    "all_cause": "Óbito por qualquer causa",
}

# CS_RACA no SINAN NET 5.0. O código 9 (ignorado) e a ausência não formam grupo de comparação.
RACA_COR = {"1": "White", "2": "Black", "3": "Asian", "4": "Mixed", "5": "Indigenous"}
RACA_COR_PT = {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}

REGIOES_PT = {"Norte": "Norte", "Nordeste": "Nordeste", "Centro-Oeste": "Centro-Oeste",
              "Sudeste": "Sudeste", "Sul": "Sul"}

PREDITORES = {
    "CS_SEXO": "Sex",
    "CS_GESTANT": "Pregnancy status",
    "CS_RACA": "Race/colour",
    "CS_ESCOL_N": "Schooling",
    "TRATAMENTO": "Type of entry",
    "RAIOX_TORA": "Chest radiography",
    "FORMA": "Clinical form",
    "AGRAVAIDS": "Declared AIDS",
    "AGRAVALCOO": "Alcohol use",
    "AGRAVDIABE": "Diabetes",
    "AGRAVDOENC": "Mental disorder",
    "BACILOSC_E": "Sputum smear",
    "BACILOSC_O": "Smear of other material",
    "NU_CONTATO": "Contacts identified",
    "DOENCA_TRA": "Work-related disease",
    "POP_LIBER": "Deprived of liberty",
    "POP_RUA": "Homelessness",
    "POP_SAUDE": "Health worker",
    "POP_IMIG": "Immigrant",
    "BENEF_GOV": "Government cash-transfer beneficiary",
    "AGRAVDROGA": "Illicit drug use",
    "AGRAVTABAC": "Tobacco use",
    "TEST_MOLEC": "Rapid molecular test",
    "IDADE": "Age",
    "POP_ESPECIAIS_BIN": "Any special population",
    "IDADE_X_DIABETES": "Age x diabetes",
    "IDADE_X_AIDS": "Age x declared AIDS",
    "REGIAO": "Macroregion",
    "LAT_MUNIC": "Municipal latitude",
    "LONG_MUNIC": "Municipal longitude",
    "IPEA_IDHM": "Municipal HDI",
    "IPEA_PCT_ADENSAMENTO_DOMICILIAR": "Household overcrowding (%)",
    "IPEA_PCT_AGUA_ESGOTO_INADEQUADO": "Inadequate water and sewage (%)",
    "IPEA_PCT_VULNERAVEIS_POBREZA": "Vulnerable to poverty (%)",
    "IPEA_PCT_EXTREMAMENTE_POBRES": "Extremely poor (%)",
    "IPEA_INDICE_GINI": "Gini index",
    "IPEA_RENDA_PER_CAPITA": "Income per capita",
    "IPEA_TAXA_ANALFABETISMO_15MAIS": "Illiteracy, age 15+ (%)",
    "IPEA_TAXA_DESEMPREGO_18MAIS": "Unemployment, age 18+ (%)",
    "IPEA_ESPERANCA_VIDA": "Life expectancy",
    "IPEA_MORTALIDADE_ATE_1ANO": "Infant mortality",
    "IPEA_PCT_PAREDE_INADEQUADA": "Inadequate walls (%)",
    "IPEA_PCT_AGUA_ENCANADA": "Piped water (%)",
    "IPEA_PCT_COLETA_LIXO": "Refuse collection (%)",
    "IPEA_PCT_TRABALHO_FORMAL": "Formal employment (%)",
    "IPEA_RAZAO_DEPENDENCIA": "Dependency ratio",
    "IPEA_TAXA_ENVELHECIMENTO": "Ageing rate",
    "IPEA_POPULACAO_TOTAL": "Municipal population",
    "IPEA_INDICE_THEIL": "Theil index",
}
PREDITORES_PT = {
    "CS_SEXO": "Sexo",
    "CS_GESTANT": "Situação gestacional",
    "CS_RACA": "Raça/cor",
    "CS_ESCOL_N": "Escolaridade",
    "TRATAMENTO": "Tipo de entrada",
    "RAIOX_TORA": "Radiografia de tórax",
    "FORMA": "Forma clínica",
    "AGRAVAIDS": "AIDS declarada",
    "AGRAVALCOO": "Uso de álcool",
    "AGRAVDIABE": "Diabetes",
    "AGRAVDOENC": "Transtorno mental",
    "BACILOSC_E": "Baciloscopia de escarro",
    "BACILOSC_O": "Baciloscopia de outro material",
    "NU_CONTATO": "Contatos identificados",
    "DOENCA_TRA": "Doença relacionada ao trabalho",
    "POP_LIBER": "Privação de liberdade",
    "POP_RUA": "Situação de rua",
    "POP_SAUDE": "Profissional de saúde",
    "POP_IMIG": "Imigrante",
    "BENEF_GOV": "Beneficiário de transferência de renda",
    "AGRAVDROGA": "Uso de drogas ilícitas",
    "AGRAVTABAC": "Tabagismo",
    "TEST_MOLEC": "Teste molecular rápido",
    "IDADE": "Idade",
    "POP_ESPECIAIS_BIN": "Alguma população especial",
    "IDADE_X_DIABETES": "Idade x diabetes",
    "IDADE_X_AIDS": "Idade x AIDS declarada",
    "REGIAO": "Macrorregião",
    "LAT_MUNIC": "Latitude do município",
    "LONG_MUNIC": "Longitude do município",
    "IPEA_IDHM": "IDH municipal",
    "IPEA_PCT_ADENSAMENTO_DOMICILIAR": "Adensamento domiciliar (%)",
    "IPEA_PCT_AGUA_ESGOTO_INADEQUADO": "Água e esgoto inadequados (%)",
    "IPEA_PCT_VULNERAVEIS_POBREZA": "Vulneráveis à pobreza (%)",
    "IPEA_PCT_EXTREMAMENTE_POBRES": "Extremamente pobres (%)",
    "IPEA_INDICE_GINI": "Índice de Gini",
    "IPEA_RENDA_PER_CAPITA": "Renda per capita",
    "IPEA_TAXA_ANALFABETISMO_15MAIS": "Analfabetismo, 15 anos ou mais (%)",
    "IPEA_TAXA_DESEMPREGO_18MAIS": "Desemprego, 18 anos ou mais (%)",
    "IPEA_ESPERANCA_VIDA": "Esperança de vida",
    "IPEA_MORTALIDADE_ATE_1ANO": "Mortalidade infantil",
    "IPEA_PCT_PAREDE_INADEQUADA": "Paredes inadequadas (%)",
    "IPEA_PCT_AGUA_ENCANADA": "Água encanada (%)",
    "IPEA_PCT_COLETA_LIXO": "Coleta de lixo (%)",
    "IPEA_PCT_TRABALHO_FORMAL": "Trabalho formal (%)",
    "IPEA_RAZAO_DEPENDENCIA": "Razão de dependência",
    "IPEA_TAXA_ENVELHECIMENTO": "Taxa de envelhecimento",
    "IPEA_POPULACAO_TOTAL": "População do município",
    "IPEA_INDICE_THEIL": "Índice de Theil",
}

FONTES_PREDITOR = {"SINAN": "SINAN notification form", "IPEA": "IPEA (2000/2010 censuses)",
                   "derived": "Derived from baseline inputs"}
FONTES_PREDITOR_PT = {"SINAN": "Ficha de notificação do SINAN", "IPEA": "IPEA (censos de 2000/2010)",
                      "derived": "Derivada de entradas basais"}


def braco(x, pt: bool = False) -> str:
    return (BRACOS_PT if pt else BRACOS).get(str(x), str(x))


def raca_cor(x, pt: bool = False) -> str:
    return (RACA_COR_PT if pt else RACA_COR).get(str(x), str(x))


def preditor(x, pt: bool = False) -> str:
    return (PREDITORES_PT if pt else PREDITORES).get(str(x), str(x))


def algoritmo_pt(x) -> str:
    """Nomes próprios de algoritmos ficam como estão; só as descrições são traduzidas."""
    return ALGORITMOS_PT.get(str(x), algoritmo(x))


ALGORITMOS_PT = {
    "hist_gradient_boost": "Gradient boosting por histogramas",
    "gradient_boosting": "Gradient boosting",
    "random_forest": "Floresta aleatória",
    "extra_trees": "Árvores extremamente aleatorizadas",
    "decision_tree": "Árvore de decisão",
    "logistic_regression": "Regressão logística",
    "logistic_plain": "Regressão logística (sem penalização)",
    "ridge_classifier": "Classificador ridge",
    "lda": "Análise discriminante linear",
    "qda": "Análise discriminante quadrática",
    "gaussian_nb": "Naive Bayes gaussiano",
    "multinomial_nb": "Naive Bayes multinomial",
    "bernoulli_nb": "Naive Bayes de Bernoulli",
    "complement_nb": "Naive Bayes complementar",
    "knn": "k vizinhos mais próximos",
    "mlp": "Perceptron multicamadas",
    "category_embedding": "Rede de embeddings categóricos",
    "majority_class": "Referência da classe majoritária",
    "stratified_random": "Referência aleatória estratificada",
}

# R13: presentation-only SINAN code labels. Sources and discrepancies:
# docs/manuscrito/colaboracao/codex/R13_ROTULOS_20261005.md
CODIGOS_SINAN = {
    "CS_ESCOL_N": {
        "0": ("Illiterate", "Analfabeto"),
        "1": ("Primary, incomplete (years 1-4)", "1a a 4a serie incompleta do EF"),
        "2": ("Primary, year 4 complete", "4a serie completa do EF"),
        "3": ("Primary, incomplete (years 5-8)", "5a a 8a serie incompleta do EF"),
        "4": ("Primary education complete", "Ensino fundamental completo"),
        "5": ("Secondary education incomplete", "Ensino medio incompleto"),
        "6": ("Secondary education complete", "Ensino medio completo"),
        "7": ("Higher education incomplete", "Superior incompleto"),
        "8": ("Higher education complete", "Superior completo"),
        "9": ("Unknown", "Ignorado"), "10": ("Not applicable", "Nao se aplica"),
    },
    "FORMA": {"1": ("Pulmonary", "Pulmonar"), "2": ("Extrapulmonary", "Extrapulmonar"),
              "3": ("Pulmonary and extrapulmonary", "Pulmonar e extrapulmonar")},
    "TRATAMENTO": {"1": ("New case", "Caso novo"), "2": ("Relapse", "Recidiva"),
                   "3": ("Re-entry after treatment interruption", "Reingresso apos abandono"),
                   "4": ("Unknown", "Nao sabe"), "5": ("Transfer", "Transferencia"),
                   "6": ("Post-mortem notification", "Pos-obito")},
    "RAIOX_TORA": {"1": ("Suggestive of TB", "Suspeito"), "2": ("Normal", "Normal"),
                   "3": ("Other pathology", "Outra patologia"),
                   "4": ("Not performed", "Nao realizado")},
    "BACILOSC_E": {"1": ("Positive", "Positiva"), "2": ("Negative", "Negativa"),
                   "3": ("Not performed", "Nao realizada"),
                   "4": ("Not applicable", "Nao se aplica")},
    "BACILOSC_O": {"1": ("Positive", "Positiva"), "2": ("Negative", "Negativa"),
                   "3": ("Not performed", "Nao realizada")},
    "TEST_MOLEC": {"1": ("Detected, rifampicin susceptible", "Detectavel, sensivel a rifampicina"),
                   "2": ("Detected, rifampicin resistant", "Detectavel, resistente a rifampicina"),
                   "3": ("Not detected", "Nao detectavel"),
                   "4": ("Inconclusive", "Inconclusivo"), "5": ("Not performed", "Nao realizado")},
}
for _campo in ("AGRAVAIDS", "AGRAVALCOO", "AGRAVDIABE", "AGRAVDOENC", "AGRAVDROGA",
               "AGRAVTABAC", "POP_RUA", "POP_LIBER", "POP_SAUDE", "POP_IMIG", "BENEF_GOV"):
    CODIGOS_SINAN[_campo] = {"1": ("Yes", "Sim"), "2": ("No", "Nao"),
                            "9": ("Unknown", "Ignorado")}
CODIGOS_SINAN["CS_RACA"] = {
    k: (v, RACA_COR_PT[k]) for k, v in RACA_COR.items()
}
CODIGOS_SINAN["CS_RACA"]["9"] = ("Unknown", "Ignorado")
CODIGOS_SINAN["CS_SEXO"] = {"M": ("Male", "Masculino"), "F": ("Female", "Feminino"),
                           "I": ("Unknown", "Ignorado")}


def codigo_sinan(campo: str, valor, pt: bool = False) -> str:
    """Label without recoding, merging or imputing values; retain code identity.

    Noncanonical values remain explicitly unmapped. In particular, a historic
    PDF lists 3=unknown for binary fields, whereas the current linked dictionary
    uses 9. Without row-level version provenance we do not assume equivalence.
    """
    s = str(valor).strip()
    if s in ("__missing__", "nan", "None", "<NA>", ""):
        return "Ausente" if pt else "Missing"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if campo == "age_band":
        return s
    if campo == "REGIAO":
        return REGIOES_PT.get(s, s) if pt else NIVEIS["Region"].get(s, s)
    par = CODIGOS_SINAN.get(campo, {}).get(s)
    if par is None:
        return f"Codigo {s} (nao mapeado)" if pt else f"Code {s} (unmapped)"
    return f"{par[int(pt)]} ({s})"
