"""Rótulos em inglês para o manuscrito.

O pipeline trabalha com códigos do SINAN e nomes internos em português. O artigo é em
inglês, então a tradução vive aqui, na camada de apresentação, e não nos módulos de
análise: mudar o idioma de saída não pode alterar a semântica do dado.

Códigos sem tradução declarada são mantidos como estão, nunca inventados. `SITUA` marca
os que vêm do dicionário SINAN NET 5.0; os demais são nomes internos do projeto.
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

# Variáveis cujo dicionário de códigos NÃO foi fornecido e que, por isso, entram na tabela
# com o código do SINAN e não com um rótulo. Rotular por inferência é o mesmo erro que
# inventar uma referência: parece certo e pode estar errado, e num artigo de equidade um
# rótulo trocado é pior que um código cru.
#
# Para rotulá-las basta acrescentar o mapa em NIVEIS, exatamente como foi feito com
# Race/colour e Schooling depois que o PI forneceu o dicionário SINAN NET 5.0.
SEM_DICIONARIO = (
    "Clinical form", "HIV status", "Alcohol use", "Illicit drug use", "Diabetes",
    "Homelessness", "Deprivation of liberty", "Chest radiography",
)

# quintis de vulnerabilidade: Q1 = menos vulnerável
QUINTIS = {f"Q{i}": f"Q{i}" for i in range(1, 6)}


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
    "local_cost_sensitive": "Sensível ao custo",
    "random_oversampling": "Sobreamostragem",
    "random_undersampling": "Subamostragem",
}

ESTRATEGIAS_EN = {
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
