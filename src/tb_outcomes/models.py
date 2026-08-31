"""Modelos históricos atrás de uma interface única e a matriz de capacidades
MEDIDA (briefing §14, §15.1; ESPECIFICACAO §2.2).

Dois achados que corrigem o §2.2, ambos medidos, nunca assumidos:
  - sem predict_proba (Ridge, LinearSVC, RBF SVM) => sem métrica probabilística
    até a calibração cross-fitted do SP4d (calibrator_required);
  - sem sample_weight (LDA, QDA, kNN, MLP) => local_cost_sensitive unsupported,
    a fonte formal do supports_weight que o SP4b consome.
Baselines são neutros: nenhum codifica fator social/clínico como regra.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pydantic import BaseModel
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ModelEntry(BaseModel):
    family: str
    preprocess_profile: str
    tuning_tier: str


class ModelsConfig(BaseModel):
    baselines: dict[str, ModelEntry]
    models: dict[str, ModelEntry]
    hyperparameters: dict[str, dict] = {}


def load_models_config(path: str | Path) -> ModelsConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        return ModelsConfig.model_validate(yaml.safe_load(fh))


ALL_IMBALANCE = ["none", "random_oversampling", "random_undersampling", "local_cost_sensitive"]
DL_MODELS = {"category_embedding", "tabnet", "tab_transformer", "ft_transformer"}

# native_probability e sample_weight são MEDIDOS por probe_capabilities (introspecção
# do estimador). gpu e native_categorical não têm probe em runtime — são DECLARADOS,
# fatos da biblioteca. Precisam estar corretos: os boosting (XGBoost, LightGBM, CatBoost)
# aceleram em GPU e consomem categóricas nativas; o HistGB consome categóricas nativas
# (categorical_features) mas é CPU-only no sklearn.
NONNEGATIVE_MODELS = {"multinomial_nb", "complement_nb"}
NATIVE_CATEGORICAL = {
    "xgboost", "lightgbm", "hist_gradient_boost", "catboost",  # boosting capazes
    *DL_MODELS,
}
GPU_MODELS = {
    "xgboost", "lightgbm", "catboost",  # boosting com backend GPU
    *DL_MODELS,
}

_SKLEARN_FACTORIES = {
    "majority_class": lambda: DummyClassifier(strategy="most_frequent"),
    "stratified_random": lambda: DummyClassifier(strategy="stratified", random_state=0),
    "logistic_plain": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
    "decision_tree": DecisionTreeClassifier,
    # n_jobs=-1: paralelismo puro (árvores independentes c/ semente fixa => resultado
    # IDÊNTICO a n_jobs=1). Usa os 32 cores em vez de 1.
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier,
    # boosting de perfil native_categorical (SP4f): consomem dtype `category` nativamente,
    # em CPU. XGBoost/HistGB precisam do flag na construção; CatBoost recebe cat_features
    # no fit (montado pelo SklearnAdapter); LightGBM detecta pelo dtype.
    "hist_gradient_boost": lambda: HistGradientBoostingClassifier(
        categorical_features="from_dtype"),
    "adaboost": AdaBoostClassifier,
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, verbosity=0, enable_categorical=True, tree_method="hist", n_jobs=-1),
    "lightgbm": lambda: LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=-1),
    "catboost": lambda: CatBoostClassifier(iterations=300, verbose=0, thread_count=-1),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
    "ridge_classifier": RidgeClassifier,
    "linear_svc": LinearSVC,
    "rbf_svm": lambda: SVC(kernel="rbf"),  # sem probability=True: proba nativa é False
    "gaussian_nb": GaussianNB,
    "multinomial_nb": MultinomialNB,
    "bernoulli_nb": BernoulliNB,
    "complement_nb": ComplementNB,
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "knn": lambda: KNeighborsClassifier(n_jobs=-1),  # busca de vizinhos paralela (mesmos vizinhos)
    "mlp": lambda: MLPClassifier(max_iter=200),  # single-proc; usa cores via BLAS
}


@dataclass
class ModelCapabilities:
    name: str
    native_probability: bool
    decision_function: bool
    sample_weight: bool
    nonnegative_input: bool
    native_categorical: bool
    gpu: bool
    deterministic: bool
    calibrator_required: bool
    preprocess_profile: str
    supported_imbalance: list[str] = field(default_factory=list)
    tuning_tier: str = "default_benchmark"


def build_estimator(name: str, entry: ModelEntry):
    if name in DL_MODELS:
        raise ValueError(f"{name} é DL; use o DLAdapter")
    return _SKLEARN_FACTORIES[name]()


def probe_capabilities(estimator, name: str, entry: ModelEntry) -> ModelCapabilities:
    proba = hasattr(estimator, "predict_proba")
    dec = hasattr(estimator, "decision_function")
    sw = "sample_weight" in inspect.signature(estimator.fit).parameters
    supported = [s for s in ALL_IMBALANCE if s != "local_cost_sensitive" or sw]
    return ModelCapabilities(
        name=name,
        native_probability=proba,
        decision_function=dec,
        sample_weight=sw,
        nonnegative_input=name in NONNEGATIVE_MODELS,
        native_categorical=name in NATIVE_CATEGORICAL,
        gpu=name in GPU_MODELS,
        deterministic=name not in {"stratified_random", "mlp", *DL_MODELS},
        calibrator_required=not proba,
        preprocess_profile=entry.preprocess_profile,
        supported_imbalance=supported,
        tuning_tier=entry.tuning_tier,
    )


def _is_nontrivial_weight(sample_weight) -> bool:
    return sample_weight is not None and not np.allclose(sample_weight, sample_weight[0])


class SklearnAdapter:
    def __init__(self, name, estimator, capabilities: ModelCapabilities):
        self.name = name
        self._est = estimator
        self.capabilities = capabilities

    @property
    def input_kind(self) -> str:
        if self.capabilities.preprocess_profile == "native_categorical":
            return "categorical_frame"
        return "matrix"

    def _category_columns(self, X):
        if isinstance(X, pd.DataFrame):
            return [c for c in X.columns if str(X[c].dtype) == "category"]
        return []

    def fit(self, X, y, sample_weight=None):
        if _is_nontrivial_weight(sample_weight) and not self.capabilities.sample_weight:
            raise ValueError(
                f"{self.name} não aceita sample_weight não-trivial; "
                f"local_cost_sensitive é unsupported para este modelo (SP4b)"
            )
        kwargs = {}
        if self.capabilities.sample_weight and sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        # CatBoost exige a lista de categóricas no fit; os demais boosting nativos leem o dtype.
        if self.name == "catboost":
            cats = self._category_columns(X)
            if cats:
                kwargs["cat_features"] = cats
        self._est.fit(X, y, **kwargs)
        return self

    def predict_raw_scores(self, X) -> np.ndarray:
        if self.capabilities.native_probability:
            return self._est.predict_proba(X)
        return self._est.decision_function(X)

    def predict_proba_if_native(self, X):
        if not self.capabilities.native_probability:
            return None
        return self._est.predict_proba(X)


class DLAdapter:
    """Envolve um modelo pytorch_tabular e declara capacidades por família.

    A introspecção de probe_capabilities é sklearn-específica; os DL declaram
    aqui. O treino real (DataConfig/TrainerConfig, GPU) é montado pelo executor
    SP4e — aqui garantimos a interface e a matriz. sample_weight não é propagado
    a DL neste projeto, logo local_cost_sensitive é unsupported.
    """

    _MODEL_CONFIGS = {
        "category_embedding": "CategoryEmbeddingModelConfig",
        "tabnet": "TabNetModelConfig",
        "tab_transformer": "TabTransformerConfig",
        "ft_transformer": "FTTransformerConfig",
    }

    def __init__(self, name, entry: ModelEntry, dl_cfg=None):
        self.name = name
        self._entry = entry
        self._dl_cfg = dl_cfg
        self.capabilities = ModelCapabilities(
            name=name,
            native_probability=True,
            decision_function=False,
            sample_weight=False,
            nonnegative_input=False,
            native_categorical=True,
            gpu=True,
            deterministic=False,
            calibrator_required=False,
            preprocess_profile=entry.preprocess_profile,
            supported_imbalance=[s for s in ALL_IMBALANCE if s != "local_cost_sensitive"],
            tuning_tier=entry.tuning_tier,
        )
        self._model = None
        self._classes = None
        self._target = "__target__"

    @property
    def input_kind(self) -> str:
        return "dl_frame"

    def _cfg(self):
        if self._dl_cfg is None:
            from tb_outcomes.dl_config import load_dl_config
            self._dl_cfg = load_dl_config("configs/deep_learning.yaml")
        return self._dl_cfg

    def fit(self, X, y, sample_weight=None):
        import pytorch_lightning as pl
        import torch
        from pytorch_tabular import TabularModel
        from pytorch_tabular import models as ptm
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        if _is_nontrivial_weight(sample_weight):
            raise ValueError(
                f"{self.name} (DL) não recebe sample_weight; local_cost_sensitive é unsupported"
            )
        cfg = self._cfg()
        pl.seed_everything(cfg.seed, workers=True)
        self._classes = sorted(np.unique(y).tolist())
        cats = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
        conts = [c for c in X.columns if c not in cats]
        # Imputa as contínuas com a MEDIANA DO TREINO (armazenada p/ o predict, leak-safe).
        # NaN não pode chegar à rede: o TabNet usa sparsemax, que com NaN gera support_size 0
        # e estoura (index -1). As categóricas já vêm com o token 'MISSING' (sem NaN).
        self._cont_cols = conts
        self._cont_medians = X[conts].median().fillna(0.0) if conts else None
        X = self._impute(X)
        train = X.copy()
        train[self._target] = np.asarray(y).astype(str)  # alvo categórico p/ classificação

        data_config = DataConfig(
            target=[self._target], continuous_cols=conts, categorical_cols=cats,
            validation_split=cfg.validation_split, num_workers=cfg.num_workers)
        model_config = getattr(ptm, self._MODEL_CONFIGS[self.name])(task="classification")
        # Desliga a métrica interna de accuracy do pytorch_tabular. Num batch de validação
        # interna de tamanho 1 (que o random_oversampling produz em alguns folds, quando
        # n_val % batch_size == 1) a predição colapsa para 0-D e o torchmetrics estoura em
        # `preds.reshape(preds.shape[0], -1)` (IndexError: tuple index out of range). O early
        # stopping usa valid_loss (calculado à parte de calculate_metrics) e o executor calcula
        # todas as métricas reais por fora (calibração/limiares), então zerar a métrica interna
        # mata o crash sem efeito colateral. `metrics=[]` no construtor NÃO vale — o pydantic do
        # pytorch_tabular faz `self.metrics or ["accuracy"]`, então sobrescrevemos pós-construção.
        model_config.metrics = []
        model_config.metrics_params = []
        model_config.metrics_prob_input = []
        trainer_config = TrainerConfig(
            max_epochs=cfg.max_epochs, early_stopping="valid_loss",
            early_stopping_mode="min", early_stopping_patience=cfg.early_stopping_patience,
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
            batch_size=cfg.batch_size, progress_bar="none")
        self._model = TabularModel(
            data_config=data_config, model_config=model_config,
            optimizer_config=OptimizerConfig(), trainer_config=trainer_config,
            verbose=False)
        self._model.fit(train=train)
        return self

    def _impute(self, X):
        if not self._cont_cols:
            return X
        X = X.copy()
        X[self._cont_cols] = X[self._cont_cols].fillna(self._cont_medians).fillna(0.0)
        return X

    def predict_raw_scores(self, X) -> np.ndarray:
        pred = self._model.predict(self._impute(X))
        prob_cols = [c for c in pred.columns if c.endswith("_probability")]
        ordered = []
        for i, c in enumerate(self._classes):
            match = [col for col in prob_cols if col == f"{c}_probability"]
            ordered.append(match[0] if match else prob_cols[i])
        return pred[ordered].to_numpy(dtype=float)

    def predict_proba_if_native(self, X):
        return self.predict_raw_scores(X)


def build_dl_adapter(name: str, entry: ModelEntry, dl_cfg=None) -> DLAdapter:
    return DLAdapter(name, entry, dl_cfg)


def build_adapter(name: str, entry: ModelEntry, dl_cfg=None):
    if name in DL_MODELS:
        return build_dl_adapter(name, entry, dl_cfg)
    est = build_estimator(name, entry)
    caps = probe_capabilities(est, name, entry)
    return SklearnAdapter(name, est, caps)


def build_registry(cfg: ModelsConfig, dl_cfg=None) -> dict:
    entries = {**cfg.baselines, **cfg.models}
    return {name: build_adapter(name, entry, dl_cfg) for name, entry in entries.items()}


def capabilities_matrix(registry: dict) -> pd.DataFrame:
    rows = []
    for name, adapter in registry.items():
        c = adapter.capabilities
        cost_ok = "local_cost_sensitive" in c.supported_imbalance
        rows.append(
            {
                "name": name,
                "native_probability": c.native_probability,
                "decision_function": c.decision_function,
                "sample_weight": c.sample_weight,
                "nonnegative_input": c.nonnegative_input,
                "native_categorical": c.native_categorical,
                "gpu": c.gpu,
                "deterministic": c.deterministic,
                "calibrator_required": c.calibrator_required,
                "preprocess_profile": c.preprocess_profile,
                "supported_imbalance": ";".join(c.supported_imbalance),
                "cost_sensitive_status": "supported" if cost_ok else "not_run_incompatible",
                "tuning_tier": c.tuning_tier,
            }
        )
    return pd.DataFrame(rows)
