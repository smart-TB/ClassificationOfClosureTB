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
import logging
import tempfile
import time
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


def build_estimator(name: str, entry: ModelEntry, *, random_state=None, params=None):
    if name in DL_MODELS:
        raise ValueError(f"{name} é DL; use o DLAdapter")
    estimator = _SKLEARN_FACTORIES[name]()
    overrides = dict(params or {})
    valid = set(estimator.get_params()) | set(inspect.signature(type(estimator).__init__).parameters)
    invalid = set(overrides) - valid
    if invalid:
        raise ValueError(f"Parâmetros desconhecidos de {name}: {sorted(invalid)}")
    if random_state is not None:
        if name == "catboost":
            overrides["random_seed"] = random_state
        elif "random_state" in estimator.get_params():
            overrides["random_state"] = random_state
    estimator.set_params(**overrides)
    return estimator


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
        # XGBoost 2.1 não recodifica categorias no predict. Preservar o mapa
        # aprendido no treino evita que 'b' mude de código se 'a' faltar no lote.
        if self.name == "xgboost":
            self._category_dtypes = {c: X[c].dtype for c in self._category_columns(X)}
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
        X = self._align_categories(X)
        if self.capabilities.native_probability:
            return self._est.predict_proba(X)
        return self._est.decision_function(X)

    def predict_proba_if_native(self, X):
        if not self.capabilities.native_probability:
            return None
        return self._est.predict_proba(self._align_categories(X))

    def _align_categories(self, X):
        dtypes = getattr(self, "_category_dtypes", {})
        if not dtypes:
            return X
        X = X.copy()
        for column, dtype in dtypes.items():
            # Categoria inédita vira NaN e segue o caminho missing do XGBoost;
            # não atribuir o código de uma categoria conhecida arbitrária.
            X[column] = pd.Categorical(X[column].astype(object), dtype=dtype)
        return X


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
        torch.set_num_threads(cfg.torch_num_threads)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        use_gpu = torch.cuda.is_available()
        self._device = torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
        if use_gpu and cfg.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16-mixed solicitado em GPU sem suporte a BF16")
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
            validation_split=cfg.validation_split, num_workers=cfg.num_workers,
            dataloader_kwargs=cfg.dataloader_kwargs())
        rate = {} if cfg.learning_rate is None else {"learning_rate": cfg.learning_rate}
        model_config = getattr(ptm, self._MODEL_CONFIGS[self.name])(task="classification", **rate)
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
        # Um processo por GPU, escolhido ANTES do import de torch pelo launcher.
        # Cada fit tem seu diretório: modelos/dobras concorrentes não compartilham
        # saved_models. Mantém load_best=True; limpar só DEPOIS da restauração.
        started = time.perf_counter()
        if cfg.checkpoint_root:
            Path(cfg.checkpoint_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tb-dl-", dir=cfg.checkpoint_root) as ckpt:
            trainer_config = TrainerConfig(
                max_epochs=cfg.max_epochs, early_stopping="valid_loss",
                early_stopping_mode="min", early_stopping_patience=cfg.early_stopping_patience,
                accelerator="gpu" if use_gpu else "cpu", devices=1,
                precision=cfg.precision if use_gpu else "32-true", seed=cfg.seed,
                batch_size=cfg.batch_size, progress_bar="none",
                checkpoints_path=ckpt, checkpoints_kwargs={"save_weights_only": True},
                trainer_kwargs={"enable_model_summary": False})
            self._model = TabularModel(
                data_config=data_config, model_config=model_config,
                optimizer_config=OptimizerConfig(), trainer_config=trainer_config,
                verbose=False)
            # Sem CSVLogger compartilhado em lightning_logs/version_N; as métricas
            # científicas e os tempos são persistidos pelo executor/coordenador.
            self._model.logger = False
            # fit tem seed=42 por default na biblioteca; propagar a semente configurada.
            self._model.fit(train=train, seed=cfg.seed)
        logging.getLogger("tb_outcomes").info(
            "[dl-fit] model=%s n=%d device=%s precision=%s epochs=%d seconds=%.3f",
            self.name, len(train), str(self._device), trainer_config.precision,
            self._model.trainer.current_epoch, time.perf_counter() - started)
        return self

    def _impute(self, X):
        if not self._cont_cols:
            return X
        X = X.copy()
        X[self._cont_cols] = X[self._cont_cols].fillna(self._cont_medians).fillna(0.0)
        return X

    def predict_raw_scores(self, X) -> np.ndarray:
        # Aumentar SOMENTE o lote de inferência não altera o lote do otimizador.
        old_batch = self._model.datamodule.batch_size
        try:
            self._model.datamodule.batch_size = self._cfg().inference_batch_size or old_batch
            # Lightning devolve a rede à CPU no teardown do fit. Sem device explícito,
            # predict() da biblioteca executava toda a inferência tabular na CPU.
            pred = self._model.predict(self._impute(X), device=self._device)
        finally:
            self._model.datamodule.batch_size = old_batch
        prob_cols = [c for c in pred.columns if c.endswith("_probability")]
        ordered = []
        for i, c in enumerate(self._classes):
            match = [col for col in prob_cols if col == f"{c}_probability"]
            ordered.append(match[0] if match else prob_cols[i])
        return pred[ordered].to_numpy(dtype=float)

    def predict_proba_if_native(self, X):
        return self.predict_raw_scores(X)

    def close(self):
        """Libera referências cíclicas de Trainer/DataModule entre ajustes independentes."""
        import gc
        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_dl_adapter(name: str, entry: ModelEntry, dl_cfg=None) -> DLAdapter:
    return DLAdapter(name, entry, dl_cfg)


def build_adapter(name: str, entry: ModelEntry, dl_cfg=None, *, random_state=None, params=None):
    if name in DL_MODELS:
        if params:
            raise ValueError("Overrides de estimadores são exclusivos dos modelos tradicionais")
        return build_dl_adapter(name, entry, dl_cfg)
    est = build_estimator(name, entry, random_state=random_state, params=params)
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
