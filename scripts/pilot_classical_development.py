"""Piloto exploratório com treino/calibração/avaliação espacial só antes de 2024.

Não seleciona nem publica um vencedor. Os resultados não são desempenho final.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight

from tb_outcomes.calibration import classify_policy, fit_calibrator, fit_thresholds, load_calibration_config
from tb_outcomes.cli import _load_benchmark_inputs
from tb_outcomes.decision_search import fit_joint_macro_policy
from tb_outcomes.executor import build_model_input, load_executor_config
from tb_outcomes.imbalance import apply_strategy, load_imbalance_config
from tb_outcomes.models import build_adapter, load_models_config
from tb_outcomes.preprocess import infer_column_types, load_preprocess_config
from tb_outcomes.splits import make_outer_folds


def main():
    out = Path('data/classical_review/pilot')
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_executor_config('configs/executor.yaml')
    X, y, strata, clusters, _, classes, specs, labels, years = _load_benchmark_inputs(
        cfg, False, with_year=True)
    postmortem = pd.to_numeric(X['TRATAMENTO'], errors='coerce').eq(6).fillna(False)
    eligible = (years < 2024).fillna(False) & ~postmortem
    idx = np.flatnonzero(eligible.to_numpy(dtype=bool))
    rng = np.random.RandomState(20260919)
    idx = np.sort(rng.choice(idx, min(60000, len(idx)), replace=False))
    X, y, strata, clusters, years = [a.iloc[idx].reset_index(drop=True)
                                    for a in (X, y, strata, clusters, years)]
    # Sensibilidade conservadora: não usar sorologia cuja disponibilidade basal é incerta.
    removed = [c for c in ('HIV', 'hiv_pos_model') if c in X]
    X = X.drop(columns=removed)
    fold = make_outer_folds(X, clusters, y, 3, 'binpack')
    tr, ca, va = [np.flatnonzero(fold.to_numpy() == i) for i in range(3)]
    assert not (set(clusters.iloc[tr]) & set(clusters.iloc[ca]))
    assert not (set(clusters.iloc[tr]) & set(clusters.iloc[va]))
    assert not (set(clusters.iloc[ca]) & set(clusters.iloc[va]))
    assert years.max() < 2024
    (out/'design.json').write_text(json.dumps({
        'purpose':'exploratory_development_only', 'n':len(X), 'n_features':X.shape[1],
        'years':[int(years.min()),int(years.max())], 'labels':labels,
        'partitions':{'train':len(tr),'calibration':len(ca),'development_validation':len(va)},
        'removed_features':removed, 'postmortem_excluded':True, 'seed':20260919,
        'note':'Split piloto 3 vias, não substitui validação aninhada completa. Sem teste de 2024.'
    },indent=2))
    entries = load_models_config('configs/models.yaml').models
    pre = load_preprocess_config('configs/preprocess.yaml')
    imb = load_imbalance_config('configs/imbalance.yaml')
    cal = load_calibration_config('configs/calibration.yaml')
    variants = [
        ('lightgbm','reference',{'n_jobs':4}),
        ('lightgbm','regularized_capacity',{'n_jobs':4,'n_estimators':600,'learning_rate':.03,
                                          'num_leaves':15,'min_child_samples':80,'reg_lambda':2.0}),
        ('hist_gradient_boost','reference',{}),
        ('hist_gradient_boost','regularized_capacity',{'max_iter':300,'learning_rate':.05,
                                                     'max_leaf_nodes':15,'l2_regularization':2.0}),
    ]
    rows = []
    for model, variant, params in variants:
        for strategy in ('none','local_cost_sensitive','global_balanced'):
            started = time.perf_counter()
            ad = build_adapter(model, entries[model], random_state=42, params=params)
            types = infer_column_types(X.iloc[tr], specs, pre.max_cardinality)
            evaluation = np.r_[ca,va]
            xt, xe = build_model_input(X.iloc[tr], X.iloc[evaluation], types,
                                       ad.capabilities.preprocess_profile, ad.input_kind, pre)
            if strategy == 'global_balanced':
                weights = compute_sample_weight('balanced', y.iloc[tr])
            else:
                res = apply_strategy(y.iloc[tr],strata.iloc[tr],strategy,imb,42,True)
                weights = res.sample_weight
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                ad.fit(xt, y.iloc[tr].to_numpy(), sample_weight=weights)
                raw = ad.predict_raw_scores(xe)
            raw_cal, raw_val = raw[:len(ca)], raw[len(ca):]
            calibration = fit_calibrator(raw_cal, y.iloc[ca], cal)
            pca, pva = calibration.transform(raw_cal), calibration.transform(raw_val)
            legacy = fit_thresholds(pca, y.iloc[ca], cal)
            joint = fit_joint_macro_policy(pca, y.iloc[ca], classes, legacy)
            death = labels.index('tb_death')
            methods = [('raw_argmax',raw_val,raw_val.argmax(axis=1)),
                       ('calibrated_argmax',pva,pva.argmax(axis=1)),
                       ('legacy_policy',pva,classify_policy(pva,legacy,classes)),
                       ('joint_policy',pva,classify_policy(pva,joint,classes))]
            for rule, probs, pred in methods:
                row = dict(model=model,variant=variant,strategy=strategy,rule=rule,
                    f1_macro=f1_score(y.iloc[va],pred,average='macro',labels=classes),
                    death_recall=recall_score(y.iloc[va]==death,pred==death,zero_division=0),
                    death_precision=precision_score(y.iloc[va]==death,pred==death,zero_division=0),
                    death_auprc=average_precision_score(y.iloc[va]==death,probs[:,death]),
                    log_loss=log_loss(y.iloc[va],probs,labels=classes),
                    seconds=time.perf_counter()-started, params=json.dumps(params),
                    warnings=json.dumps(sorted({str(w.message) for w in caught})),
                    effective_iterations=getattr(ad._est,'n_iter_',None))
                rows.append(row)
            pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
            print(json.dumps({k:rows[-1][k] for k in ('model','variant','strategy','f1_macro','death_recall','seconds')}),flush=True)


if __name__ == '__main__':
    main()
