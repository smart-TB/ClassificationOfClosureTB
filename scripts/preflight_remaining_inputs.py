"""Check correction-arm inputs and maturation equivalence without model fitting."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tb_outcomes.cli import _load_benchmark_inputs
from tb_outcomes.cohort import build_cohort, load_outcome_rules
from tb_outcomes.config import load_config
from tb_outcomes.dl_parallel import file_hash, atomic_json
from tb_outcomes.executor import load_executor_config

root=Path('data/reanalysis/c6_remaining_20260923')
cfg=load_executor_config('configs/executor.yaml')
doc=json.loads((root/'declaration.json').read_text())
baseline=None
rows=[]
for arm in doc['arms']:
    opts=arm['options']
    altered=cfg.model_copy(update={k:v for k,v in opts.items() if k in {'fold_method','outcome_schema'}})
    inputs=_load_benchmark_inputs(altered,False,k_override=opts.get('k',50),
        feature_scope=opts.get('scope','combined'),with_year=True,with_record_id=True,
        context_profile=opts.get('context_profile','baseline'),
        availability_profile=opts.get('availability_profile','primary'),population_role='development')
    X,y,strata,clusters,outer,classes,specs,labels,years,ids=inputs
    identity=pd.DataFrame({'id':ids,'y':y})
    if baseline is None:
        baseline=identity
    if opts.get('outcome_schema') != 'all_cause_3class':
        assert identity.equals(baseline),f"Population changed unexpectedly: {arm['name']}"
    else:
        assert set(baseline.id)<=set(ids), 'All-cause sensitivity lost primary cohort records'
    assert years.between(2015,2023).all()
    assert not ids.duplicated().any() and not ids.isna().any()
    assert not clusters.isna().any()
    assert pd.DataFrame({'cluster':clusters,'fold':outer}).groupby('cluster').fold.nunique().eq(1).all()
    for fold in sorted(outer.unique()):
        assert set(y[outer.eq(fold)])==set(classes), 'External fold lacks a class'
    rows.append(dict(arm=arm['name'],n=len(ids),features=list(X.columns),n_features=X.shape[1],
        class_counts={labels[c]:int(y.eq(c).sum()) for c in classes},
        clusters=int(clusters.nunique()),folds=int(outer.nunique()),
        years=sorted(int(v) for v in years.unique()),status='passed'))
    print(json.dumps({k:rows[-1][k] for k in ['arm','n','n_features','clusters','status']}),flush=True)
    del inputs,X,y,strata,clusters,outer,classes,specs,labels,years,ids,identity
    gc.collect()
study=load_config('configs/analysis_decisions.yaml')
source=pd.read_parquet('data/harmonized.parquet')
rules=load_outcome_rules('configs/outcomes.yaml')
ordinary=build_cohort(source,study,rules,cfg.outcome_schema).analytic
alternate=study.model_copy(update={'study':study.study.model_copy(update={'maturation_rule':'calendar_by_form'})})
calendar=build_cohort(source,alternate,rules,cfg.outcome_schema).analytic
equivalence=[]
for name,lo,hi in [('development',2015,2023),('temporal',2024,2024)]:
    a=ordinary.loc[ordinary.NU_ANO.between(lo,hi)]
    b=calendar.loc[calendar.NU_ANO.between(lo,hi)]
    assert a.index.equals(b.index), f'Maturation changes {name}'
    target='target_'+cfg.outcome_schema
    assert a[target].equals(b[target])
    equivalence.append(dict(population=name,n=len(a),same_ids=True,same_labels=True,
        interpretation='No change in eligible records/targets; repeat training would be identical under fixed features/partitions/seeds.'))
atomic_json(root/'input_preflight.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),
    declaration_sha256=file_hash(root/'declaration.json'),source_sha256=file_hash('data/harmonized.parquet'),
    arms=rows,maturation_equivalence=equivalence,status='passed'))
