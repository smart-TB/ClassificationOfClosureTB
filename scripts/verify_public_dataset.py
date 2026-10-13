"""Verify the public cohort without training or changing the input snapshot."""
from pathlib import Path
import json,hashlib
import pandas as pd
from tb_outcomes.config import load_config
from tb_outcomes.cohort import build_cohort,load_outcome_rules
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'datasets/2026-10-13';m=json.loads((D/'manifest.json').read_text())
p=D/'harmonized.parquet';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['files'][0]['sha256']
df=pd.read_parquet(p);assert df.shape==(1326538,168)
cfg=load_config(ROOT/'configs/analysis_decisions.yaml')
c=build_cohort(df,cfg,load_outcome_rules(ROOT/'configs/outcomes.yaml'),cfg.study.primary_target_schema)
a=c.analytic;years=pd.to_numeric(a.NU_ANO);counts={'development':int(years.lt(2024).sum()),'temporal_2024':int(years.eq(2024).sum()),'unused_2025':int(years.eq(2025).sum())}
assert counts=={'development':707870,'temporal_2024':92152,'unused_2025':42228},counts
print(json.dumps({'source_shape':list(df.shape),'analytic_n':len(a),'populations':counts,'status':'passed'},indent=2))
