import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_corrected_predictions import probability_metrics
from tb_outcomes.dl_parallel import file_hash
r=Path('data/reanalysis/c7_20261004/reporting_release')
p=Path('data/reanalysis/c7_20260923/primary_selected/selected_predictions.parquet')
s=json.loads((p.parent/'manifest.json').read_text());assert file_hash(p)==s['files'][p.name]
d=pd.read_parquet(p);classes=[0,1,2]
assert np.allclose(d[[f'raw_score_{c}' for c in classes]].sum(axis=1),1)
rows=[];bins=[]
for stage,prefix in [('before_calibration','raw_score_'),('after_calibration','proba_')]:
 for c in classes:
  y=d.y_true.eq(c).to_numpy();prob=d[prefix+str(c)].to_numpy()
  rows.append(dict(stage=stage,class_code=c,**probability_metrics(y,prob)))
  group=np.minimum((prob*10).astype(int),9)
  for b in range(10):
   mask=group==b;n=int(mask.sum());bins.append(dict(stage=stage,class_code=c,bin=b,lower=b/10,upper=(b+1)/10,n=n,mean_probability=float(prob[mask].mean()) if n else None,event_rate=float(y[mask].mean()) if n else None))
pd.DataFrame(rows).to_csv(r/'primary_calibration_before_after.csv',index=False)
pd.DataFrame(bins).to_csv(r/'primary_calibration_bins.csv',index=False)
candidate=json.loads((r/'release_candidate.json').read_text())
candidate['artifacts']=[a for a in candidate['artifacts'] if Path(a['path']).name not in ['primary_calibration_before_after.csv','primary_calibration_bins.csv']]
for name in ['primary_calibration_before_after.csv','primary_calibration_bins.csv']:
 p=r/name
 candidate['artifacts'].append(dict(path=str(p),sha256=file_hash(p),arm='primary',population='2015-2023; n=707870',decision_rule='probabilities before/after calibration; no classification rule',aggregation='pooled_records; calibration bins descriptive without confidence intervals',run_id=candidate['run_id'],class_labels=candidate['class_labels'],script_sha256=file_hash(__file__)))
(r/'release_candidate.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n')
print(pd.DataFrame(rows)[['stage','class_code','ece_fixed_10','calibration_in_large','calibration_slope']].to_string(index=False))
