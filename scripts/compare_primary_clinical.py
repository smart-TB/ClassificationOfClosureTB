"""Paired conditional comparison of the full and clinical selection procedures."""
import json
from pathlib import Path
import pickle

import pandas as pd

from tb_outcomes.cluster_inference import paired_cluster_bootstrap
from tb_outcomes.dl_parallel import atomic_json, file_hash
from tb_outcomes.utility import decision_curve

root=Path('data/reanalysis/c7_20260923')
out=root/'primary_vs_clinical'
out.mkdir(exist_ok=False)
sources={}
frames={}
for name,dirname in [('primary','primary_selected'),('clinical','clinical_selected')]:
    manifest=root/dirname/'manifest.json'
    doc=json.loads(manifest.read_text())
    path=root/dirname/'selected_predictions.parquet'
    assert file_hash(path)==doc['files'][path.name]
    frames[name]=pd.read_parquet(path)
    sources[name]={'manifest_sha256':file_hash(manifest),'predictions_sha256':file_hash(path),'experiment_id':doc['experiment_id']}
run=Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
manifest=json.loads((run/'run.json').read_text())
assert file_hash(run/'inputs.pkl')==manifest['input_sha256']
with (run/'inputs.pkl').open('rb') as f:
    bundle=pickle.load(f)
mapping=pd.DataFrame({'source_row_id':bundle['record_ids'],'cluster':bundle['clusters']})
for rule in ['pred_argmax','pred_policy']:
    result=paired_cluster_bootstrap(frames,mapping,bundle['classes'],rule=rule,
                                    reference='clinical',replicates=2000,strata_col=None)
    result.to_csv(out/f'paired_{rule}.csv',index=False)
curves=[]
for name,frame in frames.items():
    for c in bundle['classes']:
        if bundle['labels'][c]=='cure':
            continue
        curve=decision_curve(frame.y_true.eq(c).astype(int),frame[f'proba_{c}'])
        curve['comparison']=name
        curve['class_code']=c
        curve['class_label']=bundle['labels'][c]
        curves.append(curve)
pd.concat(curves,ignore_index=True).to_csv(out/'decision_curves.csv',index=False)
atomic_json(out/'manifest.json',dict(status='not_released',sources=sources,
    script_sha256=file_hash(__file__), files={p.name:file_hash(p) for p in out.iterdir()},
    decision_rules=['pred_argmax','pred_policy'],aggregation='pooled_records',
    limitations=['Conditional on fixed predictions; no refitting uncertainty',
                 'Pairs cluster bootstrap without fold strata; unequal clusters limit interval reliability',
                 'Decision curves assume hypothetical threshold trade-offs; not observed intervention benefit',
                 'Different internally selected procedures compared on the same external IDs, never selected from these differences']))
print(out,flush=True)
