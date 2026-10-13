"""Reconcile corrected artifacts and build an editorial release candidate.

Does not activate the collaboration release_manifest. All training runs stay immutable.
"""
import json
from pathlib import Path
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tb_outcomes.cohort import apply_eligibility, build_cohort, load_outcome_rules, map_outcome
from tb_outcomes.config import load_config
from tb_outcomes.dl_parallel import file_hash, atomic_json
from tb_outcomes.features import classify_feature_scope

OLD = Path('data/reanalysis/c7_20260923')
NEW = Path('data/reanalysis/c7_20261004')
OUT = NEW / 'reporting_release'
RELEASE = 'smarttb-c7-reporting-20261004'
LABELS = {'0': 'cure', '1': 'tb_death', '2': 'treatment_interruption'}
ARMS = ['individual', 'municipal', 'without_diagnostics', 'random_group', 'k27', 'k75',
        'contemporary', 'hiv_uncertain', 'all_cause']


def verify(path):
    doc = json.loads(path.read_text())
    for name, digest in doc['files'].items():
        if file_hash(path.parent / name) != digest:
            raise ValueError('Artifact changed: ' + str(path.parent / name))
    return doc


def manual_f1(frame, rule, classes):
    scores = []
    for c in classes:
        tp = ((frame.y_true == c) & (frame[rule] == c)).sum()
        den = (frame.y_true == c).sum() + (frame[rule] == c).sum()
        scores.append(2 * tp / den if den else 0.)
    return float(np.mean(scores))


def main():
    OUT.mkdir(exist_ok=False)
    artifacts = []
    def add(path, arm, population, rule='not_applicable', aggregation='metadata', labels=None, **extra):
        path = Path(path)
        artifacts.append(dict(path=str(path), sha256=file_hash(path), arm=arm, population=population,
                              decision_rule=rule, aggregation=aggregation, run_id=RELEASE,
                              class_labels=labels or LABELS, **extra))
    locations = {'primary': (OLD/'primary_selected', OLD/'primary_audit_v2'),
                 'temporal': (OLD/'temporal_selected', OLD/'temporal_audit'),
                 'clinical': (OLD/'clinical_selected', OLD/'clinical_audit_v2')}
    locations.update({a: (NEW/a/'selected', NEW/a/'audit') for a in ARMS})
    overview, checks, coverage, costs = [], [], [], []
    for arm, (selected, audit) in locations.items():
        sm, am = verify(selected/'manifest.json'), verify(audit/'audit_manifest.json')
        classes = sorted(int(c) for c in am['classes'])
        n = am['n_expected']
        population = ('2024, revised retrospective temporal evaluation' if arm == 'temporal' else
                      '2015-2023, all-cause outcome cohort' if arm == 'all_cause' else '2015-2023, primary outcome cohort')
        population += f'; n={n}'
        p = pd.read_parquet(selected/'selected_predictions.parquet')
        assert len(p) == n and p.source_row_id.nunique() == n
        row = dict(arm=arm, n=n, experiment_id=sm['experiment_id'], families=len({x['model'] for x in sm['expected_pairs']}),
                   pairs=len(sm['expected_pairs']), folds=p.outer_fold.nunique(), class_labels=am['classes'])
        for c in classes: row['class_'+str(c)+'_n'] = int(p.y_true.eq(c).sum())
        for rule in ['pred_argmax', 'pred_policy']:
            f1 = manual_f1(p, rule, classes)
            ci = pd.read_csv(audit/f'cluster_intervals_{rule}.csv')
            cell = ci.query("comparison=='nested_selection_procedure' and kind=='estimate' and metric=='f1_macro'").iloc[0]
            assert abs(f1-cell.estimate) < 1e-12
            per_fold = [manual_f1(g, rule, classes) for _, g in p.groupby('outer_fold')]
            row[rule+'_pooled_f1'] = f1
            row[rule+'_fold_mean_f1'] = float(np.mean(per_fold))
            row[rule+'_fold_sd_f1'] = float(np.std(per_fold, ddof=1)) if len(per_fold)>1 else None
            row[rule+'_pooled_ci_low'], row[rule+'_pooled_ci_high'] = cell.ci_low, cell.ci_high
            checks.append(dict(arm=arm,rule=rule,n=n,manual_f1=f1,audit_f1=cell.estimate,passed=True))
        overview.append(row)
        cov = pd.read_csv(selected/'coverage.csv');cov['arm']=arm;coverage.append(cov)
        cost = pd.read_csv(audit/'compute_cost.csv');cost['arm']=arm;costs.append(cost)
        for directory, meta in [(audit,am), (selected,sm)]:
            for name in meta['files']:
                if Path(name).suffix not in ['.csv','.json']: continue
                rule = 'pred_argmax' if 'pred_argmax' in name else 'pred_policy' if 'pred_policy' in name else 'explicit row-level stage/rule; probability metrics independent of classification rule'
                agg = ('per_outer_fold' if name in ['selected_metrics_per_fold.csv','recomputed_fold_metrics.csv','selected_by_fold.json'] else
                       'completed_job_wall_seconds_not_total_campaign_elapsed' if name=='compute_cost.csv' else
                       'pooled_records_or_explicit_group; see aggregation/rule columns' if name.endswith('.csv') else 'metadata')
                add(directory/name,arm,population,rule,agg,am['classes'],source_experiment_id=sm['experiment_id'])
            add(directory/('audit_manifest.json' if directory==audit else 'manifest.json'),arm,population,
                labels=am['classes'],source_experiment_id=sm['experiment_id'])
    assert sum(len(c) for c in costs)==2212, 'Unexpected checkpoint inventory'
    pd.DataFrame(overview).to_json(OUT/'arm_overview.json',orient='records',indent=2)
    pd.DataFrame(checks).to_csv(OUT/'independent_f1_checks.csv',index=False)
    pd.concat(coverage,ignore_index=True).to_csv(OUT/'candidate_coverage.csv',index=False)
    pd.concat(costs,ignore_index=True).to_csv(OUT/'compute_cost_all_arms.csv',index=False)
    for arm in ARMS:
        if arm=='all_cause': continue
        directory=NEW/'comparisons'/arm;meta=verify(directory/'manifest.json')
        for name in [*meta['files'],'manifest.json']:
            add(directory/name, 'primary_common_universe_vs_'+arm, '2015-2023; identical 707870 IDs/targets',
                'pred_argmax' if 'pred_argmax' in name else 'pred_policy' if 'pred_policy' in name else 'explicit rule or probability threshold',
                'pooled_records; paired primary-k50 geographic cluster bootstrap; common candidate universe')
    for directory,arm,pop in [(OLD/'primary_vs_clinical','primary_vs_clinical','2015-2023; n=707870'),
                             (NEW/'age_race_stratification','age_race_stratification','primary cohort; race/age completeness subset; see coverage'),
                             (NEW/'baseline_confusions','baselines','2015-2023; n=707870 per pair'),
                             (NEW/'attribution','attribution','external class-balanced samples of primary selected folds')]:
        meta=verify(directory/'manifest.json')
        for name in [*meta['files'],'manifest.json']:
            if Path(name).suffix not in ['.json','.csv']: continue
            add(directory/name,arm,pop,'raw_margin_precalibration' if arm=='attribution' else 'explicit row-level rule',
                'separate fold/family/sample; not population weighted' if arm=='attribution' else 'explicit stratum or pooled_records')

    run=Path('data/reanalysis/c6_20260919/primary_cpu/a0a12b4c53f318709c0c')
    rm=json.loads((run/'run.json').read_text())
    for name in ['src/tb_outcomes/cohort.py','src/tb_outcomes/config.py','src/tb_outcomes/maturation.py']:
        assert file_hash(name)==rm['sources'][name],name
    assert file_hash('data/harmonized.parquet')==rm['harmonized_sha256']
    assert file_hash(run/'inputs.pkl')==rm['input_sha256']
    with (run/'inputs.pkl').open('rb') as f:bundle=pickle.load(f)
    cfg=load_config(run/'runtime/configs/analysis_decisions.yaml')
    rules=load_outcome_rules(run/'runtime/configs/outcomes.yaml')
    df=pd.read_parquet('data/harmonized.parquet')
    cohort=build_cohort(df,cfg,rules,cfg.study.primary_target_schema)
    cohort.flow.to_csv(OUT/'corrected_cohort_flow.csv',index=False)
    cohort.exclusions.to_csv(OUT/'corrected_eligibility_exclusions.csv',index=False)
    analytic=cohort.analytic
    expected=set(analytic.loc[analytic.NU_ANO.lt(2024)].index.astype(str))
    assert expected==set(bundle['record_ids'])
    temporal=pd.read_parquet(OLD/'temporal_selected/selected_predictions.parquet',columns=['source_row_id'])
    assert set(analytic.loc[analytic.NU_ANO.eq(2024)].index.astype(str))==set(temporal.source_row_id)
    eligible,_=apply_eligibility(df,cfg)
    target=map_outcome(eligible,cfg.study.primary_target_schema,rules)
    eligible=eligible.assign(analytic_included=eligible.index.isin(analytic.index),mapped_outcome=target,
                             study_role=np.where(eligible.NU_ANO.lt(2024),'development',np.where(eligible.NU_ANO.eq(2024),'temporal_test','unused_future')))
    counts=[]
    for col in ['NU_ANO','REGIAO','FORMA','CS_SEXO','CS_RACA','AGRAVAIDS','SITUA_ENCE']:
        for (role,value), g in eligible.groupby(['study_role',col],dropna=False,observed=True):
            inc=int(g.analytic_included.sum());counts.append(dict(role=role,axis=col,group=str(value),eligible=len(g),included=inc,excluded=len(g)-inc,excluded_fraction=(len(g)-inc)/len(g)))
    pd.DataFrame(counts).to_csv(OUT/'included_excluded_by_group.csv',index=False)
    eligible.groupby(['study_role','SITUA_ENCE','mapped_outcome','analytic_included'],dropna=False,observed=True).size().rename('n').reset_index().to_csv(OUT/'outcome_selection_counts.csv',index=False)
    scopes=classify_feature_scope(list(bundle['X']),bundle['specs'])
    specs={s.raw_name:s for s in bundle['specs']}
    features=[dict(name=name,scope=scopes[name],source=specs[name].source if name in specs else 'derived',
                   evidence=specs[name].dictionary_evidence if name in specs else 'derived feature; frozen features.py',
                   availability=specs[name].availability if name in specs else 'derived_from_baseline_inputs') for name in bundle['X']]
    pd.DataFrame(features).to_csv(OUT/'primary_feature_dictionary.csv',index=False)
    assert len(features)==49 and sum(x['scope']=='municipal' for x in features)==22
    decl=json.loads(Path('data/reanalysis/c6_20260919/declaration.json').read_text())
    pd.DataFrame(decl['unsupported']).to_csv(OUT/'primary_incompatible_pairs.csv',index=False)
    allcause=build_cohort(df,cfg,rules,'all_cause_3class').analytic
    allcause_counts=allcause.groupby('NU_ANO')['target_all_cause_3class'].value_counts().rename('n').reset_index()
    allcause_counts.to_csv(OUT/'all_cause_population_counts.csv',index=False)
    assert len(allcause.loc[allcause.NU_ANO.lt(2024)])==742830
    atomic_json(OUT/'population_reconciliation.json',dict(primary_development=len(expected),temporal_test=len(temporal),
        unused_future=int(analytic.NU_ANO.gt(2024).sum()),total_analytic=len(analytic),eligible=len(eligible),
        extraction_date=cfg.study.data_extraction_date,primary_features=49,individual_features=27,municipal_features=22,
        source_harmonized_sha256=rm['harmonized_sha256'],source_input_sha256=rm['input_sha256'],
        script_sha256=file_hash(__file__),checkpoint_count=2212,
        note='Initial flow covers declared 2015-2025; 2025 is excluded from all predictive evaluations.'))
    for p in OUT.iterdir():
        if p.is_file():add(p,'cross_arm_reporting','see explicit arm/role columns and population_reconciliation.json',
                          'explicit row-level rule or not_applicable','explicit row-level aggregation; pooled CI never attached to fold mean')
    for name in ['data/correction_audit/partitions/manifest.json','data/correction_audit/partitions/population_counts.csv',
                 'data/correction_audit/maturation/manifest.json','data/correction_audit/maturation/by_year_region_form.csv',
                 'data/correction_audit/baseline_context.json','data/reanalysis/c6_20260919/declaration.json',
                 'data/reanalysis/c6_remaining_20260923/declaration.json',
                 'data/reanalysis/c7_20260923/gb_amendment_audit.json',
                 'data/reanalysis/c6_remaining_20260923/singleton_recovery_20261001/amendment.json',
                 'data/reanalysis/c7_20261004/full_tests_v2.xml',
                 'data/reanalysis/c7_20261004/completion_integrity_check.json',
                 str(run/'runtime/configs/models.yaml'),str(run/'runtime/configs/outcomes.yaml')]:
        add(name,'protocol_and_provenance','explicit source-specific population; not interchangeable')
    candidate=dict(schema_version=1,status='ready_for_reporting',owner='codex',run_id=RELEASE,
        run_id_semantics='Editorial release bundle identity; source_experiment_id/source manifests preserve distinct fitted runs. Not a single training run.',
        created_at=datetime.now(timezone.utc).isoformat(),artifacts=artifacts,class_labels=LABELS,
        limitations=['Ready for manuscript preparation only; C8 and final visual/editorial acceptance remain pending',
                     '2024 historically explored; revised temporal evaluation is not pristine independent confirmation',
                     'Conditional outcome-selected cohort; not population absolute risk or fixed-horizon prognosis',
                     'Pooled bootstrap intervals are not confidence intervals for the unweighted fold mean',
                     'Geographic bootstrap conditional on fixed predictions; unequal clusters and exploratory multiplicity limit inference',
                     'No causal, prospective utility, equivalence or global superiority claims',
                     'Compute cost covers successful jobs, excludes failed attempts and is not elapsed campaign duration',
                     'Attribution explains raw margins separately by fold/family, before calibration',
                     'Never use historical default data/*.csv files; only catalogued artifacts may supply manuscript numbers'],
        note='Source manifests remain immutable and retain historical not_released labels; this catalog grants use only for the listed files. All-cause artifacts have their own class_labels.')
    atomic_json(OUT/'release_candidate.json',candidate)
    print(json.dumps(dict(artifacts=len(artifacts),arms=len(locations),checks=len(checks),checkpoints=2212,out=str(OUT))))


if __name__=='__main__':main()
