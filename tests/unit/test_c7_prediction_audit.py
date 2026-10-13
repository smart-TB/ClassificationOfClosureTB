import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('c7_audit', Path('scripts/audit_corrected_predictions.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    frame = pd.DataFrame(dict(source_row_id=['a','b','c'], y_true=[0,1,2], outer_fold=[0,0,0],
                              pred_argmax=[0,1,2], pred_policy=[0,1,2],
                              proba_0=[.8,.1,.1],proba_1=[.1,.8,.1],proba_2=[.1,.1,.8]))
    return frame, frame[['source_row_id','y_true','outer_fold']].copy()


@pytest.mark.parametrize('change',['identity','truth','fold','duplicate','probability','argmax','policy'])
def test_audit_rejects_corrupt_predictions(change):
    frame, expected = fixture()
    if change == 'identity': frame.loc[0,'source_row_id']='d'
    elif change == 'truth': frame.loc[0,'y_true']=1
    elif change == 'fold': frame.loc[0,'outer_fold']=1
    elif change == 'duplicate': frame.loc[0,'source_row_id']='b'
    elif change == 'probability': frame.loc[0,'proba_0']=.7
    elif change == 'argmax': frame.loc[0,'pred_argmax']=1
    elif change == 'policy': frame.loc[0,'pred_policy']=3
    with pytest.raises(ValueError): audit.check_frame(frame,expected,[0,1,2],'test')


def test_order_does_not_change_identity_validation():
    frame, expected = fixture()
    audit.check_frame(frame.iloc[::-1],expected,[0,1,2],'test')


def test_constant_calibration_reports_unidentifiable_slope_and_correct_citl():
    y=np.tile([1,0,0,0],100)
    values=audit.probability_metrics(y,np.full(len(y),.5))
    assert values['calibration_in_large'] == pytest.approx(np.log(1/3))
    assert np.isnan(values['calibration_slope'])
    assert values['auprc'] == .25
    assert values['auroc'] == .5


def test_sparse_events_suppress_discrimination_and_calibration():
    values=audit.probability_metrics([1]+[0]*99,np.full(100,.1),minimum=30)
    assert values['status']=='not_estimable'
    assert np.isnan(values['auroc']) and np.isnan(values['calibration_in_large'])


def test_capacity_ties_are_invariant_to_input_order():
    ids=np.array([str(i).zfill(3) for i in range(100)])
    y=np.arange(100)%3==0
    p=np.full(100,.5)
    a=audit.capacity_rows(y,p,ids)
    b=audit.capacity_rows(y[::-1],p[::-1],ids[::-1])
    assert a==b


def test_paired_bootstrap_with_single_cluster_in_one_outer_fold():
    frame=pd.DataFrame({'source_row_id':np.arange(60),'y_true':np.tile([0,1,2],20)})
    frame['pred_argmax']=frame.y_true
    mapping=frame[['source_row_id']].copy()
    mapping['cluster']=np.repeat(np.arange(10),6)
    mapping['outer_fold']=np.where(mapping.cluster.eq(0),0,1)
    result=audit.inference({'a':frame,'b':frame.iloc[::-1]},mapping,[0,1,2],reference='a',replicates=100)
    assert result.stratified_by.isna().all()
    assert result.query("kind=='paired_difference'")['estimate'].eq(0).all()
    influence=audit.cluster_influence(frame,mapping,[0,1,2],'pred_argmax')
    assert all(row['change']==0 for row in influence)
    assert sum(row['n_records'] for row in influence)==60
