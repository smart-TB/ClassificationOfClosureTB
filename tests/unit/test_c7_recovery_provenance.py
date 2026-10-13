import json
import pytest
from audit_corrected_predictions import recovery_provenance
from tb_outcomes.dl_parallel import file_hash


def test_recovery_identity_and_amendment_hash(tmp_path):
    p=tmp_path/'amendment.json';p.write_text(json.dumps({'jobs':['job1']}))
    digest=file_hash(p)
    meta={'job':{'id':'job1'},'run_id':'base','effective_execution_id':'base+'+digest[:12],
          'recovery_amendment':{'path':str(p),'sha256':digest}}
    assert recovery_provenance(meta)==meta['recovery_amendment']
    with pytest.raises(ValueError,match='identity'):
        recovery_provenance({**meta,'effective_execution_id':'wrong'})
    with pytest.raises(ValueError,match='outside'):
        recovery_provenance({**meta,'job':{'id':'job2'}})
    p.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        recovery_provenance(meta)


def test_unamended_job_remains_unamended():
    assert recovery_provenance({'job':{'id':'original'}}) is None
