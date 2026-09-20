import copy
import importlib.util
from pathlib import Path
import sys

import pytest
from scipy.stats import beta

tools = Path(__file__).parents[1]/"tools"
sys.path.insert(0,str(tools))
spec = importlib.util.spec_from_file_location("null_audit",tools/"audit_predictability_null.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    rows = [{"source_index":i,"unavailable":False,"family_reference_tail":p,"flagged":p<=.01,
             "baseline_flagged":False} for i,p in enumerate((.009,.02,.2))]
    summary = {"status":"completed","attempts":3,"errors":0,"unavailable":0,"fatal":None,
               "hits":1,"baseline_hits":0,"iid_only_upper_97_5":float(beta.ppf(.975,2,2))}
    return rows,summary


def test_independent_document_count_and_one_sided_bound():
    rows,summary = fixture()
    assert audit.validate_summary(rows,summary,{0,1,2})["controls"]==3
    bad = {**summary,"iid_only_upper_97_5":float(beta.ppf(.975,2,5))}
    with pytest.raises(ValueError,match="reconcile"):
        audit.validate_summary(rows,bad,{0,1,2})


@pytest.mark.parametrize("change",["duplicate","missing","substituted","error","unavailable","flag","nan","total"])
def test_omission_or_changed_outcomes_cannot_be_approved(change):
    rows,summary = fixture()
    rows = copy.deepcopy(rows)
    if change=="duplicate": rows[2]=copy.deepcopy(rows[1])
    if change=="missing": rows.pop()
    if change=="substituted": rows[2]["source_index"]=99
    if change=="error": rows[2]["error"]={"type":"Failure"}
    if change=="unavailable": rows[2]["unavailable"]=True
    if change=="flag": rows[0]["flagged"]=False
    if change=="nan": rows[0]["family_reference_tail"]=float("nan")
    if change=="total": summary["hits"]=0
    with pytest.raises(ValueError):
        audit.validate_summary(rows,summary,{0,1,2})
