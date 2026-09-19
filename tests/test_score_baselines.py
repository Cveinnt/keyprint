import importlib.util
from pathlib import Path
import math
import pytest

spec=importlib.util.spec_from_file_location("score_baselines",Path(__file__).parents[1]/"tools/compare_score_baselines.py")
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)


def test_uniform_recovers_count_statistic_and_scale_is_irrelevant():
    bits=[[1,0,1],[1,1,1]]
    assert baseline.score(bits,[1,1,1])==pytest.approx((2*5-6)/math.sqrt(6))
    assert baseline.score(bits,[10,5,1])==pytest.approx(baseline.score(bits,[20,10,2]))
    assert baseline.score([[0,1],[1,0]],[10,1])==0


@pytest.mark.parametrize("bits,weights", [([], [1]), ([[1,2]],[1,1]), ([[1,0]],[0,0]), ([[1,0]],[-1,1]), ([[1,0]],[1,float('nan')])])
def test_invalid_or_unavailable_is_not_a_negative_score(bits,weights):
    with pytest.raises(ValueError):baseline.score(bits,weights)
