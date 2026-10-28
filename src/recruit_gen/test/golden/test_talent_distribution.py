import numpy as np
import json
from hypothesis import given, strategies as st

from recruit_gen.pipeline import (
    run_pipeline
)


def test_talent_pyramid_shape():
    recruits = run_pipeline(count=1000, seed = 42)

    # print(recruits[0])
    talent_scores = [r['talent'] for r in recruits]

    median = np.percentile(talent_scores, 50)
    p99 = np.percentile(talent_scores, 99)


    # print(f"median={median:.2f}  p99={p99:.2f}  ratio={p99/median:.2f}")

    # assert p99 / median > 0.25  
    assert p99 / median > 4.0


def test_golden_seed42():
    fresh = run_pipeline(count=80, seed=42)

    with open("tests/golden/basketball_seed42.json") as f:
        golden = json.load(f)

    assert fresh == golden



# @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
# def test_every_recruit_has_positive_height(seed):
#     recruits = run_pipeline(seed=seed, count=20)

#     for recruit in recruits:
#         assert recruit['physicals'] > 0


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_talent_in_unit_range(seed):
    recruits = run_pipeline(seed=seed, count=20)

    for recruit in recruits:
        assert 0 <= recruit['talent'] < 1
        # assert 0 <= recruit['talent'] < 0.5   # deliberately wrong. talent can exceed 0.5