import json
from pathlib import Path

import numpy as np
from hypothesis import given, strategies as st

from recruit_gen.models import Recruit
from recruit_gen.pipeline import run_pipeline

GOLDEN = Path(__file__).parent / "basketball_seed42.json"   # sits next to this test file


def test_talent_pyramid_shape(basketball):
    recruits = run_pipeline(count=1000, seed=42, config=basketball)
    talent_scores = [r["talent"] for r in recruits]
    median = np.percentile(talent_scores, 50)
    p99 = np.percentile(talent_scores, 99)
    assert p99 / median > 4.0


def test_golden_seed42(basketball):
    """Same seed + same config must give the exact same class. If you change the
    pipeline or configs/basketball.yaml on purpose, refresh the golden file:
        uv run python src/recruit_gen/test/update_golden.py"""
    fresh = run_pipeline(count=80, seed=42, config=basketball)
    golden = json.loads(GOLDEN.read_text())
    assert fresh == golden


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_recruit_meets_the_contract(basketball, seed):
    for recruit in run_pipeline(seed=seed, count=20, config=basketball):
        Recruit.model_validate(recruit)          # sane height/weight, talent 0-1, all fields
        assert recruit["position"] in basketball.positions


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_count_matches_request(basketball, seed):
    assert len(run_pipeline(seed=seed, count=37, config=basketball)) == 37
