import json
from pathlib import Path

import numpy as np
from hypothesis import given, strategies as st

from recruit_gen.models import parse_physicals_config
from recruit_gen.pipeline import run_pipeline

CONFIG = parse_physicals_config({"loc": 79.0, "scale": 2.5, "distribution": "normal"})
GOLDEN = Path(__file__).parent / "basketball_seed42.json"   # sits next to this test file


def test_talent_pyramid_shape():
    recruits = run_pipeline(count=1000, seed=42, config=CONFIG)
    talent_scores = [r["talent"] for r in recruits]
    median = np.percentile(talent_scores, 50)
    p99 = np.percentile(talent_scores, 99)
    assert p99 / median > 4.0


def test_golden_seed42():
    """Same seed must give the exact same class. If you change the pipeline on
    purpose, refresh the golden file:  python src/recruit_gen/test/update_golden.py"""
    fresh = run_pipeline(count=80, seed=42, config=CONFIG)
    golden = json.loads(GOLDEN.read_text())
    assert fresh == golden


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_recruit_has_a_real_body(seed):
    for recruit in run_pipeline(seed=seed, count=20, config=CONFIG):
        assert recruit["height"] > 0
        assert recruit["weight"] > 0


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_talent_in_unit_range(seed):
    for recruit in run_pipeline(seed=seed, count=20, config=CONFIG):
        assert 0 <= recruit["talent"] < 1


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_count_matches_request(seed):
    assert len(run_pipeline(seed=seed, count=37, config=CONFIG)) == 37
