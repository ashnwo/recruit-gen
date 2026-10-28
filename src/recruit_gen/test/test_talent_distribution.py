import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from recruit_gen.models import Recruit, TalentConfig
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import class_shape, rating_for, verdict_for

GOLDEN = Path(__file__).parent / "basketball_seed42.json"   # sits next to this test file


def talent_dict(basketball, **changes) -> dict:
    d = basketball.talent.model_dump()
    d.update(changes)
    return d


# ---- talent config contract ----------------------------------------------------
def test_real_talent_config_is_valid(basketball):
    assert [b.stars for b in basketball.talent.stars] == [5, 4, 3, 2]


def test_star_bands_must_go_best_to_worst(basketball):
    d = talent_dict(basketball)
    d["stars"][0]["last_rank"], d["stars"][1]["last_rank"] = d["stars"][1]["last_rank"], d["stars"][0]["last_rank"]
    with pytest.raises(ValidationError, match="best to worst"):
        TalentConfig.model_validate(d)


def test_star_bands_must_cover_the_pool(basketball):
    d = talent_dict(basketball)
    d["stars"][-1]["last_rank"] = d["pool_size"] - 1
    with pytest.raises(ValidationError, match="cover the whole pool"):
        TalentConfig.model_validate(d)


def test_verdict_extremes_must_be_bigger(basketball):
    d = talent_dict(basketball)
    d["verdicts"]["steal"] = 0.5          # smaller than underrated (0.75)
    with pytest.raises(ValidationError):
        TalentConfig.model_validate(d)


def test_rating_range_must_go_best_to_worst(basketball):
    d = talent_dict(basketball)
    d["stars"][0]["rating"] = [98, 100]
    with pytest.raises(ValidationError):
        TalentConfig.model_validate(d)


def test_pool_cant_be_bigger_than_population(basketball):
    with pytest.raises(ValidationError):
        TalentConfig.model_validate(talent_dict(basketball, population=10))


# ---- class shape: who's in the class -----------------------------------------
def test_top_mode_is_the_best_n(basketball):
    out = class_shape(30, np.random.default_rng(1), basketball, top=True)
    assert [r["rank"] for r in out] == list(range(1, 31))


def test_mix_mode_picks_different_ranks_from_the_pool(basketball):
    out = class_shape(300, np.random.default_rng(2), basketball)
    ranks = [r["rank"] for r in out]
    assert len(set(ranks)) == 300                      # nobody ranked twice
    assert ranks == sorted(ranks)                      # best first
    assert 1 <= ranks[0] and ranks[-1] <= basketball.talent.pool_size


def test_count_bigger_than_pool_is_refused(basketball):
    with pytest.raises(ValueError, match="ranked"):
        class_shape(basketball.talent.pool_size + 1, np.random.default_rng(3), basketball)


def test_talent_pyramid_shape(basketball):
    """Top of the rankings = big talent gaps; deep in the rankings = tiny gaps."""
    z = {r["rank"]: r["scout_z"] for r in class_shape(1000, np.random.default_rng(4), basketball, top=True)}
    top_gap = z[1] - z[10]
    deep_gap = z[500] - z[510]
    assert top_gap > 20 * deep_gap


# ---- stars and ratings ---------------------------------------------------------
def test_full_pool_has_exactly_the_configured_stars(basketball):
    recruits = run_pipeline(basketball.talent.pool_size, seed=5, config=basketball)
    counts = Counter(r["stars"] for r in recruits)
    first = 0
    for band in basketball.talent.stars:
        assert counts[band.stars] == band.last_rank - first
        first = band.last_rank


def test_realistic_mix_is_mostly_low_stars(basketball):
    counts = Counter(r["stars"] for seed in range(20) for r in run_pipeline(100, seed=seed, config=basketball))
    total = sum(counts.values())
    assert 0.01 < counts[5] / total < 0.045          # ~2.5% five-stars
    assert counts[2] > counts[3] > counts[4] > counts[5]


def test_ratings_follow_the_bands(basketball):
    bands = basketball.talent.stars
    assert rating_for(1, bands) == 100
    assert rating_for(25, bands) == 98
    assert rating_for(26, bands) == 97
    assert rating_for(1000, bands) == 70
    ratings = [rating_for(r, bands) for r in range(1, 1001)]
    assert ratings == sorted(ratings, reverse=True)   # better rank never gets a lower rating


# ---- hidden true talent and verdicts -----------------------------------------
@pytest.mark.parametrize("true_rank, rank, expected", [
    (10, 100, "Steal"), (50, 100, "Underrated"), (100, 100, "As Advertised"),
    (200, 100, "Overrated"), (500, 100, "Bust"),
    (2, 1, "As Advertised"),     # 2x worse, but only 1 spot apart (min_gap)
    (6, 3, "As Advertised"),     # 2x worse, only 3 spots apart
    (9, 3, "Overrated"),         # 3x worse and 6 spots apart
    (12, 3, "Bust"),             # 4x worse and 9 spots apart
])
def test_verdict_labels(basketball, true_rank, rank, expected):
    assert verdict_for(true_rank, rank, basketball.talent) == expected


def test_verdict_mix_is_believable(basketball):
    v = Counter(r["verdict"] for seed in range(5) for r in run_pipeline(1000, seed=seed, config=basketball))
    total = sum(v.values())
    assert 0.35 < v["As Advertised"] / total < 0.60
    for label in ("Steal", "Bust"):
        assert 0.02 < v[label] / total < 0.12, label
    for label in ("Underrated", "Overrated"):
        assert 0.10 < v[label] / total < 0.30, label


def test_tiny_slips_at_the_top_are_as_advertised(basketball):
    min_gap = basketball.talent.verdicts.min_gap
    for seed in range(30):
        for r in run_pipeline(10, seed=seed, config=basketball, top=True):
            if abs(r["true_rank"] - r["rank"]) < min_gap:
                assert r["verdict"] == "As Advertised"


def test_steals_really_rank_better_and_busts_worse(basketball):
    for r in run_pipeline(1000, seed=8, config=basketball):
        if r["verdict"] == "Steal":
            assert r["true_rank"] < r["rank"]
        if r["verdict"] == "Bust":
            assert r["true_rank"] > r["rank"]


def test_perfect_scouts_mean_no_surprises(basketball):
    perfect = basketball.model_copy(update={"talent": basketball.talent.model_copy(update={"scout_error": 0.0})})
    for r in run_pipeline(300, seed=6, config=perfect):
        assert r["true_rank"] == r["rank"] and r["verdict"] == "As Advertised"


def test_position_ranks_follow_national_rank(basketball):
    recruits = run_pipeline(200, seed=7, config=basketball)
    for pos in basketball.positions:
        group = [r for r in recruits if r["position"] == pos]
        assert [r["pos_rank"] for r in group] == list(range(1, len(group) + 1))
        assert [r["rank"] for r in group] == sorted(r["rank"] for r in group)


# ---- whole pipeline ------------------------------------------------------------
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
        Recruit.model_validate(recruit)
        assert recruit["position"] in basketball.positions
        assert "scout_z" not in recruit            # working values don't leak into output


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_count_matches_request(basketball, seed):
    assert len(run_pipeline(seed=seed, count=37, config=basketball)) == 37
    assert len(run_pipeline(seed=seed, count=37, config=basketball, top=True)) == 37
