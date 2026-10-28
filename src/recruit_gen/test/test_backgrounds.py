"""Piece 4: where players come from, ethnicity, high school, state rank."""
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from recruit_gen.models import ETHNICITIES, SportConfig
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import backgrounds, origins


def config_dict(basketball) -> dict:
    return basketball.model_dump()


def force_origin(basketball, origin, international, position="SF", stars=3, n=3000, seed=1):
    """Run the backgrounds stage on players we pin to one place."""
    info = (basketball.places.countries if international else basketball.places.states)[origin]
    city = next(iter(info.cities))
    recruits = [{"rank": i + 1, "position": position, "stars": stars, "origin": origin,
                 "international": international, "city": city} for i in range(n)]
    return backgrounds(recruits, np.random.default_rng(seed), basketball)


# ---- config rules ----------------------------------------------------------------
def test_places_file_loaded(basketball):
    assert len(basketball.places.states) >= 51            # 50 states + DC (+ territories)
    assert {"NGA", "CAN", "SRB"} <= set(basketball.places.countries)


def test_unknown_origin_code_is_rejected(basketball):
    d = config_dict(basketball)
    d["origins"]["states"]["XX"] = 5
    with pytest.raises(ValidationError, match="not found in places"):
        SportConfig.model_validate(d)


def test_position_ethnicity_mix_must_add_to_one(basketball):
    d = config_dict(basketball)
    d["positions"]["PG"]["ethnicity"]["Black"] = 0.5
    with pytest.raises(ValidationError, match="add up to 1"):
        SportConfig.model_validate(d)


def test_unknown_ethnicity_label_is_rejected(basketball):
    d = config_dict(basketball)
    d["positions"]["PG"]["ethnicity"] = {"Martian": 1.0}
    with pytest.raises(ValidationError):
        SportConfig.model_validate(d)


def test_international_needs_countries(basketball):
    d = config_dict(basketball)
    d["origins"]["countries"] = {}
    with pytest.raises(ValidationError, match="no countries"):
        SportConfig.model_validate(d)


# ---- origins stage ----------------------------------------------------------------
def test_international_share_matches_config(basketball):
    out = origins([{} for _ in range(20_000)], np.random.default_rng(2), basketball)
    share = sum(r["international"] for r in out) / len(out)
    assert abs(share - basketball.origins.international_rate) < 0.015


def test_hotbed_states_produce_the_most(basketball):
    out = origins([{} for _ in range(20_000)], np.random.default_rng(3), basketball)
    counts = Counter(r["origin"] for r in out if not r["international"])
    top3 = [code for code, _ in counts.most_common(3)]
    biggest = sorted(basketball.origins.states, key=basketball.origins.states.get, reverse=True)[:3]
    assert top3[0] == biggest[0]
    assert len(set(top3) & set(biggest)) >= 2


def test_hometown_is_a_real_city_in_that_place(basketball):
    for r in origins([{} for _ in range(2_000)], np.random.default_rng(4), basketball):
        places = basketball.places
        info = places.countries[r["origin"]] if r["international"] else places.states[r["origin"]]
        assert r["city"] in info.cities
        suffix = info.name if r["international"] else r["origin"]
        assert r["hometown"] == f"{r['city']}, {suffix}"


# ---- ethnicity -------------------------------------------------------------------
def test_us_ethnicity_mix_is_realistic(basketball):
    recruits = [r for seed in range(3) for r in run_pipeline(1000, seed=seed, config=basketball)]
    us = Counter(r["ethnicity"] for r in recruits if len(r["origin"]) == 2)
    total = sum(us.values())
    assert 0.70 < us["Black"] / total < 0.85
    assert 0.13 < us["White"] / total < 0.27


def test_hawaii_leans_polynesian(basketball):
    def poly_share(state):
        out = force_origin(basketball, state, international=False, n=5000)
        return sum(r["ethnicity"] == "Polynesian" for r in out) / len(out)
    assert poly_share("HI") > 0.25
    assert poly_share("HI") > 20 * poly_share("OH")


def test_georgia_leans_black(basketball):
    counts = Counter(r["ethnicity"] for r in force_origin(basketball, "GA", international=False))
    assert counts["Black"] / sum(counts.values()) > 0.75


@pytest.mark.parametrize("country, only", [("NGA", "Black"), ("SRB", "White"), ("LTU", "White")])
def test_countries_use_their_own_mix(basketball, country, only):
    assert {r["ethnicity"] for r in force_origin(basketball, country, international=True, n=500)} == {only}


def test_centers_are_more_often_white_than_point_guards(basketball):
    def white_share(pos):
        out = force_origin(basketball, "OH", international=False, position=pos, n=5000, seed=5)
        return sum(r["ethnicity"] == "White" for r in out) / len(out)
    assert white_share("C") > white_share("PG") + 0.08


# ---- high school and state rank ---------------------------------------------------
def test_five_stars_go_to_prep_school_far_more_often(basketball):
    prep_schools = set(basketball.prep.schools)

    def prep_share(stars):
        out = force_origin(basketball, "TX", international=False, stars=stars, n=5000, seed=6)
        return sum(r["high_school"] in prep_schools for r in out) / len(out)

    p = basketball.prep
    assert abs(prep_share(5) - min(0.9, p.rate * p.star_boost[5])) < 0.03
    assert abs(prep_share(2) - min(0.9, p.rate * p.star_boost[2])) < 0.03
    assert prep_share(5) > 3 * prep_share(2)


def test_local_schools_use_the_hometown(basketball):
    prep_schools = set(basketball.prep.schools)
    for r in force_origin(basketball, "GA", international=False, stars=2, n=500, seed=7):
        assert r["high_school"]
    intl = force_origin(basketball, "NGA", international=True, n=500, seed=8)
    local = [r for r in intl if r["high_school"] not in prep_schools]
    assert local and all(r["high_school"] == f"{r['city']} Academy" for r in local)


def test_state_ranks_follow_national_rank(basketball):
    recruits = run_pipeline(300, seed=9, config=basketball)
    for origin in {r["origin"] for r in recruits}:
        group = [r for r in recruits if r["origin"] == origin]
        assert [r["state_rank"] for r in group] == list(range(1, len(group) + 1))


def test_every_ethnicity_label_is_allowed(basketball):
    for r in run_pipeline(1000, seed=10, config=basketball):
        assert r["ethnicity"] in ETHNICITIES
