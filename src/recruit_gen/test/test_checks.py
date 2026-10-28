"""Piece 7: the clean-up check - contract, realism, duplicates, rejects, safety stop."""
import csv

import pytest

from recruit_gen.checks import (
    ClassCheckError, contract_problems, duplicate_problems, problem_kind, realism_problems, validate_dedup,
)
from recruit_gen.cli import rejects_path, write_rejects
from recruit_gen.pipeline import build_class
from recruit_gen.names import plain


def tightened(basketball, **checks):
    """A copy of the basketball config with stricter (or looser) check settings."""
    return basketball.model_copy(update={"checks": basketball.checks.model_copy(update=checks)})


@pytest.fixture(scope="module")
def good_player(basketball):
    return dict(build_class(1, seed=1, config=basketball).recruits[0])


# ---- single-player checks --------------------------------------------------------
def test_a_normal_player_passes_everything(basketball, good_player):
    assert contract_problems(good_player) == []
    assert realism_problems(good_player, basketball) == []


def test_contract_problems_name_the_field(good_player):
    assert any(r.startswith("height") for r in contract_problems({**good_player, "height": 200}))
    below = {**good_player, "potential": good_player["ovr"] - 1}
    assert any("below ovr" in r for r in contract_problems(below))
    missing = {k: v for k, v in good_player.items() if k != "last_name"}
    assert any(r.startswith("last_name") for r in contract_problems(missing))


def test_unrealistic_body_for_the_position_is_caught(basketball, good_player):
    lo, hi = basketball.checks.body[good_player["position"]].height
    reasons = realism_problems({**good_player, "height": int(hi) + 1}, basketball)
    assert reasons and "height" in reasons[0] and "outside" in reasons[0]


def test_odd_wingspan_is_caught(basketball, good_player):
    lo, hi = basketball.checks.wingspan_vs_height
    reasons = realism_problems({**good_player, "wingspan": good_player["height"] + hi + 2}, basketball)
    assert reasons and "wingspan" in reasons[0]


def test_duplicates_are_caught(good_player):
    name = plain(f'{good_player["first_name"]} {good_player["last_name"]}')
    reasons = duplicate_problems(good_player, {name}, {good_player["rank"]})
    assert any("duplicate name" in r for r in reasons)
    assert any("duplicate rank" in r for r in reasons)


@pytest.mark.parametrize("reason, kind", [
    ("PG height 76 in outside 74-74", "PG height"),
    ("wingspan +13 in vs height outside -2 to +11", "wingspan"),
    ("duplicate name Jalen Smith", "duplicate name"),
    ("height: Input should be less than or equal to 96", "height"),
])
def test_problems_are_grouped_for_the_summary(reason, kind):
    assert problem_kind(reason) == kind


# ---- what happens to failures ----------------------------------------------------
def test_failures_are_recorded_and_replaced(basketball, good_player):
    broken = {**good_player, "height": 200}
    fixed = {**good_player}
    result = validate_dedup([broken], basketball, rebuild=lambda failed, kept: fixed)
    assert result.recruits == [fixed]                       # class size kept
    assert len(result.rejects) == 1
    assert result.rejects[0]["rank"] == good_player["rank"]
    assert "height" in result.rejects[0]["reasons"]


def test_one_odd_player_in_a_small_class_is_fine(basketball):
    result = build_class(10, seed=8, config=basketball, top=True)
    assert len(result.recruits) == 10


def test_endless_failures_stop_with_a_clear_message(basketball, good_player):
    broken = {**good_player, "height": 200}
    cfg = tightened(basketball, max_rebuilds=2, max_reject_share=1.0)
    with pytest.raises(ClassCheckError, match="after 3 tries"):
        validate_dedup([broken], cfg, rebuild=lambda failed, kept: broken)


def test_real_classes_replace_the_odd_player(basketball):
    result = build_class(1000, seed=4, config=basketball)
    assert len(result.recruits) == 1000
    assert result.rejects                                    # ~1% of players are odd
    assert len({x["rank"] for x in result.rejects}) / 1000 < basketball.checks.max_reject_share
    assert sorted(r["rank"] for r in result.recruits) == sorted({r["rank"] for r in result.recruits})


def test_replacements_keep_ranks_and_ranks_stay_consistent(basketball):
    result = build_class(1000, seed=4, config=basketball)
    replaced = {x["rank"] for x in result.rejects}
    ranks = {r["rank"] for r in result.recruits}
    assert replaced <= ranks                                 # the rank was refilled, not dropped
    for key, field in (("position", "pos_rank"), ("origin", "state_rank")):
        for value in {r[key] for r in result.recruits}:
            group = sorted((r for r in result.recruits if r[key] == value), key=lambda r: r["rank"])
            assert [r[field] for r in group] == list(range(1, len(group) + 1))


def test_every_finished_player_passes_every_check(basketball):
    result = build_class(1000, seed=5, config=basketball)
    names = [plain(f'{r["first_name"]} {r["last_name"]}') for r in result.recruits]
    assert len(names) == len(set(names))
    for r in result.recruits:
        assert contract_problems(r) == [] and realism_problems(r, basketball) == []


def test_a_bad_setting_trips_the_safety_stop(basketball):
    # Pretend every PG must be exactly 6'2": most PGs fail -> the run should stop, not hide it.
    body = {**basketball.checks.body, "PG": basketball.checks.body["PG"].model_copy(update={"height": (74, 74)})}
    with pytest.raises(ClassCheckError, match=r"failed the clean-up check.*PG height \(\d+\)"):
        build_class(300, seed=6, config=tightened(basketball, body=body))


def test_same_seed_same_rejects(basketball):
    a = build_class(500, seed=7, config=basketball)
    b = build_class(500, seed=7, config=basketball)
    assert a.recruits == b.recruits and a.rejects == b.rejects


# ---- the rejects file ------------------------------------------------------------
def test_rejects_file_lists_who_and_why(basketball, tmp_path):
    result = build_class(1000, seed=4, config=basketball)
    path = rejects_path(tmp_path / "class.csv")
    assert path.name == "class_rejects.csv"
    write_rejects(result.rejects, path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.rejects)
    assert {"attempt", "rank", "position", "reasons"} <= set(rows[0])
    assert all(row["reasons"] for row in rows)
