"""Piece 11: basketball players are NBA draft prospects (football players stay high school recruits).

Draft prospects get a big-board rank, a projected round and pick, a current team
(college, overseas club, NBL Next Stars, Overtime Elite, G League), a class year
and an age. Their stars are high school background.
"""
import csv
import json
from collections import Counter

import pytest
from pydantic import ValidationError

from recruit_gen.cli import generate, unused_columns, write_csv
from recruit_gen.export import story_hints
from recruit_gen.locks import build_locks
from recruit_gen.models import SportConfig
from recruit_gen.pipeline import run_pipeline
from recruit_gen.report import class_report


@pytest.fixture(scope="module")
def board(basketball):
    """A big pile of prospects across the whole pool (realistic mix)."""
    return [r for seed in range(4) for r in run_pipeline(1000, seed, basketball)]


@pytest.fixture(scope="module")
def tops(basketball):
    """Many top-60 boards."""
    return [r for seed in range(25) for r in run_pipeline(60, seed, basketball, top=True)]


def cli(**changes):
    options = dict(sport="basketball", count=None, seed=None, top=None, position=None, ethnicity=None,
                   origin=None, stars=None, projection=None, archetype=None, development=None, height=None,
                   weight=None, first=None, last=None, spec_file=None, distinctive="normal", out=None, show=0,
                   json_out=True, report=False, config=None)
    options.update(changes)
    generate(**options)


# ---- settings ----------------------------------------------------------------------
def test_sports_are_framed_correctly(basketball, football):
    assert basketball.draft is not None and basketball.teams is not None
    assert football.draft is None and football.teams is None and football.ages


def test_projection_bands_must_cover_the_pool(basketball):
    d = basketball.model_dump()
    d["draft"]["projection"][-1]["last_rank"] = 500
    with pytest.raises(ValidationError, match="projection"):
        SportConfig.model_validate(d)


def test_unknown_path_is_rejected(basketball):
    d = basketball.model_dump()
    d["teams"]["paths"]["US"] = {"College": 0.9, "Junior Hockey": 0.1}
    with pytest.raises(ValidationError, match="Junior Hockey"):
        SportConfig.model_validate(d)


# ---- the big board -------------------------------------------------------------------
def test_projection_and_pick_follow_rank(board, basketball):
    bands = basketball.draft.projection
    for r in board:
        first = 1
        for b in bands:
            if r["rank"] <= b.last_rank:
                assert r["projection"] == b.label and r["rank"] >= first
                break
            first = b.last_rank + 1
        assert r["projected_pick"] == (r["rank"] if r["rank"] <= 60 else None)


def test_full_pool_has_the_right_number_per_round(basketball):
    counts = Counter(r["projection"] for r in run_pipeline(1000, 3, basketball))
    assert counts == {"Lottery": 14, "1st round": 16, "2nd round": 30, "Undrafted": 940}


def test_draft_classes_default_to_top_of_the_board(tmp_path):
    out = tmp_path / "b.csv"
    cli(count=20, seed=1, out=out)
    assert [int(r["rank"]) for r in csv.DictReader(open(out))] == list(range(1, 21))
    cli(count=20, seed=1, out=out, top=False)                              # --mix
    assert [int(r["rank"]) for r in csv.DictReader(open(out))] != list(range(1, 21))


def test_football_still_defaults_to_a_mix(tmp_path):
    out = tmp_path / "f.csv"
    cli(sport="football", count=20, seed=1, out=out)
    assert [int(r["rank"]) for r in csv.DictReader(open(out))] != list(range(1, 21))


# ---- high school stars are background ---------------------------------------------------
def test_international_prospects_have_no_recruiting_stars(board, basketball):
    countries = basketball.places.countries
    for r in board:
        if r["origin"] in countries:
            assert r["stars"] is None and r["rating"] is None
        else:
            assert r["stars"] in (2, 3, 4, 5) and 70 <= r["rating"] <= 100


def test_top_prospects_were_mostly_top_recruits(tops):
    lottery = [r["stars"] for r in tops if r["rank"] <= 14 and r["stars"] is not None]
    assert sum(s == 5 for s in lottery) / len(lottery) > 0.55
    assert sum(s <= 3 for s in lottery) / len(lottery) < 0.06              # a few late risers


def test_early_bloomers_were_rated_higher_in_high_school(board):
    def avg(dev):
        s = [r["stars"] for r in board if r["development"] == dev and r["stars"]]
        return sum(s) / len(s)
    assert avg("Early Bloomer") > avg("Steady Climber") > avg("Late Bloomer")


def test_projection_lock(basketball):
    for r in run_pipeline(14, 2, basketball, locks=build_locks(basketball, projection="lottery")):
        assert r["projection"] == "Lottery"


# ---- where they play now ----------------------------------------------------------------
def test_every_prospect_has_a_team_and_age(board, basketball):
    college = {t for tier in basketball.teams.college_tiers.values() for t in tier.teams}
    for r in board:
        assert r["team"] and r["level"] and 17 <= r["age"] <= 24
        if r["level"] == "College":
            assert r["team"] in college and r["class_year"] in ("Fr", "So", "Jr", "Sr")
            assert abs(r["age"] - basketball.teams.college_age[r["class_year"]]) <= 1
        else:
            assert r["class_year"] == ""


def test_most_prospects_play_college(board):
    levels = Counter(r["level"] for r in board)
    assert 0.80 < levels["College"] / len(board) < 0.95
    assert {"Overseas pro", "NBL Next Stars", "Overtime Elite", "G League"} <= set(levels)


def test_international_prospects_mostly_play_at_home(board, basketball):
    abroad = [r for r in board if r["origin"] in basketball.places.countries
              and r["origin"] not in ("CAN", "AUS", "GBR")]
    assert sum(r["level"] == "Overseas pro" for r in abroad) / len(abroad) > 0.45
    clubs = basketball.teams.overseas_clubs
    for r in abroad:
        if r["level"] == "Overseas pro":
            assert r["team"] in (clubs.get(r["origin"]) or clubs["other"])


def test_top_prospects_are_mostly_freshmen(tops, board):
    def fresh(rs):
        college = [r for r in rs if r["level"] == "College"]
        return sum(r["class_year"] == "Fr" for r in college) / len(college)
    assert fresh([r for r in tops if r["rank"] <= 14]) > 0.6               # one-and-done
    assert fresh([r for r in board if r["rank"] > 300]) < 0.4


def test_late_bloomers_stay_in_school_longer(board):
    def upper(dev):
        college = [r for r in board if r["development"] == dev and r["level"] == "College"]
        return sum(r["class_year"] in ("Jr", "Sr") for r in college) / len(college)
    assert upper("Late Bloomer") > upper("Early Bloomer") + 0.08


def test_best_prospects_land_at_big_programs(tops, board, basketball):
    blue = set(basketball.teams.college_tiers["blue_blood"].teams)

    def share(rs):
        college = [r for r in rs if r["level"] == "College"]
        return sum(r["team"] in blue for r in college) / len(college)
    assert share([r for r in tops if r["rank"] <= 30]) > 3 * share([r for r in board if r["rank"] > 300])


def test_football_recruits_get_an_age_but_no_team(football):
    for r in run_pipeline(200, 1, football):
        assert r["age"] in (17, 18, 19) and r["team"] == "" and r["projection"] == ""


# ---- output files ----------------------------------------------------------------------
def test_csv_columns_fit_the_sport(basketball, football, tmp_path):
    assert {"forty", "dev_trait"} <= unused_columns(basketball)
    assert not {"team", "projection", "age", "wingspan"} & unused_columns(basketball)
    assert {"team", "class_year", "projection", "projected_pick", "wingspan"} <= unused_columns(football)
    assert "age" not in unused_columns(football)
    out = tmp_path / "b.csv"
    write_csv(run_pipeline(10, 1, basketball, top=True), out, basketball)
    header = next(csv.reader(open(out)))
    assert {"projection", "projected_pick", "team", "level", "class_year", "age"} <= set(header)
    assert "forty" not in header


def test_report_shows_projection(basketball):
    report = class_report(run_pipeline(300, 1, basketball), basketball, top=False, custom=False)
    assert abs(sum(line.target for line in report["Projection"]) - 1) < 1e-9
    assert all(line.target is None for line in report["HS stars"])


def test_story_hints_talk_about_the_draft(tops, basketball):
    hints = [h for r in tops for h in story_hints(r, basketball)]
    assert any("one-and-done" in h for h in hints)
    assert any("recruit in high school" in h for h in hints)
    assert any("Already a pro" in h or "Skipped college" in h for h in hints)


def test_json_has_a_career_section(tmp_path):
    out = tmp_path / "b.csv"
    cli(count=5, seed=3, out=out)
    rec = json.loads(out.with_suffix(".json").read_text())["recruits"][0]
    assert rec["career"]["team"] and rec["career"]["age"]
    assert rec["scouting"]["projection"] == "Lottery" and rec["scouting"]["projected_pick"] == 1
