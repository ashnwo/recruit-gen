"""Piece 10: class check report, JSON lore export, and full command-line runs."""
import csv
import json

import pytest

from recruit_gen.cli import generate
from recruit_gen.export import key_ratings, lore_json, story_hints
from recruit_gen.locks import build_locks
from recruit_gen.pipeline import run_pipeline
from recruit_gen.report import Line, class_report, format_report, position_targets


def cli(**changes):
    """Call the command the way the terminal would, with every option spelled out."""
    options = dict(sport="basketball", count=None, seed=None, top=False, position=None, ethnicity=None,
                   origin=None, stars=None, projection=None, archetype=None, development=None, height=None, weight=None,
                   first=None, last=None, spec_file=None, distinctive="normal", out=None, show=0,
                   json_out=True, report=True, config=None)
    options.update(changes)
    generate(**options)


# ---- class check report ----------------------------------------------------------
@pytest.mark.parametrize("sport", ["basketball", "football"])
def test_targets_add_up(sport, request):
    config = request.getfixturevalue(sport)
    report = class_report(run_pipeline(300, 1, config), config, top=False, custom=False)
    for name in ("Ethnicity", "Positions", "Projection" if config.draft else "Stars"):
        assert abs(sum(line.target for line in report[name]) - 1) < 0.02, name


def test_a_normal_class_is_on_target(basketball):
    n = 900                       # basketball ranks 1,000 players
    report = class_report(run_pipeline(n, 2, basketball), basketball, top=False, custom=False)
    flagged = [(k, line.label) for k, lines in report.items() for line in lines if line.off(n)]
    assert flagged == []


def test_flags_only_real_gaps():
    assert not Line("x", 0.22, 0.20).off(100)        # small class, small gap: chance
    assert Line("x", 0.40, 0.20).off(1000)           # big class, big gap: flagged
    assert not Line("x", 0.215, 0.20).off(100_000)   # tiny gap never flagged (needs 3+ points)


def test_top_classes_expect_fewer_kickers(football):
    top = position_targets(list(range(1, 51)), football)
    deep = position_targets(list(range(3000, 3051)), football)
    assert top["K"] < deep["K"] / 5


def test_custom_requests_show_no_targets(basketball):
    recruits = run_pipeline(20, 3, basketball, locks=build_locks(basketball, position="C"))
    text = format_report(class_report(recruits, basketball, top=False, custom=True), 20, custom=True)
    assert "custom request" in text and "(" not in text.split("\n", 1)[1]


# ---- JSON lore export --------------------------------------------------------------
def test_lore_json_shape(football):
    recruits = run_pipeline(40, 4, football, top=True)
    data = json.loads(json.dumps(lore_json(recruits, football, 4, True, ["40 x no locks"])))
    assert data["class"]["sport"] == "football" and data["class"]["game"]
    assert data["class"]["seed"] == 4 and data["class"]["count"] == 40
    r = data["recruits"][0]
    assert set(r) >= {"name", "bio", "body", "scouting", "ratings", "story_hints"}
    assert r["scouting"]["hidden_truth"]["verdict"] in ("Steal", "Underrated", "As Advertised", "Overrated", "Bust")
    assert len(r["ratings"]["strengths"]) == 3 and len(r["ratings"]["weaknesses"]) == 3


def test_strengths_are_ratings_that_matter_for_his_position(basketball):
    for r in run_pipeline(50, 5, basketball):
        keys = key_ratings(r, basketball)
        record = lore_json([r], basketball, 5, False, [])["recruits"][0]
        top = record["ratings"]["strengths"]
        assert all(s["rating"] in keys for s in top)
        assert top[0]["value"] >= top[-1]["value"]


def test_story_hints_tell_the_truth(basketball):
    recruits = run_pipeline(1000, 6, basketball)
    steal = next(r for r in recruits if r["verdict"] == "Steal")
    assert any("hidden gem" in h and f"#{steal['true_rank']}" in h for h in story_hints(steal, basketball))
    abroad = next(r for r in recruits if r["origin"] in basketball.places.countries)
    assert any("International" in h for h in story_hints(abroad, basketball))


# ---- the whole command, start to finish -------------------------------------------
@pytest.mark.parametrize("sport, extra", [
    ("basketball", {}), ("football", {"top": True}), ("basketball", {"position": "guard", "ethnicity": "white"}),
])
def test_full_run_writes_csv_and_json(tmp_path, sport, extra):
    out = tmp_path / "class.csv"
    cli(sport=sport, count=30, seed=7, out=out, **extra)
    rows = list(csv.DictReader(open(out)))
    data = json.loads(out.with_suffix(".json").read_text())
    assert len(rows) == 30 == len(data["recruits"])
    assert [int(r["rank"]) for r in rows] == [r["rank"] for r in data["recruits"]]
    assert rows[0]["first_name"] == data["recruits"][0]["name"]["first"]


def test_no_json_option(tmp_path):
    out = tmp_path / "class.csv"
    cli(count=5, seed=8, out=out, json_out=False)
    assert out.exists() and not out.with_suffix(".json").exists()


def test_same_seed_same_files(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    cli(sport="football", count=25, seed=9, out=a)
    cli(sport="football", count=25, seed=9, out=b)
    assert a.read_text() == b.read_text()
    ja, jb = json.loads(a.with_suffix(".json").read_text()), json.loads(b.with_suffix(".json").read_text())
    assert ja["recruits"] == jb["recruits"]
