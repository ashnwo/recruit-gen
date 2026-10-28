"""Piece 9: football - added as a settings file, with no football-only code."""
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import recruit_gen
from recruit_gen.cli import write_csv
from recruit_gen.locks import build_locks
from recruit_gen.pipeline import build_class, run_pipeline

GOLDEN = Path(__file__).parent / "football_seed42.json"


@pytest.fixture(scope="module")
def big_fb(football):
    """Two full football classes (7,000 recruits) shared by the tests below."""
    return [r for seed in range(2) for r in run_pipeline(3500, seed=seed, config=football)]


def by_pos(recruits, pos):
    return [r for r in recruits if r["position"] == pos]


def avg(recruits, key, attr=None):
    return float(np.mean([r["ratings"][attr] if attr else r[key] for r in recruits]))


# ---- the design rule ------------------------------------------------------------
def test_engine_never_checks_which_sport_it_is():
    """One engine, sport as data: no engine file compares against a sport's name."""
    engine = Path(recruit_gen.__file__).parent
    for path in engine.glob("*.py"):
        if path.name == "cli.py":                 # the CLI may have a default sport
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in ("football", "basketball"):
                pytest.fail(f"{path.name} line {node.lineno} mentions '{node.value}'")


# ---- settings ---------------------------------------------------------------------
def test_football_config_loads(football):
    assert football.sport == "football"
    assert len(football.positions) == 15
    assert abs(sum(p.share for p in football.positions.values()) - 1) < 0.01
    assert football.attributes.order[:3] == ["SPD", "ACC", "AGI"]


def test_football_golden_seed42(football):
    """Same seed + same config = the exact same class. After an intended change:
        uv run python src/recruit_gen/test/update_golden.py"""
    assert run_pipeline(120, seed=42, config=football) == json.loads(GOLDEN.read_text())


def test_full_class_builds_cleanly(football):
    result = build_class(3500, seed=3, config=football)
    assert len(result.recruits) == 3500
    assert len({x["rank"] for x in result.rejects}) / 3500 < 0.02


def test_csv_uses_game_abbreviations(football, tmp_path):
    out = tmp_path / "fb.csv"
    write_csv(run_pipeline(10, seed=4, config=football), out)
    header = next(csv.reader(open(out)))
    assert header[-len(football.attributes.order):] == football.attributes.order
    assert "forty" in header and "dev_trait" in header


# ---- bodies (fixing the game's undersized linemen) -------------------------------
def test_linemen_are_real_linemen(big_fb):
    for pos in ("OT", "OG"):
        g = by_pos(big_fb, pos)
        assert avg(g, "weight") > 290 and avg(g, "height") > 75.5
    assert avg(by_pos(big_fb, "OT"), "height") > avg(by_pos(big_fb, "OG"), "height")


def test_forty_times_are_realistic(football, big_fb):
    cb, ot = avg(by_pos(big_fb, "CB"), "forty"), avg(by_pos(big_fb, "OT"), "forty")
    assert 4.35 < cb < 4.65 and 4.95 < ot < 5.4
    f = football.forty
    assert all(f.min <= r["forty"] <= f.max for r in big_fb)
    assert np.corrcoef([r["ratings"]["SPD"] for r in big_fb], [r["forty"] for r in big_fb])[0, 1] < -0.9


# ---- positions play like their position ---------------------------------------
@pytest.mark.parametrize("better, worse, attr", [
    ("QB", "OT", "THP"), ("OT", "WR", "PBK"), ("CB", "OG", "MCV"), ("EDGE", "CB", "PMV"),
    ("RB", "OT", "CAR"), ("K", "QB", "KPW"), ("WR", "LB", "CTH"),
])
def test_position_ratings(big_fb, better, worse, attr):
    assert avg(by_pos(big_fb, better), None, attr) > avg(by_pos(big_fb, worse), None, attr) + 15


def test_skills_a_position_never_uses_sit_at_the_bottom(football, big_fb):
    floor = football.ratings.levels["none"][0]
    assert avg(by_pos(big_fb, "OT"), None, "KPW") < floor + 6     # tackles can't kick


# ---- who gets ranked where ---------------------------------------------------------
def test_kickers_and_punters_are_never_elite(big_fb):
    specialists = [r for r in big_fb if r["position"] in ("K", "P")]
    assert specialists and min(r["rank"] for r in specialists) > 100
    assert all(r["stars"] < 5 for r in specialists)
    assert sum(r["stars"] == 4 for r in specialists) <= 2              # a 4-star kicker is a rare event


def test_some_punters_come_from_australia(football):
    punters = [r for seed in range(6) for r in run_pipeline(3500, seed=seed, config=football) if r["position"] == "P"]
    share = sum(r["origin"] == "AUS" for r in punters) / len(punters)
    assert 0.15 < share < 0.45                     # setting is 30%


def test_hotbed_states(big_fb):
    top = [s for s, _ in Counter(r["origin"] for r in big_fb).most_common(4)]
    assert set(top) == {"TX", "FL", "GA", "CA"}


def test_ethnicity_by_position(big_fb):
    def share(pos, eth):
        g = by_pos(big_fb, pos)
        return sum(r["ethnicity"] == eth for r in g) / len(g)
    assert share("QB", "White") > share("CB", "White") + 0.25
    assert share("DT", "Polynesian") > share("CB", "Polynesian") + 0.05
    assert share("K", "White") > 0.6


# ---- dev traits ---------------------------------------------------------------------
def test_dev_traits_follow_talent(big_fb):
    def elite_share(stars):
        g = [r for r in big_fb if r["stars"] == stars]
        return sum(r["dev_trait"] in ("Elite", "Star") for r in g) / len(g)
    assert elite_share(5) > elite_share(4) > elite_share(3) > elite_share(2)
    assert {r["dev_trait"] for r in big_fb} == {"Normal", "Impact", "Star", "Elite"}


def test_basketball_has_no_football_extras(basketball):
    r = run_pipeline(5, seed=5, config=basketball)[0]
    assert r["dev_trait"] == "" and r["forty"] is None and r["wingspan"] is not None


# ---- custom requests work for football too ------------------------------------------
def test_football_locks(football):
    ol = run_pipeline(200, 6, football, locks=build_locks(football, position="OL", ethnicity="polynesian"))
    assert all(r["position"] in ("OT", "OG", "C") and r["ethnicity"] == "Polynesian" for r in ol)
    homes = Counter(r["origin"] for r in ol)
    west = sum(homes[c] for c in ("HI", "UT", "CA", "AS", "AZ", "WA", "TX", "NV", "OR", "AK"))
    assert west / len(ol) > 0.75                     # Polynesian families are concentrated out West
    assert homes["FL"] / len(ol) < 0.1
    qbs = run_pipeline(5, 7, football, top=True, locks=build_locks(football, position="QB", origin="TX", stars="5"))
    assert all(r["position"] == "QB" and r["origin"] == "TX" and r["stars"] == 5 for r in qbs)


# ---- same seed = same class, on every run --------------------------------------------
def test_same_seed_same_class_across_separate_runs():
    """Python shuffles set order differently each time it starts (its 'hash seed').
    A class must not depend on that, or --seed wouldn't reproduce a class tomorrow."""
    import os
    import subprocess
    import sys
    code = ("import json, hashlib\n"
            "from recruit_gen.cli import find_config\n"
            "from recruit_gen.stages import load_sport_config\n"
            "from recruit_gen.pipeline import run_pipeline\n"
            "out = [run_pipeline(150, 42, load_sport_config(find_config(s))) for s in ('basketball', 'football')]\n"
            "print(hashlib.md5(json.dumps(out).encode()).hexdigest())\n")
    results = set()
    for hash_seed in ("1", "11", "555"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": os.pathsep.join(sys.path)}
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[3], check=True)
        results.add(done.stdout.strip())
    assert len(results) == 1
