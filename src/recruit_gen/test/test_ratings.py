"""Piece 6: archetypes, development, 2K ratings, OVR and potential."""
import csv
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from recruit_gen.cli import write_csv
from recruit_gen.models import SportConfig
from recruit_gen.pipeline import run_pipeline


@pytest.fixture(scope="module")
def big_class(basketball):
    """4,000 recruits (a realistic mix) shared by the tests below."""
    return [r for seed in range(4) for r in run_pipeline(1000, seed=seed, config=basketball)]


def avg(recruits, key, attr=None):
    vals = [r["ratings"][attr] if attr else r[key] for r in recruits]
    return float(np.mean(vals))


# ---- config rules ----------------------------------------------------------------
def test_ratings_config_loaded(basketball):
    assert len(basketball.attributes.order) == 36
    assert set(basketball.development.trajectories) == {"Early Bloomer", "Steady Climber", "Late Bloomer", "Raw Project"}


@pytest.mark.parametrize("breakage, message", [
    (lambda d: d["positions"]["PG"]["overrides"].update({"Juggling": "core"}), "override 'Juggling'"),
    (lambda d: d["positions"]["PG"]["archetypes"].update({"Wizard": 1.0}), "archetype 'Wizard'"),
    (lambda d: d["positions"]["C"]["roles"].update({"dancing": "major"}), "role group 'dancing'"),
    (lambda d: d["archetypes"]["Sharpshooter"]["mods"].update({"Moonwalk": 5}), "mod 'Moonwalk'"),
    (lambda d: d["positions"]["SG"]["secondary"].update({"QB": 0.1}), "secondary 'QB'"),
])
def test_typos_in_rating_names_are_caught(basketball, breakage, message):
    d = basketball.model_dump()
    breakage(d)
    with pytest.raises(ValidationError, match=message):
        SportConfig.model_validate(d)


def test_attribute_in_two_groups_is_caught(basketball):
    d = basketball.model_dump()
    d["attributes"]["groups"]["shooting"].append("Block")
    with pytest.raises(ValidationError, match="in two places"):
        SportConfig.model_validate(d)


# ---- every recruit -----------------------------------------------------------------
def test_every_recruit_has_all_ratings_in_order(basketball, big_class):
    order = basketball.attributes.order
    lo, hi = basketball.ratings.min, basketball.ratings.max
    for r in big_class[:500]:
        assert list(r["ratings"]) == order
        assert all(lo <= v <= hi for v in r["ratings"].values())
        assert lo <= r["ovr"] <= r["potential"] <= hi


def test_csv_has_one_column_per_rating(basketball, tmp_path):
    out = tmp_path / "class.csv"
    write_csv(run_pipeline(20, seed=1, config=basketball), out)
    with open(out) as f:
        header = next(csv.reader(f))
    assert header[-36:] == basketball.attributes.order
    assert "ovr" in header and "archetype" in header and "wingspan_ft" in header


# ---- talent shows up in the ratings -------------------------------------------
def test_better_recruits_have_higher_ovr_and_potential(big_class):
    by_stars = {s: [r for r in big_class if r["stars"] == s] for s in (5, 4, 3, 2)}
    ovr = [avg(by_stars[s], "ovr") for s in (5, 4, 3, 2)]
    pot = [avg(by_stars[s], "potential") for s in (5, 4, 3, 2)]
    assert ovr == sorted(ovr, reverse=True) and pot == sorted(pot, reverse=True)
    assert 60 < ovr[0] < 80 and 45 < ovr[3] < 65         # 2K draft-class range


def test_steals_have_higher_ceilings_than_busts_at_the_same_level(big_class):
    mid = [r for r in big_class if r["stars"] in (2, 3)]
    steals = [r for r in mid if r["verdict"] == "Steal"]
    busts = [r for r in mid if r["verdict"] == "Bust"]
    assert avg(steals, "potential") > avg(busts, "potential") + 8


# ---- positions and archetypes shape the ratings ---------------------------------
def test_positions_play_like_their_position(big_class):
    pg = [r for r in big_class if r["position"] == "PG"]
    c = [r for r in big_class if r["position"] == "C"]
    assert avg(pg, None, "Pass Accuracy") > avg(c, None, "Pass Accuracy") + 10
    assert avg(pg, None, "Speed") > avg(c, None, "Speed") + 10
    assert avg(c, None, "Block") > avg(pg, None, "Block") + 10
    assert avg(c, None, "Strength") > avg(pg, None, "Strength") + 10


@pytest.mark.parametrize("pos, better, worse, attr", [
    ("SG", "Sharpshooter", "Slasher", "Three-Point Shot"),
    ("SG", "Slasher", "Sharpshooter", "Driving Dunk"),
    ("C", "Rim Protector", "Stretch Big", "Block"),
    ("C", "Stretch Big", "Rim Protector", "Three-Point Shot"),
    ("PG", "Floor General", "3-and-D Guard", "Pass Vision"),
])
def test_archetypes_change_the_right_ratings(big_class, pos, better, worse, attr):
    a = [r for r in big_class if r["position"] == pos and r["archetype"] == better]
    b = [r for r in big_class if r["position"] == pos and r["archetype"] == worse]
    assert avg(a, None, attr) > avg(b, None, attr) + 5


def test_archetype_mix_matches_config(basketball, big_class):
    centers = [r for r in big_class if r["position"] == "C"]
    counts = Counter(r["archetype"] for r in centers)
    for name, share in basketball.positions["C"].archetypes.items():
        assert abs(counts[name] / len(centers) - share) < 0.06, name


def test_rim_protectors_have_longer_arms(big_class):
    def reach(arch):
        group = [r for r in big_class if r["position"] == "C" and r["archetype"] == arch]
        return np.mean([r["wingspan"] - r["height"] for r in group])
    assert reach("Rim Protector") > reach("Stretch Big") + 1


def test_wingspan_matches_position_settings(basketball, big_class):
    for pos, cfg in basketball.positions.items():
        group = [r for r in big_class if r["position"] == pos]
        extra = np.mean([r["wingspan"] - r["height"] for r in group])
        assert abs(extra - cfg.wingspan[0]) < 1.0, pos


# ---- bodies shape the body ratings ---------------------------------------------
def test_heavier_players_are_stronger(big_class):
    pf = [r for r in big_class if r["position"] == "PF"]
    assert np.corrcoef([r["weight"] for r in pf], [r["ratings"]["Strength"] for r in pf])[0, 1] > 0.3


def test_heavier_players_are_slower(big_class):
    sf = [r for r in big_class if r["position"] == "SF"]
    assert np.corrcoef([r["weight"] for r in sf], [r["ratings"]["Speed"] for r in sf])[0, 1] < 0


# ---- development -----------------------------------------------------------------
def test_trajectory_mix_matches_config(basketball, big_class):
    counts = Counter(r["development"] for r in big_class)
    for name, t in basketball.development.trajectories.items():
        assert abs(counts[name] / len(big_class) - t.weight) < 0.03, name


def test_late_bloomers_have_more_room_to_grow(big_class):
    def room(name):
        return np.mean([r["potential"] - r["ovr"] for r in big_class if r["development"] == name])
    assert room("Late Bloomer") > room("Steady Climber") > room("Early Bloomer")


def test_scouts_overrate_early_bloomers_and_miss_late_bloomers(big_class):
    def share(name, labels):
        group = [r for r in big_class if r["development"] == name]
        return sum(r["verdict"] in labels for r in group) / len(group)
    assert share("Early Bloomer", {"Overrated", "Bust"}) > share("Late Bloomer", {"Overrated", "Bust"}) + 0.15
    assert share("Late Bloomer", {"Underrated", "Steal"}) > share("Early Bloomer", {"Underrated", "Steal"}) + 0.15


def test_international_players_are_harder_to_scout(big_class):
    def surprise(group):
        return sum(r["verdict"] != "As Advertised" for r in group) / len(group)
    intl = [r for r in big_class if len(r["origin"]) == 3]
    us = [r for r in big_class if len(r["origin"]) == 2]
    assert surprise(intl) > surprise(us)
