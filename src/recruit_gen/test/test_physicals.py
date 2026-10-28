import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.stats import skewnorm

from recruit_gen.cli import find_config
from recruit_gen.models import NormalParams, SkewNormalParams, SportConfig, parse_physicals_config
from recruit_gen.stages import load_sport_config, physicals, positions

NORMAL = {"loc": 79.0, "scale": 2.5, "distribution": "normal"}
SKEW = {"loc": 76.0, "scale": 3.0, "distribution": "skewnormal", "a": 6.0}
WEIGHT = {"ref_height": 79.0, "at_ref": 205, "lbs_per_inch": 6, "sd": 12}
TALENT = {"population": 1000, "pool_size": 100, "scout_error": 0.3,
          "stars": [{"stars": 4, "last_rank": 10, "rating": [97, 90]},
                    {"stars": 3, "last_rank": 100, "rating": [89, 80]}],
          "verdicts": {"steal": 4, "underrated": 1.75, "overrated": 1.75, "bust": 4, "min_gap": 5}}
ETH = {"Black": 0.5, "White": 0.5}
PLACES = {"states": {"ZZ": {"name": "Testland", "cities": {"Testville": 1}}},
          "schools": {"patterns": [{"text": "{city} High School", "weight": 1}], "international": "{city} Academy",
                      "directions": ["North"], "namesakes": ["Lincoln"], "saints": ["St. John's"]}}
ORIGINS = {"international_rate": 0, "states": {"ZZ": 1}}
PREP = {"rate": 0.1, "international_rate": 0.5, "schools": {"Test Prep": 1}}


def tiny_sport(**overrides) -> dict:
    """A small valid sport config to break on purpose in tests."""
    cfg = {"sport": "test", "positions": {
        "A": {"share": 0.5, "height": NORMAL, "weight": WEIGHT, "ethnicity": ETH},
        "B": {"share": 0.5, "height": SKEW, "weight": WEIGHT, "ethnicity": ETH},
    }, "talent": TALENT, "origins": ORIGINS, "prep": PREP, "places": PLACES}
    cfg.update(overrides)
    return cfg


def expected_mean(height_cfg) -> float:
    if height_cfg.distribution == "skewnormal":
        return skewnorm.mean(height_cfg.a, height_cfg.loc, height_cfg.scale)
    return height_cfg.loc


def bodies(config, n, seed, position=None):
    rng = np.random.default_rng(seed)
    recruits = [{"position": position} for _ in range(n)] if position else positions([{} for _ in range(n)], rng, config)
    return physicals(recruits, rng, config)


# ---- height config contract (Days 8-14) --------------------------------------
def test_good_height_config_passes():
    cfg = parse_physicals_config(NORMAL)
    assert isinstance(cfg, NormalParams) and cfg.scale == 2.5


def test_skewnormal_config_picks_the_right_class():
    assert isinstance(parse_physicals_config(SKEW), SkewNormalParams)


@pytest.mark.parametrize("bad", [
    {**NORMAL, "scale": -2},                 # value wrong
    {**NORMAL, "scale": "tall"},             # type wrong
    {**NORMAL, "distribution": "banana"},    # name wrong
    {"loc": 76.0, "scale": 3.0, "distribution": "skewnormal"},   # 'a' missing
])
def test_bad_height_config_rejected(bad):
    with pytest.raises(ValidationError):
        parse_physicals_config(bad)


# ---- sport config contract (Piece 2) -----------------------------------------
def test_tiny_sport_is_valid():
    assert set(SportConfig.model_validate(tiny_sport()).positions) == {"A", "B"}


def test_shares_must_add_up_to_one():
    cfg = tiny_sport()
    cfg["positions"]["A"]["share"] = 0.7          # 0.7 + 0.5 = 1.2
    with pytest.raises(ValidationError, match="add up to 1"):
        SportConfig.model_validate(cfg)


def test_sport_needs_positions():
    with pytest.raises(ValidationError):
        SportConfig.model_validate(tiny_sport(positions={}))


def test_position_needs_weight_settings():
    cfg = tiny_sport()
    del cfg["positions"]["A"]["weight"]
    with pytest.raises(ValidationError):
        SportConfig.model_validate(cfg)


def test_bad_distribution_inside_a_position_rejected():
    cfg = tiny_sport()
    cfg["positions"]["B"]["height"] = {**NORMAL, "distribution": "banana"}
    with pytest.raises(ValidationError):
        SportConfig.model_validate(cfg)


# ---- loading YAML files --------------------------------------------------------
def test_real_basketball_config_is_valid(basketball):
    assert basketball.sport == "basketball"
    assert set(basketball.positions) == {"PG", "SG", "SF", "PF", "C"}


def test_missing_config_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        load_sport_config(tmp_path / "nope.yaml")


def test_broken_yaml_exits_cleanly(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("sport: [basketball\n")
    with pytest.raises(SystemExit):
        load_sport_config(f)


def write_sport(folder, cfg) -> "Path":
    """Write a sport config + its shared files the way the loader expects.
    The name tables are big, so link to the real ones instead of copying."""
    (folder / "shared").mkdir(exist_ok=True)
    (folder / "shared" / "places.yaml").write_text(yaml.safe_dump(cfg.pop("places")))
    (folder / "shared" / "names").symlink_to(find_config("basketball").parent / "shared" / "names")
    f = folder / "test.yaml"
    f.write_text(yaml.safe_dump(cfg))
    return f


def test_tiny_sport_loads_from_files(tmp_path):
    assert load_sport_config(write_sport(tmp_path, tiny_sport())).sport == "test"


def test_bad_values_exit_cleanly(tmp_path):
    cfg = tiny_sport()
    cfg["positions"]["A"]["height"] = {**NORMAL, "scale": -2}
    with pytest.raises(SystemExit):
        load_sport_config(write_sport(tmp_path, cfg))


def test_missing_places_file_exits_cleanly(tmp_path):
    f = write_sport(tmp_path, tiny_sport())
    (tmp_path / "shared" / "places.yaml").unlink()
    with pytest.raises(SystemExit):
        load_sport_config(f)


# ---- positions stage -----------------------------------------------------------
def test_position_mix_matches_config(basketball):
    out = positions([{} for _ in range(20_000)], np.random.default_rng(5), basketball)
    for pos, cfg in basketball.positions.items():
        share = sum(r["position"] == pos for r in out) / len(out)
        assert abs(share - cfg.share) < 0.015, f"{pos}: {share:.3f} vs {cfg.share}"


# ---- physicals stage -----------------------------------------------------------
def test_every_position_centers_where_config_says(basketball):
    for pos, cfg in basketball.positions.items():
        heights = [r["height"] for r in bodies(basketball, 4_000, seed=11, position=pos)]
        assert abs(np.mean(heights) - expected_mean(cfg.height)) < 0.3, pos


def test_positions_go_small_to_big(basketball):
    avg = {pos: np.mean([r["height"] for r in bodies(basketball, 3_000, seed=12, position=pos)])
           for pos in ["PG", "SG", "SF", "PF", "C"]}
    assert avg["PG"] < avg["SG"] < avg["SF"] < avg["PF"] < avg["C"]


def test_height_spread_is_realistic(basketball):
    heights = [r["height"] for r in bodies(basketball, 5_000, seed=13, position="SF")]
    assert 1.0 < np.std(heights) < 2.5      # brick fails low, smear fails high


def test_centers_lean_tall(basketball):
    heights = np.array([r["height"] for r in bodies(basketball, 10_000, seed=14, position="C")])
    above = np.mean(heights) - np.median(heights)
    assert above >= 0                                    # long tail on the tall side
    assert np.percentile(heights, 99) - np.median(heights) > np.median(heights) - np.percentile(heights, 1)


def test_taller_players_weigh_more_within_a_position(basketball):
    out = bodies(basketball, 5_000, seed=15, position="PF")
    assert np.corrcoef([r["height"] for r in out], [r["weight"] for r in out])[0, 1] > 0.4


def test_bigs_outweigh_guards(basketball):
    pg = np.mean([r["weight"] for r in bodies(basketball, 3_000, seed=16, position="PG")])
    c = np.mean([r["weight"] for r in bodies(basketball, 3_000, seed=16, position="C")])
    assert c - pg > 35


def test_height_written_in_feet_and_inches(basketball):
    for r in bodies(basketball, 200, seed=17):
        feet, inches = r["height_ft"].rstrip('"').split("'")
        assert int(feet) * 12 + int(inches) == r["height"]


def test_skewnormal_follows_the_seed():
    cfg = SportConfig.model_validate(tiny_sport())
    a = [r["height"] for r in bodies(cfg, 50, seed=1, position="B")]
    b = [r["height"] for r in bodies(cfg, 50, seed=1, position="B")]
    c = [r["height"] for r in bodies(cfg, 50, seed=2, position="B")]
    assert a == b and a != c
