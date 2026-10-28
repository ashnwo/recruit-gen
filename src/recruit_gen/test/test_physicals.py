import numpy as np
import pytest
from pydantic import ValidationError

from recruit_gen.models import NormalParams, SkewNormalParams, parse_physicals_config
from recruit_gen.stages import load_physicals_config, physicals

NORMAL = {"loc": 79.0, "scale": 2.5, "distribution": "normal"}
SKEW = {"loc": 76.0, "scale": 3.0, "distribution": "skewnormal", "a": 6.0}


# ---- config contract ---------------------------------------------------------
def test_good_config_passes():
    cfg = parse_physicals_config(NORMAL)
    assert isinstance(cfg, NormalParams)
    assert cfg.scale == 2.5


def test_skewnormal_config_picks_the_right_class():
    assert isinstance(parse_physicals_config(SKEW), SkewNormalParams)


def test_negative_scale_rejected():          # value wrong
    with pytest.raises(ValidationError):
        parse_physicals_config({**NORMAL, "scale": -2})


def test_string_scale_rejected():            # type wrong
    with pytest.raises(ValidationError):
        parse_physicals_config({**NORMAL, "scale": "tall"})


def test_unknown_distribution_rejected():    # name wrong
    with pytest.raises(ValidationError):
        parse_physicals_config({**NORMAL, "distribution": "banana"})


def test_skewnormal_without_a_rejected():    # field missing
    with pytest.raises(ValidationError):
        parse_physicals_config({"loc": 76.0, "scale": 3.0, "distribution": "skewnormal"})


# ---- loading the YAML file ---------------------------------------------------
def test_config_file_loads(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("loc: 79\nscale: 2.5\ndistribution: normal\n")
    assert load_physicals_config(f).loc == 79


def test_missing_config_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        load_physicals_config(tmp_path / "nope.yaml")


def test_bad_config_value_exits_cleanly(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("loc: 79\nscale: -2\ndistribution: normal\n")
    with pytest.raises(SystemExit):
        load_physicals_config(f)


def test_broken_yaml_exits_cleanly(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("loc: [79\nscale: 2.5\n")
    with pytest.raises(SystemExit):
        load_physicals_config(f)


# ---- the physicals stage -----------------------------------------------------
def test_physicals_gives_every_recruit_a_body():
    out = physicals([{} for _ in range(200)], np.random.default_rng(22), parse_physicals_config(NORMAL))
    for r in out:
        assert isinstance(r["height"], int) and r["height"] > 0
        assert isinstance(r["weight"], int) and r["weight"] > 0


def test_height_distribution_shape():
    out = physicals([{} for _ in range(10_000)], np.random.default_rng(42), parse_physicals_config(NORMAL))
    heights = [r["height"] for r in out]
    assert 1.5 < np.std(heights) < 3.5        # brick fails low, smear fails high
    assert 78.5 < np.mean(heights) < 79.5     # centered where the config says


def test_taller_players_weigh_more():
    out = physicals([{} for _ in range(5_000)], np.random.default_rng(7), parse_physicals_config(NORMAL))
    heights = [r["height"] for r in out]
    weights = [r["weight"] for r in out]
    assert np.corrcoef(heights, weights)[0, 1] > 0.5


def test_skewnormal_follows_the_seed():
    cfg = parse_physicals_config(SKEW)
    a = [r["height"] for r in physicals([{} for _ in range(50)], np.random.default_rng(1), cfg)]
    b = [r["height"] for r in physicals([{} for _ in range(50)], np.random.default_rng(1), cfg)]
    c = [r["height"] for r in physicals([{} for _ in range(50)], np.random.default_rng(2), cfg)]
    assert a == b      # same seed, same heights
    assert a != c      # different seed, different heights


def test_skewnormal_leans_tall():
    out = physicals([{} for _ in range(10_000)], np.random.default_rng(3), parse_physicals_config(SKEW))
    heights = [r["height"] for r in out]
    assert np.mean(heights) > SKEW["loc"] + 1   # a > 0 pushes the pile to the tall side
