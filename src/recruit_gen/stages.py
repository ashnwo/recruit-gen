"""Pipeline stages. Every stage takes the list of recruit dicts and returns it
enriched. Recruits stay plain dicts the whole way through; the last stage
(validate_dedup) checks each one against the Recruit contract."""

from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError
from scipy.stats import skewnorm

from recruit_gen.models import PhysicalsConfig, Recruit, SportConfig


def load_sport_config(path: Path) -> SportConfig:
    """Read a sport's YAML file into a checked config. Any problem = clear message + stop."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return SportConfig.model_validate(data)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        print(f"Config file {path} isn't valid YAML:\n{e}")
    except ValidationError as e:
        print(f"Bad config in {path}:\n{e}")
    raise SystemExit(1)


def class_shape(count: int, rng) -> list[dict]:
    return [{} for _ in range(count)]   # stub: N blank recruits


def origins(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["origin"] = "USA"   # stub (Piece 4)
    return recruits


def positions(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Weighted draw: each position comes up about as often as its share says."""
    names = list(config.positions)
    shares = [config.positions[p].share for p in names]
    total = sum(shares)
    probs = [s / total for s in shares]   # tiny rounding fix so they add to exactly 1
    for r, pos in zip(recruits, rng.choice(names, size=len(recruits), p=probs)):
        r["position"] = str(pos)
    return recruits


def make_height_sampler(height_cfg: PhysicalsConfig, rng):
    """Turn one height config into a no-argument function that returns a height.
    The config's "distribution" field picks the function (no if-branches)."""

    def skewnormal(**args):
        # SciPy ignores numpy's seed unless you hand it the generator.
        return skewnorm.rvs(**args, random_state=rng)

    samplers = {
        "normal": rng.normal,
        "skewnormal": skewnormal,
    }
    draw = samplers[height_cfg.distribution]
    args = height_cfg.model_dump(exclude={"distribution"})   # loc, scale (+ a)
    return lambda: draw(**args)


MAX_WEIGHT_SWING = 2.5   # in sd units


def feet_inches(height: int) -> str:
    return f"{height // 12}'{height % 12}\""


def physicals(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Height from the player's position, then weight from that height."""
    samplers = {pos: make_height_sampler(p.height, rng) for pos, p in config.positions.items()}
    for r in recruits:
        pos_cfg = config.positions[r["position"]]
        height = int(round(samplers[r["position"]]()))
        w = pos_cfg.weight
        center = w.at_ref + w.lbs_per_inch * (height - w.ref_height)
        r["height"] = height
        r["height_ft"] = feet_inches(height)
        # Cap the random swing at 2.5 sd each way (covers ~99% of real players) so a
        # freak draw can't make a 5'7" guard weigh 105 lbs.
        swing = float(np.clip(rng.normal(), -MAX_WEIGHT_SWING, MAX_WEIGHT_SWING))
        r["weight"] = int(round(center + swing * w.sd))
    return recruits


def talent(recruits: list[dict], rng, exponent: int = 3) -> list[dict]:
    for r in recruits:
        r["talent"] = rng.random() ** exponent   # cubing a 0-1 number = few high, many low
    return recruits


def development(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["development"] = "star"   # stub (Piece 6)
    return recruits


def names(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["name"] = "bobby smith"   # stub (Piece 5)
    return recruits


def validate_dedup(recruits: list[dict], rng) -> list[dict]:
    """Last stage: every recruit must pass the Recruit contract.
    Duplicate checks and what to do with rejects come in Piece 7."""
    checked = []
    for i, r in enumerate(recruits, start=1):
        try:
            checked.append(Recruit.model_validate(r).model_dump())
        except ValidationError as e:
            raise ValueError(f"Recruit #{i} broke the contract:\n{e}") from e
    return checked
