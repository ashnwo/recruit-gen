"""Pipeline stages. Every stage takes the list of recruit dicts and returns it
enriched. Recruits stay plain dicts the whole way through; the last stage
(validate_dedup) checks each one against the Recruit contract."""

import yaml
from pydantic import ValidationError
from scipy.stats import skewnorm

from recruit_gen.models import PhysicalsConfig, Recruit, parse_physicals_config


def class_shape(count: int, rng) -> list[dict]:
    return [{} for _ in range(count)]   # stub: N blank recruits


def origins(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["origin"] = "USA"   # stub (Piece 4)
    return recruits


def positions(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["position"] = "PG"   # stub (Piece 2)
    return recruits


def load_physicals_config(path) -> PhysicalsConfig:
    """Read a YAML file into a checked config. Any problem = clear message + stop."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return parse_physicals_config(data)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        print(f"Config file {path} isn't valid YAML:\n{e}")
    except ValidationError as e:
        print(f"Bad config in {path}:\n{e}")
    raise SystemExit(1)


# Placeholder weight until Piece 2 gives each position its own body config:
# a 6'3" (75 in) player centers at 190 lbs, +6 lbs per extra inch, give or take ~10.
WEIGHT_AT_75_IN = 190
LBS_PER_INCH = 6
WEIGHT_SD = 10


def physicals(recruits: list[dict], rng, config: PhysicalsConfig) -> list[dict]:

    def skewnormal(**args):
        # SciPy ignores numpy's seed unless you hand it the generator.
        return skewnorm.rvs(**args, random_state=rng)

    samplers = {
        "normal": rng.normal,
        "skewnormal": skewnormal,
    }
    draw = samplers[config.distribution]                     # the config picks the function
    args = config.model_dump(exclude={"distribution"})       # loc, scale (+ a for skewnormal)

    for r in recruits:
        height = int(round(draw(**args)))
        center = WEIGHT_AT_75_IN + LBS_PER_INCH * (height - 75)
        r["height"] = height
        r["weight"] = int(round(rng.normal(loc=center, scale=WEIGHT_SD)))
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
