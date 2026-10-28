"""Pipeline stages. Every stage takes the list of recruit dicts and returns it
enriched. Recruits stay plain dicts the whole way through; the last stage
(validate_dedup) checks each one against the Recruit contract."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError
from scipy.stats import norm, skewnorm

from recruit_gen.models import PhysicalsConfig, Recruit, SportConfig, StarBand, TalentConfig


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


def class_shape(count: int, rng, config: SportConfig, top: bool = False) -> list[dict]:
    """Decide WHO is in the class by national rank.
    top=True  -> the best `count` players in the country (ranks 1..count)
    top=False -> a realistic mix: `count` different ranks picked at random from
                 everyone ranked (mostly 2-3 stars, the odd 5-star)

    Each rank becomes a scouted talent score (scout_z) by asking: "if a player is
    ranked #r out of everyone, how far above average must he be?" That's the far
    right tail of a bell curve, which is where the talent pyramid comes from:
    the gap between #1 and #10 is huge, the gap between #500 and #510 is tiny."""
    t = config.talent
    if count > t.pool_size:
        raise ValueError(f"Can't make {count} recruits: only {t.pool_size} players are ranked "
                         f"(pool_size in the config).")
    if top:
        ranks = np.arange(1, count + 1)
    else:
        ranks = np.sort(rng.choice(np.arange(1, t.pool_size + 1), size=count, replace=False))
    return [{"rank": int(r), "scout_z": float(norm.isf((r - 0.5) / t.population))} for r in ranks]


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


def star_band(rank: int, bands: list[StarBand]) -> tuple[StarBand, int]:
    """The star band a national rank falls in, plus the first rank of that band."""
    first = 1
    for band in bands:
        if rank <= band.last_rank:
            return band, first
        first = band.last_rank + 1
    return bands[-1], first


def rating_for(rank: int, bands: list[StarBand]) -> float:
    """247-style rating: best rank in a band gets the band's top rating, the band's
    last rank gets its bottom rating, everyone between is spread evenly."""
    band, first = star_band(rank, bands)
    best, worst = band.rating
    if band.last_rank == first:
        return best
    frac = (min(rank, band.last_rank) - first) / (band.last_rank - first)
    return round(best - frac * (best - worst), 1)


def verdict_for(true_rank: int, rank: int, t: TalentConfig) -> str:
    """Compare true rank to scouted rank. ratio below 1 = really better than ranked."""
    v = t.verdicts
    if abs(true_rank - rank) < v.min_gap:
        return "As Advertised"
    ratio = true_rank / rank
    if ratio <= 1 / v.steal:
        return "Steal"
    if ratio <= 1 / v.underrated:
        return "Underrated"
    if ratio >= v.bust:
        return "Bust"
    if ratio >= v.overrated:
        return "Overrated"
    return "As Advertised"


def talent(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Stars and rating from the scouted rank; then the hidden truth.
    True talent = what scouts saw + scouting error. Where that true talent would
    rank him nationally decides if he's a steal, a bust, or as advertised."""
    t = config.talent
    for r in recruits:
        band, _ = star_band(r["rank"], t.stars)
        r["stars"] = band.stars
        r["rating"] = rating_for(r["rank"], t.stars)
        true_z = r["scout_z"] + rng.normal(0, t.scout_error)
        r["true_rank"] = max(1, math.ceil(t.population * norm.sf(true_z)))
        r["verdict"] = verdict_for(r["true_rank"], r["rank"], t)

    # Position rank: order within each position by national rank.
    by_pos = defaultdict(list)
    for r in recruits:
        by_pos[r["position"]].append(r)
    for group in by_pos.values():
        for i, r in enumerate(sorted(group, key=lambda x: x["rank"]), start=1):
            r["pos_rank"] = i
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
    model_dump() keeps only the contract's fields, so working values like
    scout_z drop out here. Duplicate checks and rejects come in Piece 7."""
    checked = []
    for i, r in enumerate(recruits, start=1):
        try:
            checked.append(Recruit.model_validate(r).model_dump())
        except ValidationError as e:
            raise ValueError(f"Recruit #{i} broke the contract:\n{e}") from e
    return checked
