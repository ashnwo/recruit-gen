"""Pipeline stages. Every stage takes the list of recruit dicts and returns it
enriched. Recruits stay plain dicts the whole way through; the clean-up check
in checks.py (validate_dedup) checks each one at the end."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError
from scipy.stats import norm, skewnorm

from recruit_gen.models import (
    ETHNICITIES, PhysicalsConfig, SportConfig, StarBand, TalentConfig,
)
from recruit_gen.locks import Locks, ethnicity_mix_for, joint_table, match_share, star_ranks
from recruit_gen.names import name_book, name_class
from recruit_gen.ratings import frame, rate
from recruit_gen.sampling import WeightedPicker


def read_yaml(path: Path):
    """Read a YAML file. Missing or broken file = clear message + stop."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        print(f"Config file {path} isn't valid YAML:\n{e}")
    raise SystemExit(1)


def load_sport_config(path: Path) -> SportConfig:
    """Read a sport's YAML file plus the shared files next to it (shared/places.yaml,
    shared/names/names.yaml) into one checked config. Any problem = clear message + stop."""
    path = Path(path)
    data = read_yaml(path) or {}
    shared = path.parent / "shared"
    names_data = read_yaml(shared / "names" / "names.yaml") or {}
    data = {**data, "places": read_yaml(shared / "places.yaml"),
            "names": {**names_data, "folder": shared / "names"}}
    for table in ("first_names.csv", "surnames.csv"):
        if not (shared / "names" / table).exists():
            print(f"Name table not found: {shared / 'names' / table}")
            raise SystemExit(1)
    try:
        return SportConfig.model_validate(data)
    except ValidationError as e:
        print(f"Bad config in {path} (or the shared files in {shared}):\n{e}")
    raise SystemExit(1)


NO_LOCKS = Locks()


def locks_of(r: dict) -> Locks:
    return r.get("locks") or NO_LOCKS


def class_shape(count: int, rng, config: SportConfig, top: bool = False,
                locks: Locks | None = None, taken: set | None = None) -> list[dict]:
    """Decide WHO is in the class by national rank.
    top=True  -> the best `count` players in the country (ranks 1..count)
    top=False -> a realistic mix: `count` different ranks picked at random from
                 everyone ranked (mostly 2-3 stars, the odd 5-star)
    locks     -> a star lock limits the ranks; in top mode, other locks (say, centers
                 only) spread the ranks out the way the best centers really rank
    taken     -> ranks already used by other requests in the same class

    Each rank becomes a scouted talent score (scout_z) by asking: "if a player is
    ranked #r out of everyone, how far above average must he be?" That's the far
    right tail of a bell curve, which is where the talent pyramid comes from:
    the gap between #1 and #10 is huge, the gap between #500 and #510 is tiny."""
    t = config.talent
    locks = locks or NO_LOCKS
    taken = taken or set()
    allowed = [r for r in star_ranks(locks.stars, config) if r not in taken]
    if count > len(allowed):
        raise ValueError(f"Can't make {count} recruits: only {len(allowed)} matching ranked spots are left "
                         f"(pool_size in the config is {t.pool_size}).")
    if not top:
        ranks = sorted(int(x) for x in rng.choice(allowed, size=count, replace=False))
    elif not locks.shapes_identity:
        ranks = allowed[:count]
    else:
        # Walk down the rankings; each rank is "one of ours" with the chance that a
        # random ranked player matches the locks. The first `count` hits are the best
        # matching players in the country (rare combos -> spread out further).
        share = min(1.0, match_share(locks, config) or 1e-9)
        ranks = [r for r in allowed if rng.random() < share][:count]
        if len(ranks) < count:   # very rare combination: fill with the best ranks left
            ranks = sorted(ranks + [r for r in allowed if r not in ranks][:count - len(ranks)])
    return [{"rank": r, "scout_z": float(norm.isf((r - 0.5) / t.population)), "locks": locks}
            for r in ranks]


def origins(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Where each player is from: international or not, then the state/country
    (weighted by how many recruits it produces), then a hometown in it.
    Players with identity locks get home AND position drawn together from the
    locks' joint table, so the combination is realistic."""
    o, places = config.origins, config.places
    state_picker = WeightedPicker(o.states)
    country_picker = WeightedPicker(o.countries) if o.countries else None
    joint_pickers: dict[Locks, WeightedPicker] = {}
    city_pickers: dict[str, WeightedPicker] = {}
    for r in recruits:
        lk = locks_of(r)
        if lk.shapes_identity:
            if lk not in joint_pickers:
                joint_pickers[lk] = WeightedPicker(joint_table(lk, config))
            code, international, pos = joint_pickers[lk].pick(rng)
            r["position"] = pos                  # positions stage keeps this
        else:
            international = country_picker is not None and rng.random() < o.international_rate
            code = (country_picker if international else state_picker).pick(rng)
        info = places.countries[code] if international else places.states[code]
        if code not in city_pickers:
            city_pickers[code] = WeightedPicker(info.cities)
        city = city_pickers[code].pick(rng)
        r["origin"] = code
        r["international"] = international   # working value, not in the CSV
        r["city"] = city                     # working value, not in the CSV
        r["hometown"] = f"{city}, {info.name}" if international else f"{city}, {code}"
    return recruits


def positions(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Weighted draw: each position comes up about as often as its share says.
    (Players whose position was already set by their locks keep it.)"""
    picker = WeightedPicker({p: c.share for p, c in config.positions.items()})
    for r in recruits:
        if "position" not in r:
            r["position"] = picker.pick(rng)
    return recruits


def archetypes(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Play style and second position, weighted by the position's settings.
    Runs before physicals because some styles shape the body (Rim Protectors run taller)."""
    arch_pickers = {p: WeightedPicker(c.archetypes) for p, c in config.positions.items() if c.archetypes}
    second_pickers = {p: WeightedPicker({k: v for k, v in c.secondary.items()} or {"": 1})
                      for p, c in config.positions.items()}
    for r in recruits:
        pos = r["position"]
        locked = locks_of(r).archetype
        r["archetype"] = locked or (arch_pickers[pos].pick(rng) if pos in arch_pickers else "")
        r["secondary_position"] = second_pickers[pos].pick(rng) if second_pickers[pos] else ""
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


def feet_inches_half(inches: float) -> str:
    whole = int(inches // 12)
    rest = inches - whole * 12
    return f"{whole}'{rest:g}\""


def physicals(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Height from the player's position, then weight from that height, then wingspan.
    His archetype can nudge all three (shift is in spreads; wingspan in inches)."""
    samplers = {pos: make_height_sampler(p.height, rng) for pos, p in config.positions.items()}
    spreads = {pos: frame(p)[1] for pos, p in config.positions.items()}
    for r in recruits:
        pos_cfg = config.positions[r["position"]]
        arch = config.archetypes.get(r.get("archetype") or "")
        shift = arch.shift if arch else {}
        lk = locks_of(r)
        if lk.height is not None:
            height = lk.height
        else:
            height = int(round(samplers[r["position"]]() + shift.get("height", 0.0) * spreads[r["position"]]))
        w = pos_cfg.weight
        center = w.at_ref + w.lbs_per_inch * (height - w.ref_height) + shift.get("weight", 0.0) * w.sd
        r["height"] = height
        r["height_ft"] = feet_inches(height)
        # Cap the random swing at 2.5 sd each way (covers ~99% of real players) so a
        # freak draw can't make a 5'7" guard weigh 105 lbs.
        swing = float(np.clip(rng.normal(), -MAX_WEIGHT_SWING, MAX_WEIGHT_SWING))
        r["weight"] = lk.weight if lk.weight is not None else int(round(center + swing * w.sd))
        if pos_cfg.wingspan:
            extra, spread = pos_cfg.wingspan
            span = height + extra + spread * rng.normal() + shift.get("wingspan", 0.0)
            r["wingspan"] = round(span * 2) / 2                      # nearest half inch
            r["wingspan_ft"] = feet_inches_half(r["wingspan"])
    return recruits


def rank_within(recruits: list[dict], group_key: str, field: str) -> None:
    """Rank players inside each group (position, state...) by national rank: 1, 2, 3..."""
    groups = defaultdict(list)
    for r in recruits:
        groups[r[group_key]].append(r)
    for group in groups.values():
        for i, r in enumerate(sorted(group, key=lambda x: x["rank"]), start=1):
            r[field] = i


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
    True talent = what scouts saw + scouting error (bigger for international players
    and raw prospects; early bloomers tend to be overrated, late bloomers underrated).
    Where that true talent would rank him nationally decides steal / bust / as advertised."""
    t = config.talent
    for r in recruits:
        band, _ = star_band(r["rank"], t.stars)
        r["stars"] = band.stars
        r["rating"] = rating_for(r["rank"], t.stars)
        # How hard he is to scout: international players and raw prospects are harder.
        traj = config.development.trajectories[r["development"]] if r.get("development") else None
        error = t.scout_error * (t.international_error_mult if r.get("international") else 1.0)
        error *= 1.0 + (traj.noise if traj else 0.0)
        bias = (traj.bias if traj else 0.0) * t.scout_error   # + = scouts overrate this type
        true_z = r["scout_z"] - bias + rng.normal(0, error)
        r["true_z"] = true_z
        r["true_rank"] = max(1, math.ceil(t.population * norm.sf(true_z)))
        r["verdict"] = verdict_for(r["true_rank"], r["rank"], t)

    rank_within(recruits, "position", "pos_rank")
    return recruits


def ethnicity_mix(r: dict, config: SportConfig) -> dict:
    """US-born: the position's mix, nudged by the state (Hawaii -> far more Polynesian).
    International: the country's own mix."""
    return ethnicity_mix_for(r["position"], r["origin"], r["international"], config)


def high_school(r: dict, rng, config: SportConfig, prep_picker: WeightedPicker,
                pattern_picker: WeightedPicker) -> str:
    """Some players go to a national prep school (5-stars much more often).
    Everyone else gets a made-up local school built from the name patterns."""
    prep = config.prep
    if r["international"]:
        chance = prep.international_rate
    else:
        chance = min(0.9, prep.rate * prep.star_boost.get(r["stars"], 1.0))
    if rng.random() < chance:
        return prep_picker.pick(rng)
    names = config.places.schools
    if r["international"]:
        return names.international.format(city=r["city"])
    return pattern_picker.pick(rng).format(
        city=r["city"],
        dir=names.directions[int(rng.integers(len(names.directions)))],
        name=names.namesakes[int(rng.integers(len(names.namesakes)))],
        saint=names.saints[int(rng.integers(len(names.saints)))],
    )


def backgrounds(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """Ethnicity (from position + where he's from), high school, and state rank.
    Runs after talent because prep-school odds depend on stars."""
    prep_picker = WeightedPicker(config.prep.schools)
    pattern_picker = WeightedPicker({p.text: p.weight for p in config.places.schools.patterns})
    eth_pickers: dict[tuple, WeightedPicker] = {}
    for r in recruits:
        key = (r["position"], r["origin"], r["international"])
        if key not in eth_pickers:
            eth_pickers[key] = WeightedPicker(ethnicity_mix(r, config))
        r["ethnicity"] = locks_of(r).ethnicity or eth_pickers[key].pick(rng)
        r["high_school"] = high_school(r, rng, config, prep_picker, pattern_picker)

    rank_within(recruits, "origin", "state_rank")
    return recruits


def development(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """How each player grows: his trajectory and how far current ability sits below
    his ceiling (the gap). Runs before talent, because scouts misjudge some types."""
    if config.development is None:
        raise ValueError("No development settings in the sport config.")
    d = config.development
    picker = WeightedPicker({k: v.weight for k, v in d.trajectories.items()})
    for r in recruits:
        name = locks_of(r).development or picker.pick(rng)
        traj = d.trajectories[name]
        r["development"] = name
        r["growth"] = traj.growth
        r["gap"] = traj.gap * max(0.2, rng.normal(1.0, d.gap_spread))   # working value
    return recruits


def ratings(recruits: list[dict], rng, config: SportConfig) -> list[dict]:
    """2K ratings, OVR and potential from position, archetype, body and talent (see ratings.py)."""
    if config.ratings is None or config.attributes is None:
        raise ValueError("No ratings/attributes settings in the sport config.")
    for r in recruits:
        rate(r, rng, config)
    return recruits


def names(recruits: list[dict], rng, config: SportConfig, distinctive: str = "normal") -> list[dict]:
    """Realistic, unique names that fit each player's background (see names.py).
    distinctive = low / normal / high: how often creative African-American names come up."""
    if config.names is None:
        raise ValueError("No name settings loaded (configs/shared/names/names.yaml).")
    name_class(recruits, rng, name_book(config.names, distinctive), config.places)
    return recruits
