"""Custom requests (Piece 8): lock any field; everything else fills in to match.

Locks don't just get stamped on at the end. They shape the rest of the player:
  - position, home and ethnicity are drawn TOGETHER from one table, so locking
    "white" makes centers more likely, and locking "Polynesian" makes Hawaii,
    Utah and California more likely - the way real recruiting classes look
  - locking a height makes positions that fit that height more likely (7'0" -> C)
  - locking an archetype limits positions to ones that play that style
  - locking stars limits which national ranks he can have
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
from scipy.stats import norm, skewnorm

from recruit_gen.models import ETHNICITIES, Ethnicity, SportConfig

ETHNICITY_ALIASES = {
    "black": "Black", "african-american": "Black", "african american": "Black", "aa": "Black",
    "white": "White", "caucasian": "White",
    "hispanic": "Hispanic", "latino": "Hispanic", "latin": "Hispanic", "mexican": "Hispanic",
    "asian": "Asian", "asian-american": "Asian", "asian american": "Asian",
    "polynesian": "Polynesian", "samoan": "Polynesian", "tongan": "Polynesian",
    "hawaiian": "Polynesian", "pacific islander": "Polynesian", "islander": "Polynesian",
}
CONTRACT_HEIGHT = (60, 96)
CONTRACT_WEIGHT = (100, 400)


class LockError(ValueError):
    """A request that can't be understood or can't be met. The message says why."""


class Locks(BaseModel):
    """Fields fixed by hand. Anything left as None is generated."""
    model_config = ConfigDict(frozen=True)       # frozen = can be used as a cache key
    position: Optional[tuple[str, ...]] = None   # one or more positions ("guard" -> PG, SG)
    ethnicity: Optional[Ethnicity] = None
    origin: Optional[str] = None                 # state code (TX) or country code (NGA)
    stars: Optional[tuple[int, ...]] = None
    archetype: Optional[str] = None
    development: Optional[str] = None
    height: Optional[int] = None                 # inches
    weight: Optional[int] = None                 # lbs
    first: Optional[str] = None
    last: Optional[str] = None

    @property
    def any(self) -> bool:
        return any(v is not None for v in self.model_dump().values())

    @property
    def shapes_identity(self) -> bool:
        """Does this lock change who he is (position/home/ethnicity)?"""
        return any(getattr(self, f) is not None for f in ("position", "ethnicity", "origin", "archetype", "height"))

    def describe(self) -> str:
        parts = []
        for k, v in self.model_dump().items():
            if v is not None:
                parts.append(f"{k}={'/'.join(map(str, v)) if isinstance(v, tuple) else v}")
        return ", ".join(parts) or "no locks"


@dataclass
class Spec:
    """One request: 'make N recruits with these locks'."""
    count: int
    locks: Locks


# ---- reading what the user typed ---------------------------------------------------
def parse_position(text: str, config: SportConfig) -> tuple[str, ...]:
    out: list[str] = []
    groups = {k.lower(): v for k, v in config.position_groups.items()}
    for part in str(text).split(","):
        t = part.strip()
        if t.upper() in config.positions:
            out.append(t.upper())
        elif t.lower() in groups:
            out.extend(groups[t.lower()])
        else:
            raise LockError(f"Unknown {config.sport} position '{t}'. Use one of: "
                            f"{', '.join(config.positions)}, or a group: {', '.join(config.position_groups)}")
    return tuple(dict.fromkeys(out))


def parse_ethnicity(text: str) -> str:
    t = str(text).strip().lower()
    if t in ETHNICITY_ALIASES:
        return ETHNICITY_ALIASES[t]
    raise LockError(f"Unknown ethnicity '{text}'. Use one of: {', '.join(e.lower() for e in ETHNICITIES)}")


def parse_origin(text: str, config: SportConfig) -> str:
    """'TX', 'texas', 'NGA', 'nigeria' -> code."""
    t = str(text).strip().lower()
    places = config.places
    for code, info in list(places.states.items()) + list(places.countries.items()):
        if t in (code.lower(), info.name.lower()):
            return code
    raise LockError(f"Unknown state or country '{text}'. Use a code like TX or a name like Nigeria.")


def parse_stars(text: str) -> tuple[int, ...]:
    """'5', '4-5', '4+', '3,4' -> star levels."""
    t = str(text).replace(" ", "").replace("*", "").lower().replace("stars", "").replace("star", "")
    try:
        if t.endswith("+"):
            out = tuple(range(int(t[:-1]), 6))
        elif "-" in t:
            a, b = (int(x) for x in t.split("-"))
            out = tuple(range(min(a, b), max(a, b) + 1))
        else:
            out = tuple(int(x) for x in t.split(","))
    except ValueError:
        raise LockError(f"Can't read stars '{text}'. Use 5, 4-5, 4+ or 3,4.") from None
    if not out or any(not 1 <= s <= 5 for s in out):
        raise LockError(f"Stars must be between 1 and 5, got '{text}'.")
    return out


def parse_height(text) -> int:
    """'6-5', 6'5", '6 5', '77' -> inches."""
    t = str(text).strip().replace('"', "").replace("''", "")
    m = re.fullmatch(r"(\d)\s*['\-\s]\s*(\d{1,2}(?:\.\d)?)", t)
    if m:
        return int(round(int(m.group(1)) * 12 + float(m.group(2))))
    if t.isdigit() and CONTRACT_HEIGHT[0] <= int(t) <= CONTRACT_HEIGHT[1]:
        return int(t)
    raise LockError(f"Can't read height '{text}'. Use 6-5 or 77 (inches).")


def parse_weight(text) -> int:
    t = str(text).strip().lower().replace("lbs", "").replace("lb", "").strip()
    if t.isdigit():
        return int(t)
    raise LockError(f"Can't read weight '{text}'. Use pounds, like 215.")


def parse_choice(text: str, options, what: str) -> str:
    """Case-insensitive match; a unique start also works ('late' -> 'Late Bloomer')."""
    t = str(text).strip().lower()
    exact = [o for o in options if o.lower() == t]
    if exact:
        return exact[0]
    starts = [o for o in options if o.lower().startswith(t)]
    if len(starts) == 1:
        return starts[0]
    raise LockError(f"Unknown {what} '{text}'. Use one of: {', '.join(options)}")


def build_locks(config: SportConfig, position=None, ethnicity=None, origin=None, stars=None,
                archetype=None, development=None, height=None, weight=None,
                first=None, last=None) -> Locks:
    """Turn typed text (any of it can be blank) into checked Locks."""
    blank = lambda v: v is None or str(v).strip() == ""   # noqa: E731
    return Locks(
        position=None if blank(position) else parse_position(position, config),
        ethnicity=None if blank(ethnicity) else parse_ethnicity(ethnicity),
        origin=None if blank(origin) else parse_origin(origin, config),
        stars=None if blank(stars) else parse_stars(stars),
        archetype=None if blank(archetype) else parse_choice(archetype, list(config.archetypes), "archetype"),
        development=None if blank(development) else parse_choice(
            development, list(config.development.trajectories) if config.development else [], "development"),
        height=None if blank(height) else parse_height(height),
        weight=None if blank(weight) else parse_weight(weight),
        first=None if blank(first) else str(first).strip(),
        last=None if blank(last) else str(last).strip(),
    )


def read_spec_file(path: Path, config: SportConfig) -> list[Spec]:
    """A CSV 'board': one request per row. Columns (all optional except count):
    count, position, ethnicity, state (or origin/country), stars, archetype,
    development, height, weight, first, last"""
    specs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            try:
                count = int(row.get("count") or 1)
                if count < 1:
                    raise LockError("count must be at least 1")
                locks = build_locks(config, position=row.get("position"), ethnicity=row.get("ethnicity"),
                                    origin=row.get("state") or row.get("origin") or row.get("country"),
                                    stars=row.get("stars"), archetype=row.get("archetype"),
                                    development=row.get("development"), height=row.get("height"),
                                    weight=row.get("weight"), first=row.get("first"), last=row.get("last"))
            except (LockError, ValueError) as e:
                raise LockError(f"{Path(path).name} line {line}: {e}") from None
            specs.append(Spec(count=count, locks=locks))
    if not specs:
        raise LockError(f"{Path(path).name} has no requests in it.")
    return specs


# ---- the joint table: who could this player realistically be? ---------------------
def ethnicity_mix_for(position: str, origin: str, international: bool, config: SportConfig) -> dict:
    """US-born: the position's mix, nudged by the state (Hawaii -> far more Polynesian).
    International: the country's own mix. Always adds up to 1."""
    places = config.places
    if international:
        mix = dict(places.countries[origin].ethnicity)
    else:
        pos_mix = config.positions[position].ethnicity
        nudge = places.states[origin].eth_mult
        default = places.default_eth_mult
        mix = {e: pos_mix.get(e, 0.0) * nudge.get(e, default.get(e, 1.0)) for e in ETHNICITIES}
    total = sum(mix.values())
    return {e: v / total for e, v in mix.items()} if total else mix


def origin_odds(config: SportConfig, locked: Optional[str] = None,
                position: Optional[str] = None) -> dict[tuple[str, bool], float]:
    """Chance of each home (code, international?), for a player at `position`.
    A locked home not in the sport's list still works: it's simply the only option.
    A position's origin_override mixes in its special places (Australian punters)."""
    o, places = config.origins, config.places
    if locked:
        return {(locked, locked in places.countries): 1.0}
    out = {}
    st, ct = sum(o.states.values()), sum(o.countries.values())
    for code, w in o.states.items():
        out[(code, False)] = (1 - o.international_rate) * w / st
    for code, w in o.countries.items():
        out[(code, True)] = o.international_rate * w / ct
    ov = config.positions[position].origin_override if position else None
    if ov:
        special = {(c, False): w for c, w in ov.states.items()} | {(c, True): w for c, w in ov.countries.items()}
        total = sum(special.values())
        out = {k: v * (1 - ov.rate) for k, v in out.items()}
        for k, w in special.items():
            out[k] = out.get(k, 0.0) + ov.rate * w / total
    return out


def height_odds(position: str, height: int, config: SportConfig) -> float:
    """How likely this exact height is at this position (one-inch slice of the curve)."""
    h = config.positions[position].height
    if h.distribution == "skewnormal":
        return float(skewnorm.cdf(height + 0.5, h.a, h.loc, h.scale) - skewnorm.cdf(height - 0.5, h.a, h.loc, h.scale))
    return float(norm.cdf(height + 0.5, h.loc, h.scale) - norm.cdf(height - 0.5, h.loc, h.scale))


_JOINT_CACHE: dict = {}


def joint_table(locks: Locks, config: SportConfig) -> dict[tuple[str, bool, str], float]:
    """{(home, international?, position): weight} for players matching the locks.
    Built once per (locks, config) and cached."""
    key = (locks, id(config))
    if key in _JOINT_CACHE:
        return _JOINT_CACHE[key][1]
    positions = locks.position or tuple(config.positions)
    table = {}
    for pos in positions:
        pc = config.positions[pos]
        for (code, intl), p_origin in origin_odds(config, locks.origin, pos).items():
            w = p_origin * pc.share
            if locks.ethnicity:
                w *= ethnicity_mix_for(pos, code, intl, config).get(locks.ethnicity, 0.0)
            if locks.archetype:
                total = sum(pc.archetypes.values())
                w *= pc.archetypes.get(locks.archetype, 0.0) / total if total else 0.0
            if locks.height is not None:
                w *= height_odds(pos, locks.height, config)
            if w > 0:
                table[(code, intl, pos)] = w
    _JOINT_CACHE[key] = (config, table)      # keep config alive so its id stays unique
    return table


def match_share(locks: Locks, config: SportConfig) -> float:
    """Roughly what share of ranked recruits would match these locks."""
    share = sum(joint_table(locks, config).values())
    if locks.development and config.development:
        tr = config.development.trajectories
        share *= tr[locks.development].weight / sum(t.weight for t in tr.values())
    return share


def star_ranks(stars: Optional[tuple[int, ...]], config: SportConfig) -> list[int]:
    """National ranks that carry these star levels (all ranks if no star lock)."""
    t = config.talent
    if not stars:
        return list(range(1, t.pool_size + 1))
    out, first = [], 1
    for band in t.stars:
        if band.stars in stars:
            out.extend(range(first, min(band.last_rank, t.pool_size) + 1))
        first = band.last_rank + 1
    return out


# ---- checking a request before building it ---------------------------------------
def check_request(spec: Spec, config: SportConfig) -> list[str]:
    """Raise LockError if the request can't be met; return warnings for unusual ones."""
    lk, warnings = spec.locks, []
    if lk.height is not None and not CONTRACT_HEIGHT[0] <= lk.height <= CONTRACT_HEIGHT[1]:
        raise LockError(f"Height {lk.height} in is outside what the generator allows (5'0\" to 8'0\").")
    if lk.weight is not None and not CONTRACT_WEIGHT[0] <= lk.weight <= CONTRACT_WEIGHT[1]:
        raise LockError(f"Weight {lk.weight} lbs is outside what the generator allows (100-400).")
    if lk.first and lk.last and spec.count > 1:
        raise LockError("You locked both the first and last name, so only 1 player can be made (names are unique).")
    if lk.archetype:
        allowed = lk.position or tuple(config.positions)
        if not any(lk.archetype in config.positions[p].archetypes for p in allowed):
            plays = [p for p, c in config.positions.items() if lk.archetype in c.archetypes]
            raise LockError(f"No {'/'.join(allowed)} plays as a {lk.archetype}. "
                            f"That style is for: {', '.join(plays)}.")
    ranks = star_ranks(lk.stars, config)
    if spec.count > len(ranks):
        what = f"{'/'.join(map(str, lk.stars))}-star" if lk.stars else "ranked"
        raise LockError(f"Only {len(ranks)} {what} players exist in a class; you asked for {spec.count}.")
    if lk.shapes_identity and not joint_table(lk, config):
        raise LockError(f"No realistic player matches: {lk.describe()}. "
                        f"(For example, a country with no {lk.ethnicity} recruits in places.yaml.)")
    # Unusual but allowed: you asked for it, so you get it - with a heads-up.
    if config.checks and (lk.height is not None or lk.weight is not None):
        allowed = lk.position or tuple(config.positions)
        for attr, value, unit in (("height", lk.height, "in"), ("weight", lk.weight, "lbs")):
            if value is None:
                continue
            fits = [p for p in allowed if (b := config.checks.body.get(p)) is None or getattr(b, attr) is None
                    or getattr(b, attr)[0] <= value <= getattr(b, attr)[1]]
            if not fits:
                warnings.append(f"{attr} {value} {unit} is unusual for {'/'.join(allowed)}; "
                                f"making him anyway because you asked for it.")
    return warnings
