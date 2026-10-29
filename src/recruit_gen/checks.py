"""Clean-up check (Piece 7): the last line of defense.

Every player must pass:
  1. the Recruit contract (models.py): right fields, right types, sane values
  2. realism rules from the sport config (checks: body size by position, wingspan)
  3. no duplicates: no two players with the same name or the same national rank

A player who fails is NOT silently fixed or dropped. He's recorded as a reject
(with every reason) and a new player is built for the same rank. If too many
players fail, the whole run stops: that points at a bad setting, and replacing
players would hide it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from recruit_gen.models import Recruit, SportConfig
from recruit_gen.names import plain


class ClassCheckError(Exception):
    """The class can't be built cleanly. The message says why."""


@dataclass
class ClassResult:
    recruits: list[dict]                 # the finished class, every player passed
    rejects: list[dict] = field(default_factory=list)   # every failed attempt, with reasons


def contract_problems(r: dict) -> list[str]:
    try:
        Recruit.model_validate(r)
        return []
    except ValidationError as e:
        return [f"{'.'.join(str(x) for x in err['loc']) or 'record'}: {err['msg']}" for err in e.errors()]


def realism_problems(r: dict, config: SportConfig) -> list[str]:
    c = config.checks
    if c is None:
        return []
    out = []
    bounds = c.body.get(r.get("position", ""))
    lk = r.get("locks")
    for attr, unit in (("height", "in"), ("weight", "lbs")):
        if lk is not None and getattr(lk, attr) is not None:
            continue                     # you locked it on purpose (you got a warning up front)
        if attr == "weight" and lk is not None and lk.height is not None:
            continue                     # his weight follows the height you locked
        rng_ = getattr(bounds, attr) if bounds else None
        value = r.get(attr)
        if rng_ and value is not None and not rng_[0] <= value <= rng_[1]:
            out.append(f"{r['position']} {attr} {value} {unit} outside {rng_[0]:g}-{rng_[1]:g}")
    height_locked = lk is not None and lk.height is not None
    if c.wingspan_vs_height and not height_locked and r.get("wingspan") is not None and r.get("height") is not None:
        diff = r["wingspan"] - r["height"]
        lo, hi = c.wingspan_vs_height
        if not lo <= diff <= hi:
            out.append(f"wingspan {diff:+g} in vs height outside {lo:+g} to {hi:+g}")
    return out


def duplicate_problems(r: dict, seen_names: set, seen_ranks: set) -> list[str]:
    out = []
    name = plain(f"{r.get('first_name', '')} {r.get('last_name', '')}")
    if name and name in seen_names:
        out.append(f"duplicate name {r.get('first_name')} {r.get('last_name')}")
    if r.get("rank") in seen_ranks:
        out.append(f"duplicate rank {r.get('rank')}")
    return out


def problem_kind(reason: str) -> str:
    """Group reasons for the summary: 'PG height 76 in outside 74-74' -> 'PG height'."""
    words = reason.split()
    if " outside" in reason:
        return "wingspan" if words[0] == "wingspan" else " ".join(words[:2])
    if reason.startswith("duplicate"):
        return " ".join(words[:2])
    return reason.split(":")[0]


def reject_row(r: dict, reasons: list[str], attempt: int) -> dict:
    """What gets written to the rejects file: who it was, why, and which try."""
    keep = ("rank", "stars", "projection", "position", "archetype", "first_name", "last_name", "height", "weight",
            "wingspan", "ovr", "potential", "hometown", "team")
    return {"attempt": attempt, **{k: r.get(k, "") for k in keep}, "reasons": "; ".join(reasons)}


def validate_dedup(recruits: list[dict], config: SportConfig, rebuild) -> ClassResult:
    """Check every player; replace failures using rebuild(failed_player, kept_players).
    Returns the clean class plus a record of every reject."""
    c = config.checks
    max_rebuilds = c.max_rebuilds if c else 0
    kept, rejects, stuck = [], [], []
    seen_names, seen_ranks = set(), set()
    for r in recruits:
        attempt = 0
        while True:
            reasons = contract_problems(r) + realism_problems(r, config) + duplicate_problems(r, seen_names, seen_ranks)
            if not reasons:
                break
            attempt += 1
            rejects.append(reject_row(r, reasons, attempt))
            if attempt > max_rebuilds:
                stuck.append((r.get("rank"), reasons))
                break
            r = rebuild(r, kept)
        if reasons:
            continue                     # no passing player for this rank; reported below
        kept.append(r)
        seen_names.add(plain(f"{r['first_name']} {r['last_name']}"))
        seen_ranks.add(r["rank"])

    # Big picture first: lots of failures means a setting is off.
    failed_ranks = {x["rank"] for x in rejects}
    limit = c.max_reject_share if c else 1.0
    allowed = max(limit * len(recruits), c.always_allow if c else 0)
    if len(failed_ranks) > allowed:
        top = Counter(problem_kind(reason) for x in rejects
                      for reason in x["reasons"].split("; ")).most_common(3)
        summary = ", ".join(f"{why} ({n})" for why, n in top)
        raise ClassCheckError(
            f"{len(failed_ranks)} of {len(recruits)} players failed the clean-up check "
            f"(limit {limit:.0%}). Most common problems: {summary}. "
            f"A setting in your sport config is probably off.")
    if stuck:
        rank, reasons = stuck[0]
        raise ClassCheckError(
            f"Couldn't build a passing player for rank {rank} after {max_rebuilds + 1} tries. "
            f"Last problem: {'; '.join(reasons)}. Check the settings in your sport config.")
    return ClassResult(recruits=kept, rejects=rejects)


def finish(result: ClassResult) -> ClassResult:
    """Keep only the contract's fields (drops working values like scout_z)."""
    result.recruits = [Recruit.model_validate(r).model_dump() for r in result.recruits]
    return result
