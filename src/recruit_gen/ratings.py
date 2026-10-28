"""2K-style ratings (Piece 6).

  skill rating = level middle + level spread x talent + group bonus + archetype mod + noise
  body rating  = position base + athletic x A + height x H + weight x W + talent x T + noise
  OVR          = weighted average: core skills count most, low skills not at all
  potential    = center + per_sd x TRUE talent (+ noise), never below OVR

"talent" for ratings is CURRENT ability: true talent minus his development gap.
It's on a bell-curve scale for ranked recruits: 0 = typical ranked recruit,
+3 = the very best, -3 = the bottom of the rankings.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, skewnorm

from recruit_gen.models import PositionConfig, SportConfig


def frame(pos: PositionConfig) -> tuple[float, float, float, float]:
    """Average height, height spread, average weight and weight spread at a position."""
    h = pos.height
    if h.distribution == "skewnormal":
        mean_h, sd_h = skewnorm.mean(h.a, h.loc, h.scale), skewnorm.std(h.a, h.loc, h.scale)
    else:
        mean_h, sd_h = h.loc, h.scale
    w = pos.weight
    mean_w = w.at_ref + w.lbs_per_inch * (mean_h - w.ref_height)
    sd_w = math.sqrt(w.sd ** 2 + (w.lbs_per_inch * sd_h) ** 2)
    return float(mean_h), float(sd_h), float(mean_w), sd_w


def talent_index(true_z: float, config: SportConfig) -> float:
    """Population talent score -> bell-curve score among ranked recruits.
    Where would his true talent rank him? Turn that rank into a normal score."""
    t = config.talent
    share = t.population * norm.sf(true_z) / t.pool_size     # true rank as a share of the pool
    return float(norm.isf(min(max(share, 1e-4), 0.999)))


def mean_gap(config: SportConfig) -> float:
    """Average development gap. Added back so a typical ranked recruit's CURRENT talent is 0."""
    tr = config.development.trajectories.values()
    return sum(x.weight * x.gap for x in tr) / sum(x.weight for x in tr)


def rate(r: dict, rng, config: SportConfig) -> None:
    """Fill in ratings, ovr and potential for one recruit."""
    rc, attrs = config.ratings, config.attributes
    pos = config.positions[r["position"]]
    arch = config.archetypes.get(r.get("archetype", ""))
    mods = arch.mods if arch else {}
    shift = arch.shift if arch else {}

    true_t = talent_index(r["true_z"], config)
    current = true_t - r["gap"] + mean_gap(config)
    a = rc.athletic_from_talent
    athletic = a * true_t + math.sqrt(1 - a * a) * rng.normal() + shift.get("athletic", 0.0)
    mean_h, sd_h, mean_w, sd_w = frame(pos)
    hz, wz = (r["height"] - mean_h) / sd_h, (r["weight"] - mean_w) / sd_w

    group_of = attrs.group_of
    group_bonus = {g: rng.normal(0, rc.group_spread) for g in attrs.groups}
    out = {}
    for attr in attrs.order:
        if attr in attrs.body:
            load = attrs.body[attr]
            v = (pos.body.get(attr, 50) + load.athletic * athletic + load.height * hz
                 + load.weight * wz + load.talent * current + rng.normal(0, rc.attr_noise))
        elif attr in attrs.fixed:
            mu, sd = attrs.fixed[attr]
            v = rng.normal(mu, sd)
        else:
            group = group_of[attr]
            role = pos.overrides.get(attr, pos.roles.get(group, rc.default_role))
            middle, spread = rc.levels[role]
            v = middle + spread * current + group_bonus[group] + rng.normal(0, rc.attr_noise)
        v += mods.get(attr, 0.0)
        out[attr] = int(round(min(max(v, rc.min), rc.max)))
    r["ratings"] = out

    # OVR: skills weighted by their role at this position, plus the position's key body ratings.
    num = den = 0.0
    for attr, v in out.items():
        if attr in group_of:
            w = rc.ovr_weights.get(pos.overrides.get(attr, pos.roles.get(group_of[attr], rc.default_role)), 0.0)
        else:
            w = pos.ovr_body.get(attr, 0.0)
        num += w * v
        den += w
    r["ovr"] = int(round(num / den)) if den else rc.min

    p = rc.potential
    pot = p.center + p.per_sd * true_t + rng.normal(0, p.noise)
    pot = int(round(min(max(pot, p.min), p.max)))
    r["potential"] = max(pot, min(r["ovr"] + 1, rc.max))     # ceiling is never below current

    if config.dev_traits is not None:        # e.g. College Football: Normal / Impact / Star / Elite
        dv = config.dev_traits
        score = true_t + rng.normal(0, dv.noise)
        r["dev_trait"] = next((name for name, need in dv.levels.items() if score >= need),
                              list(dv.levels)[-1])
    if config.forty is not None:             # 40-yard dash from the speed rating
        f = config.forty
        t = f.at_95 + (95 - out[f.rating]) * f.per_point + rng.normal(0, f.noise)
        r["forty"] = round(min(max(t, f.min), f.max), 2)
