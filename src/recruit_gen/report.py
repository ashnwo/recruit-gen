"""Class check vs. real-life targets (Piece 10).

After a class is built, compare what came out to what the settings say should come
out: ethnicity, positions, stars (or projected round, for draft classes) and
international share. Random classes never
match targets exactly, so a difference only gets a "!" when it's bigger than
chance explains (more than 3 standard errors AND more than 3 percentage points).
That's the same idea as a data-quality check on any pipeline: did the output
drift from what the inputs promise?
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from scipy.stats import norm

from recruit_gen.locks import ethnicity_mix_for, origin_odds
from recruit_gen.models import ETHNICITIES, SportConfig


@dataclass
class Line:
    label: str
    actual: float
    target: float | None

    def off(self, n: int) -> bool:
        """Further from the target than chance explains?"""
        if self.target is None:
            return False
        se = math.sqrt(max(self.target * (1 - self.target), 1e-9) / n)
        gap = abs(self.actual - self.target)
        return gap > 3 * se and gap > 0.03


def position_targets(ranks: list[int], config: SportConfig) -> dict[str, float]:
    """Expected position mix for players at these ranks (elite tilt makes some
    positions rarer near the top, so this depends on which ranks are in the class)."""
    t = config.talent
    out = Counter()
    for rank in ranks:
        idx = max(float(norm.isf(min(max((rank - 0.5) / t.pool_size, 1e-6), 1 - 1e-6))), 0.0)
        w = {p: c.share * math.exp(c.elite_tilt * idx) for p, c in config.positions.items()}
        total = sum(w.values())
        for p, v in w.items():
            out[p] += v / total / len(ranks)
    return dict(out)


def background_targets(pos_mix: dict[str, float], config: SportConfig) -> tuple[dict[str, float], float]:
    """Expected ethnicity mix and international share, given the position mix."""
    eth, intl = Counter(), 0.0
    for pos, p_pos in pos_mix.items():
        for (code, international), p_origin in origin_odds(config, position=pos).items():
            for e, p_e in ethnicity_mix_for(pos, code, international, config).items():
                eth[e] += p_pos * p_origin * p_e
            intl += p_pos * p_origin * international
    return dict(eth), intl


def star_targets(config: SportConfig) -> dict[int, float]:
    """In a realistic mix, each star level's share of the ranked pool."""
    t, first, out = config.talent, 1, {}
    for band in t.stars:
        last = min(band.last_rank, t.pool_size)
        out[band.stars] = max(0, last - first + 1) / t.pool_size
        first = band.last_rank + 1
    return out


def projection_targets(config: SportConfig) -> dict[str, float]:
    """Draft sports, realistic mix: each projected round's share of the ranked pool."""
    first, out = 1, {}
    for band in config.draft.projection:
        last = min(band.last_rank, config.talent.pool_size)
        out[band.label] = max(0, last - first + 1) / config.talent.pool_size
        first = band.last_rank + 1
    return out


def class_report(recruits: list[dict], config: SportConfig, top: bool, custom: bool) -> dict[str, list[Line]]:
    n = len(recruits)
    share = lambda key, value: sum(r[key] == value for r in recruits) / n   # noqa: E731
    with_targets = not custom
    pos_t = position_targets([r["rank"] for r in recruits], config)
    eth_t, intl_t = background_targets(pos_t, config)
    draft = config.draft is not None
    star_t = star_targets(config) if not top and not draft else {}
    report = {}
    if draft:
        proj_t = projection_targets(config) if not top else {}
        report["Projection"] = [Line(b.label, share("projection", b.label),
                                     proj_t.get(b.label) if with_targets else None)
                                for b in config.draft.projection]
    report |= {
        "Ethnicity": [Line(e, share("ethnicity", e), eth_t.get(e) if with_targets else None)
                      for e in ETHNICITIES if share("ethnicity", e) > 0 or eth_t.get(e, 0) >= 0.01],
        "Positions": [Line(p, share("position", p), pos_t.get(p) if with_targets else None)
                      for p in config.positions],
        # Draft sports: stars are high school background, so there's no target to hit.
        ("HS stars" if draft else "Stars"): [
            Line(f"{s}*", share("stars", s), star_t.get(s) if with_targets else None)
            for s in sorted({b.stars for b in config.talent.stars}, reverse=True)
            if not draft or share("stars", s) > 0],
        "International": [Line("share", sum(r["origin"] in config.places.countries for r in recruits) / n,
                               intl_t if with_targets else None)],
    }
    return report


def format_report(report: dict[str, list[Line]], n: int, custom: bool) -> str:
    title = ("Class check (custom request: shown without targets)" if custom else
             "Class check vs. targets in your settings  (! = further off than chance explains)")
    lines = [title]
    for name, items in report.items():
        cells = []
        for line in items:
            cell = f"{line.label} {line.actual:.0%}"
            if line.target is not None:
                cell += f" ({line.target:.0%})" + ("!" if line.off(n) else "")
            cells.append(cell)
        lines.append(f"  {name + ':':<15}" + "  ".join(cells))
    flagged = [f"{name} {line.label}" for name, items in report.items() for line in items if line.off(n)]
    if flagged:
        lines.append(f"  Off target: {', '.join(flagged)}. Fine once in a while; if it keeps happening, check the settings.")
    elif not custom:
        lines.append("  Everything is within normal random variation.")
    return "\n".join(lines)
