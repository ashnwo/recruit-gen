import numpy as np

from recruit_gen.checks import ClassResult, finish, validate_dedup
from recruit_gen.locks import Locks, Spec
from recruit_gen.models import SportConfig
from recruit_gen.names import name_book, name_class
from recruit_gen.stages import (
    class_shape, origins, positions, archetypes, physicals, development,
    talent, backgrounds, ratings, names, rank_within,
)

# Stages that build one player at a time (the clean-up check reuses these to
# build a replacement player for a failed rank).
PLAYER_STAGES = [origins, positions, archetypes, physicals, development, talent, backgrounds, ratings]


def build_board(specs: list[Spec], seed: int, config: SportConfig, top: bool = False,
                distinctive: str = "normal") -> ClassResult:
    """One class from one or more requests ("3 white guards + 2 Texas 5-stars + 40 anyone").
    Each request gets its own ranks; then every stage runs once over the whole class,
    honoring each player's locks. Returns the clean class plus every reject."""
    rng = np.random.default_rng(seed)
    recruits, taken = [], set()
    for spec in specs:                                 # source: who's in the class (by rank)
        batch = class_shape(spec.count, rng, config, top, spec.locks, taken)
        taken |= {r["rank"] for r in batch}
        recruits += batch
    recruits.sort(key=lambda r: r["rank"])
    for stage in PLAYER_STAGES:                        # origins -> ... -> ratings
        recruits = stage(recruits, rng, config)
    recruits = names(recruits, rng, config, distinctive)   # needs ethnicity + origin

    def rebuild(failed: dict, kept: list[dict]) -> dict:
        """A brand-new player at the same national rank, with the same locks."""
        new = [{"rank": failed["rank"], "scout_z": failed["scout_z"], "locks": failed.get("locks")}]
        for stage in PLAYER_STAGES:
            new = stage(new, rng, config)
        name_class(new, rng, name_book(config.names, distinctive), config.places, taken=kept)
        return new[0]

    result = validate_dedup(recruits, config, rebuild)    # clean-up check: last line of defense
    rank_within(result.recruits, "position", "pos_rank")  # replacements may change these
    rank_within(result.recruits, "origin", "state_rank")
    return finish(result)


def build_class(count: int, seed: int, config: SportConfig, top: bool = False,
                distinctive: str = "normal", locks: Locks | None = None) -> ClassResult:
    """One request: `count` recruits, optionally with locks."""
    return build_board([Spec(count=count, locks=locks or Locks())], seed, config, top, distinctive)


def run_pipeline(count: int, seed: int, config: SportConfig, top: bool = False,
                 distinctive: str = "normal", locks: Locks | None = None) -> list[dict]:
    """Just the finished class (see build_class for the rejects too)."""
    return build_class(count, seed, config, top, distinctive, locks).recruits
