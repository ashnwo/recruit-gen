import numpy as np

from recruit_gen.models import SportConfig
from recruit_gen.stages import (
    class_shape, origins, positions, archetypes, physicals, development,
    talent, backgrounds, ratings, names, validate_dedup,
)


def run_pipeline(count: int, seed: int, config: SportConfig, top: bool = False,
                 distinctive: str = "normal") -> list[dict]:
    """Each stage adds to the recruits, in an order where every stage has what it needs."""
    rng = np.random.default_rng(seed)
    recruits = class_shape(count, rng, config, top)   # source: who's in the class (by rank)
    recruits = origins(recruits, rng, config)         # state/country + hometown
    recruits = positions(recruits, rng, config)
    recruits = archetypes(recruits, rng, config)      # play style (shapes the body) + 2nd position
    recruits = physicals(recruits, rng, config)       # height, weight, wingspan
    recruits = development(recruits, rng, config)     # trajectory + growth gap
    recruits = talent(recruits, rng, config)          # stars, rating, true talent, verdict, position rank
    recruits = backgrounds(recruits, rng, config)     # ethnicity, high school, state rank
    recruits = ratings(recruits, rng, config)         # 2K ratings, OVR, potential
    recruits = names(recruits, rng, config, distinctive)   # needs ethnicity + origin
    recruits = validate_dedup(recruits, rng)          # contract check: last line of defense
    return recruits
