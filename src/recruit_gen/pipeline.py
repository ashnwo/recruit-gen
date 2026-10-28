import numpy as np

from recruit_gen.models import SportConfig
from recruit_gen.stages import (
    class_shape, origins, positions, physicals,
    talent, backgrounds, development, names, validate_dedup,
)


def run_pipeline(count: int, seed: int, config: SportConfig, top: bool = False) -> list[dict]:
    rng = np.random.default_rng(seed)
    recruits = class_shape(count, rng, config, top)   # source: who's in the class (by rank)
    recruits = origins(recruits, rng, config)     # transforms: state/country + hometown
    recruits = positions(recruits, rng, config)
    recruits = physicals(recruits, rng, config)   # body depends on position
    recruits = talent(recruits, rng, config)      # stars, rating, true talent, position rank
    recruits = backgrounds(recruits, rng, config) # ethnicity, high school, state rank
    recruits = development(recruits, rng)
    recruits = names(recruits, rng)
    recruits = validate_dedup(recruits, rng)      # contract check: last line of defense
    return recruits
