import numpy as np

from recruit_gen.models import SportConfig
from recruit_gen.stages import (
    class_shape, origins, positions, physicals,
    talent, development, names, validate_dedup,
)


def run_pipeline(count: int, seed: int, config: SportConfig) -> list[dict]:
    rng = np.random.default_rng(seed)
    recruits = class_shape(count, rng)            # source: makes the list
    recruits = origins(recruits, rng)             # transforms: enrich it
    recruits = positions(recruits, rng, config)
    recruits = physicals(recruits, rng, config)   # body depends on position
    recruits = talent(recruits, rng)
    recruits = development(recruits, rng)
    recruits = names(recruits, rng)
    recruits = validate_dedup(recruits, rng)      # contract check: last line of defense
    return recruits
