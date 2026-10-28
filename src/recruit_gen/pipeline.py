import numpy as np

from recruit_gen.models import PhysicalsConfig
from recruit_gen.stages import (
    class_shape, origins, positions, physicals,
    talent, development, names, validate_dedup,
)


def run_pipeline(count: int, seed: int, config: PhysicalsConfig) -> list[dict]:
    rng = np.random.default_rng(seed)
    recruits = class_shape(count, rng)          # source: makes the list
    recruits = origins(recruits, rng)           # transforms: enrich it
    recruits = positions(recruits, rng)
    recruits = physicals(recruits, rng, config)
    recruits = talent(recruits, rng)
    recruits = development(recruits, rng)
    recruits = names(recruits, rng)
    recruits = validate_dedup(recruits, rng)    # contract check: last line of defense
    return recruits
