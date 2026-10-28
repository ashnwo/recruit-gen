import numpy as np
from scipy.stats import skewnorm

def build(rng):
    def skewnormal(**args):
        return skewnorm.rvs(args)
    return {"normal": rng.normal, "skewnormal": skewnormal}

rng = np.random.default_rng(42)
samplers = build(rng)
print(samplers["skewnormal"](a=4, loc=190, scale=20))

rng = np.random.default_rng(42)
# samplers = build(rng)
print(samplers["skewnormal"](a=4, loc=190, scale=20))