import json, yaml
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import physicals, load_physicals_config


# recruits = run_pipeline(count=80, seed=42)
# with open("tests/golden/basketball_seed42.json", "w") as f:
#     json.dump(recruits, f, indent=2)

# import numpy as np
# rng = np.random.default_rng(42)
# dummy = [{} for _ in range(10000)]
# out = physicals(dummy, rng)

# heights = [r.height for r in out]
# print("mean:", np.mean(heights))
# print("sd:  ", np.std(heights))

data = load_physicals_config("basketball.yaml")
print(data)
print(type(data))