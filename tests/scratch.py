import json
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import physicals

# recruits = run_pipeline(count=80, seed=42)
# with open("tests/golden/basketball_seed42.json", "w") as f:
#     json.dump(recruits, f, indent=2)

import numpy as np
rng = np.random.default_rng(42)
dummy = [{} for _ in range(10000)]
out = physicals(dummy, rng)

heights = [r.height for r in out]
print("mean:", np.mean(heights))
print("sd:  ", np.std(heights))