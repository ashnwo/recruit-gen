"""Rebuild the golden file after an INTENDED change to the pipeline.

    python src/recruit_gen/test/update_golden.py

Then look at the diff (git diff) before committing, so you know exactly what changed.
"""
import json
from pathlib import Path

from recruit_gen.models import parse_physicals_config
from recruit_gen.pipeline import run_pipeline

CONFIG = parse_physicals_config({"loc": 79.0, "scale": 2.5, "distribution": "normal"})
GOLDEN = Path(__file__).parent / "basketball_seed42.json"

if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(run_pipeline(count=80, seed=42, config=CONFIG), indent=2) + "\n")
    print(f"Updated {GOLDEN}")
