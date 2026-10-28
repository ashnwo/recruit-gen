"""Rebuild the golden files after an INTENDED change to the pipeline or configs.

    uv run python src/recruit_gen/test/update_golden.py

Then look at the diff (git diff) before committing, so you know exactly what changed.
"""
import json
from pathlib import Path

from recruit_gen.cli import find_config
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import load_sport_config

HERE = Path(__file__).parent
GOLDEN = {"basketball": (80, HERE / "basketball_seed42.json"), "football": (120, HERE / "football_seed42.json")}

if __name__ == "__main__":
    for sport, (count, path) in GOLDEN.items():
        config = load_sport_config(find_config(sport))
        path.write_text(json.dumps(run_pipeline(count=count, seed=42, config=config), indent=2) + "\n")
        print(f"Updated {path}")
