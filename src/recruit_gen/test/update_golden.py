"""Rebuild the golden file after an INTENDED change to the pipeline or config.

    uv run python src/recruit_gen/test/update_golden.py

Then look at the diff (git diff) before committing, so you know exactly what changed.
"""
import json
from pathlib import Path

from recruit_gen.cli import find_config
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import load_sport_config

GOLDEN = Path(__file__).parent / "basketball_seed42.json"

if __name__ == "__main__":
    config = load_sport_config(find_config("basketball"))
    GOLDEN.write_text(json.dumps(run_pipeline(count=80, seed=42, config=config), indent=2) + "\n")
    print(f"Updated {GOLDEN}")
