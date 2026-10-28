import csv
from pathlib import Path
from typing import Optional

import typer

from recruit_gen.models import Recruit
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import load_sport_config

PACKAGE_DIR = Path(__file__).resolve().parent     # src/recruit_gen
REPO_ROOT = PACKAGE_DIR.parent.parent             # top of the repo (src layout)
# Where sport configs can live, checked in this order.
CONFIG_DIRS = list(dict.fromkeys([Path.cwd() / "configs", REPO_ROOT / "configs", PACKAGE_DIR / "configs"]))


def available_sports() -> list[str]:
    return sorted({f.stem for d in CONFIG_DIRS if d.is_dir() for f in d.glob("*.yaml")})


def find_config(sport: str, explicit: Optional[Path] = None) -> Path:
    """Use --config if given. Otherwise find configs/<sport>.yaml, checking the
    folder you ran the command from, the top of the repo, then the package folder.
    Adding a sport = adding a YAML file; no code change."""
    if explicit is not None:
        return explicit
    for d in CONFIG_DIRS:
        if (d / f"{sport}.yaml").exists():
            return d / f"{sport}.yaml"
    looked = "\n".join(f"  {d / (sport + '.yaml')}" for d in CONFIG_DIRS)
    have = ", ".join(available_sports()) or "none found"
    typer.echo(f"No config for '{sport}'. Looked in:\n{looked}\n"
               f"Sports available: {have}\n"
               f"Or point to a file with:  --config path/to/file.yaml")
    raise typer.Exit(1)


def write_csv(recruits: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        # Column order comes from the Recruit contract, so it never depends on luck.
        writer = csv.DictWriter(f, fieldnames=list(Recruit.model_fields))
        writer.writeheader()
        writer.writerows(recruits)


app = typer.Typer()


@app.command()
def generate(
    sport: str = "basketball",
    count: int = 10,
    seed: int = 42,
    config: Optional[Path] = None,
    out: Path = Path("recruits.csv"),
):
    if count < 1:
        typer.echo("--count must be at least 1.")
        raise typer.Exit(1)
    sport_config = load_sport_config(find_config(sport, config))
    recruits = run_pipeline(count, seed, sport_config)
    write_csv(recruits, out)
    typer.echo(f"Wrote {len(recruits)} {sport_config.sport} recruits to {out}")


if __name__ == "__main__":
    app()
