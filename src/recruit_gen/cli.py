import csv
from pathlib import Path

import typer

from recruit_gen.models import Recruit
from recruit_gen.pipeline import run_pipeline
from recruit_gen.stages import load_physicals_config

SUPPORTED_SPORTS = ["basketball"]   # football arrives in Piece 9


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
    config: Path = Path("config.yaml"),
    out: Path = Path("recruits.csv"),
):
    if sport not in SUPPORTED_SPORTS:
        typer.echo(f"Only {', '.join(SUPPORTED_SPORTS)} works right now. Football comes later.")
        raise typer.Exit(1)
    if count < 1:
        typer.echo("--count must be at least 1.")
        raise typer.Exit(1)
    physicals_config = load_physicals_config(config)
    recruits = run_pipeline(count, seed, physicals_config)
    write_csv(recruits, out)
    typer.echo(f"Wrote {len(recruits)} recruits to {out}")


if __name__ == "__main__":
    app()
