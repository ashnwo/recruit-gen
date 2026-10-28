import csv
from pathlib import Path
from typing import Optional

import typer

from recruit_gen.checks import ClassCheckError
from recruit_gen.models import Recruit
from recruit_gen.pipeline import build_class
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
    """One row per recruit. Columns follow the Recruit contract; the 2K ratings
    are spread out into one column each, in edit-screen order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = [f for f in Recruit.model_fields if f != "ratings"]
    attrs = list(recruits[0].get("ratings", {})) if recruits else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base + attrs)
        writer.writeheader()
        for r in recruits:
            writer.writerow({**{k: r[k] for k in base}, **r.get("ratings", {})})


def write_rejects(rejects: list[dict], path: Path) -> None:
    """Every player the clean-up check turned away, with the reasons."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rejects[0]))
        writer.writeheader()
        writer.writerows(rejects)


def rejects_path(out: Path) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_rejects.csv")


app = typer.Typer()


@app.command()
def generate(
    sport: str = "basketball",
    count: int = 10,
    seed: int = 42,
    config: Optional[Path] = None,
    out: Path = Path("recruits.csv"),
    top: bool = typer.Option(False, "--top", help="Best N in the country instead of a realistic mix."),
    distinctive: str = typer.Option("normal", help="How often creative names appear: low, normal or high."),
):
    if count < 1:
        typer.echo("--count must be at least 1.")
        raise typer.Exit(1)
    sport_config = load_sport_config(find_config(sport, config))
    pool = sport_config.talent.pool_size
    if count > pool:
        typer.echo(f"--count can be at most {pool} for {sport_config.sport} "
                   f"(that's how many players are ranked; see pool_size in the config).")
        raise typer.Exit(1)
    if distinctive not in ("low", "normal", "high"):
        typer.echo("--distinctive must be low, normal or high.")
        raise typer.Exit(1)
    try:
        result = build_class(count, seed, sport_config, top=top, distinctive=distinctive)
    except ClassCheckError as e:
        typer.echo(f"Stopped by the clean-up check: {e}")
        raise typer.Exit(1)
    recruits = result.recruits
    write_csv(recruits, out)
    old_rejects = rejects_path(out)
    if result.rejects:
        write_rejects(result.rejects, old_rejects)
    elif old_rejects.exists():
        old_rejects.unlink()          # don't leave a stale rejects file from an earlier run
    stars = {s: sum(r["stars"] == s for r in recruits) for s in (5, 4, 3, 2)}
    mix = ", ".join(f"{s}-star {n}" for s, n in stars.items() if n)
    kind = f"top {len(recruits)}" if top else f"{len(recruits)}"
    typer.echo(f"Wrote {kind} {sport_config.sport} recruits to {out}  ({mix})")
    replaced = len({x["rank"] for x in result.rejects})
    if replaced:
        typer.echo(f"Clean-up check: replaced {replaced} player(s) who failed; reasons in {rejects_path(out)}")
    else:
        typer.echo("Clean-up check: every player passed.")


if __name__ == "__main__":
    app()
