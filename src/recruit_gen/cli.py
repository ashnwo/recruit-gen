"""Command line.

  uv run python -m recruit_gen.cli                         (asks you questions)
  uv run python -m recruit_gen.cli --count 80
  uv run python -m recruit_gen.cli --count 25 --top
  uv run python -m recruit_gen.cli --position guard --ethnicity white
  uv run python -m recruit_gen.cli --position C --stars 5 --count 3 --state TX
  uv run python -m recruit_gen.cli --spec-file examples/board.csv
"""
import csv
import secrets
import sys
from pathlib import Path
from typing import Optional

import typer

from recruit_gen.checks import ClassCheckError
from recruit_gen.locks import LockError, Spec, build_locks, check_request, read_spec_file
from recruit_gen.models import Recruit
from recruit_gen.pipeline import build_board
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


def default_out(sport: str, count: int, seed: int, top: bool, custom: bool) -> Path:
    """output/basketball_80_seed1234.csv, output/basketball_25_top_seed7.csv,
    output/basketball_3_custom_seed99.csv ... in the folder you run the command from."""
    parts = [sport, str(count)] + (["top"] if top else []) + (["custom"] if custom else []) + [f"seed{seed}"]
    return Path.cwd() / "output" / ("_".join(parts) + ".csv")


def summary_table(recruits: list[dict], show: int) -> str:
    """A quick look at the class in the terminal (the CSV has everything)."""
    if show <= 0 or not recruits:
        return ""
    lines = [f"{'#':>4} {'*':>1}  {'Name':<26} {'Pos':<5} {'Ht':>5} {'Wt':>4} {'OVR':>4} {'POT':>4}  {'Hometown':<24} Verdict",
             "-" * 100]
    for r in recruits[:show]:
        name = " ".join(x for x in (r["first_name"], r["last_name"], r["suffix"]) if x)
        lines.append(f"{r['rank']:>4} {r['stars']:>1}  {name[:26]:<26} {r['position']:<5} {r['height_ft']:>5} "
                     f"{r['weight']:>4} {r['ovr']:>4} {r['potential']:>4}  {r['hometown'][:24]:<24} {r['verdict']}")
    if len(recruits) > show:
        lines.append(f"... and {len(recruits) - show} more in the CSV")
    return "\n".join(lines)


def wizard(ask=None) -> dict:
    """Question-by-question mode (runs when you give no options). Enter = default."""
    ask = ask or (lambda text, default="": typer.prompt(text, default=default, show_default=bool(default)))
    typer.echo("\n=== Recruit Generator ===  (press Enter to take the [default])\n")
    sports = available_sports() or ["basketball"]
    answers = {"sport": ask(f"Sport ({' / '.join(sports)})", default=sports[0])}
    typer.echo("Want a specific type of player? Leave any of these blank.")
    answers["position"] = ask("Position (PG, SG, SF, PF, C, or guard / wing / forward / big)", default="")
    answers["ethnicity"] = ask("Ethnicity (black / white / hispanic / asian / polynesian)", default="")
    answers["origin"] = ask("Home state or country (TX, Georgia, Nigeria...)", default="")
    answers["stars"] = ask("Stars (5, 4-5, 3+)", default="")
    custom = any(answers[k] for k in ("position", "ethnicity", "origin", "stars"))
    answers["count"] = ask("How many recruits", default="1" if custom else "50")
    answers["top"] = ask("Best in the country only? (y/n)", default="n").strip().lower().startswith("y")
    answers["seed"] = ask("Seed (same seed = same class; blank = random)", default="")
    return answers


app = typer.Typer(add_completion=False)


@app.command()
def generate(
    sport: str = typer.Option("basketball", help="Sport (a file in configs/)."),
    count: Optional[int] = typer.Option(None, help="How many recruits (default 50, or 1 if you lock fields)."),
    seed: Optional[int] = typer.Option(None, help="Same seed + same options = same class. Blank = random."),
    top: bool = typer.Option(False, "--top", help="Best N in the country instead of a realistic mix."),
    position: Optional[str] = typer.Option(None, help="PG, C, 'PG,SG', or a group: guard, wing, forward, big."),
    ethnicity: Optional[str] = typer.Option(None, help="black, white, hispanic, asian, polynesian."),
    origin: Optional[str] = typer.Option(None, "--state", "--country", "--from", help="Home state or country: TX, Georgia, Nigeria."),
    stars: Optional[str] = typer.Option(None, help="5, 4-5, 4+, or 3,4."),
    archetype: Optional[str] = typer.Option(None, help="Play style, e.g. 'Rim Protector' (see configs/<sport>.yaml)."),
    development: Optional[str] = typer.Option(None, help="Early Bloomer, Steady Climber, Late Bloomer, Raw Project."),
    height: Optional[str] = typer.Option(None, help="6-5 or 77 (inches)."),
    weight: Optional[str] = typer.Option(None, help="Pounds."),
    first: Optional[str] = typer.Option(None, help="First name."),
    last: Optional[str] = typer.Option(None, help="Last name."),
    spec_file: Optional[Path] = typer.Option(None, help="CSV of many custom requests (one per row)."),
    distinctive: str = typer.Option("normal", help="How often creative names appear: low, normal or high."),
    out: Optional[Path] = typer.Option(None, help="Output CSV (default: output/<sport>_<count>_seed<seed>.csv)."),
    show: int = typer.Option(15, help="Rows to print in the terminal (0 = none)."),
    config: Optional[Path] = typer.Option(None, help="Use this sport config file instead."),
):
    """Generate a class of realistic fictional recruits as a CSV edit sheet."""
    if distinctive not in ("low", "normal", "high"):
        typer.echo("--distinctive must be low, normal or high.")
        raise typer.Exit(1)
    sport_config = load_sport_config(find_config(sport, config))
    try:
        if spec_file:
            specs = read_spec_file(spec_file, sport_config)
        else:
            locks = build_locks(sport_config, position=position, ethnicity=ethnicity, origin=origin,
                                stars=stars, archetype=archetype, development=development,
                                height=height, weight=weight, first=first, last=last)
            n = count if count is not None else (1 if locks.any else 50)
            if n < 1:
                raise LockError("--count must be at least 1.")
            specs = [Spec(count=n, locks=locks)]
        for spec in specs:
            for warning in check_request(spec, sport_config):
                typer.echo(f"Heads up: {warning}")
        total = sum(s.count for s in specs)
        if total > sport_config.talent.pool_size:
            raise LockError(f"That's {total} recruits; at most {sport_config.talent.pool_size} are ranked "
                            f"in {sport_config.sport} (pool_size in the config).")
    except LockError as e:
        typer.echo(f"Can't make that: {e}")
        raise typer.Exit(1)

    seed = seed if seed is not None else secrets.randbelow(1_000_000)
    custom = bool(spec_file) or any(s.locks.any for s in specs)
    out = out or default_out(sport_config.sport, total, seed, top, custom)
    try:
        result = build_board(specs, seed, sport_config, top=top, distinctive=distinctive)
    except ClassCheckError as e:
        typer.echo(f"Stopped by the clean-up check: {e}")
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Can't make that: {e}")
        raise typer.Exit(1)

    recruits = result.recruits
    write_csv(recruits, out)
    old_rejects = rejects_path(out)
    if result.rejects:
        write_rejects(result.rejects, old_rejects)
    elif old_rejects.exists():
        old_rejects.unlink()          # don't leave a stale rejects file from an earlier run

    table = summary_table(recruits, show)
    if table:
        typer.echo("\n" + table + "\n")
    star_mix = {s: sum(r["stars"] == s for r in recruits) for s in (5, 4, 3, 2)}
    mix = ", ".join(f"{s}-star {n}" for s, n in star_mix.items() if n)
    kind = f"top {len(recruits)}" if top else f"{len(recruits)}"
    typer.echo(f"Wrote {kind} {sport_config.sport} recruits to {out}  ({mix})")
    replaced = len({x["rank"] for x in result.rejects})
    if replaced:
        typer.echo(f"Clean-up check: replaced {replaced} player(s) who failed; reasons in {rejects_path(out)}")
    else:
        typer.echo("Clean-up check: every player passed.")
    typer.echo(f"Same class again: add --seed {seed}")


def main() -> None:
    """Entry point. No options at all = question-by-question mode."""
    if len(sys.argv) == 1:
        a = wizard()
        args = ["--sport", a["sport"], "--count", a["count"]]
        for flag, key in (("--position", "position"), ("--ethnicity", "ethnicity"), ("--state", "origin"),
                          ("--stars", "stars"), ("--seed", "seed")):
            if a[key]:
                args += [flag, a[key]]
        if a["top"]:
            args.append("--top")
        sys.argv += args
    app()


if __name__ == "__main__":
    main()
