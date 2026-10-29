"""Command line.

  uv run python -m recruit_gen.cli                         (asks you questions)
  uv run python -m recruit_gen.cli --count 80
  uv run python -m recruit_gen.cli --count 25 --mix               (realistic spread)
  uv run python -m recruit_gen.cli --position guard --ethnicity white
  uv run python -m recruit_gen.cli --position C --projection lottery --count 3
  uv run python -m recruit_gen.cli --sport football --position QB --stars 5 --state TX
  uv run python -m recruit_gen.cli --spec-file examples/board.csv
"""
import csv
import secrets
import sys
from pathlib import Path
from typing import Optional

import typer

from recruit_gen.checks import ClassCheckError
from recruit_gen.export import lore_json, write_json
from recruit_gen.locks import LockError, Spec, build_locks, check_request, read_spec_file
from recruit_gen.models import Recruit, SportConfig
from recruit_gen.pipeline import build_board
from recruit_gen.report import class_report, format_report
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


def unused_columns(config: Optional[SportConfig]) -> set[str]:
    """Columns this sport never fills (no 40-yard dash in basketball, no college team
    for football recruits...). Left out of the CSV so the sheet stays readable."""
    if config is None:
        return set()
    drop = set()
    if not any(p.wingspan for p in config.positions.values()):
        drop |= {"wingspan", "wingspan_ft"}
    if config.forty is None:
        drop.add("forty")
    if config.dev_traits is None:
        drop.add("dev_trait")
    if config.draft is None:
        drop |= {"projection", "projected_pick"}
    if config.teams is None:
        drop |= {"team", "level", "class_year"}
        if not config.ages:
            drop.add("age")
    return drop


def write_csv(recruits: list[dict], path: Path, config: Optional[SportConfig] = None) -> None:
    """One row per recruit. Columns follow the Recruit contract (minus ones the sport
    never uses); the ratings are spread out into one column each, in edit-screen order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    drop = unused_columns(config)
    base = [f for f in Recruit.model_fields if f != "ratings" and f not in drop]
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
    draft = any(r.get("projection") for r in recruits)
    teams = any(r.get("team") for r in recruits)
    head = f"{'#':>4} {'*':>1}  " + (f"{'Proj':<9} " if draft else "")
    head += f"{'Name':<26} {'Pos':<5} {'Ht':>5} {'Wt':>4} {'OVR':>4} {'POT':>4}  "
    head += f"{'Team':<24} " if teams else f"{'Hometown':<24} "
    lines = [head + "Verdict", "-" * (len(head) + 10)]
    short = {"Lottery": "Lottery", "1st round": "1st", "2nd round": "2nd", "Undrafted": "Undrafted"}
    for r in recruits[:show]:
        name = " ".join(x for x in (r["first_name"], r["last_name"], r["suffix"]) if x)
        stars = r["stars"] if r["stars"] is not None else "-"
        row = f"{r['rank']:>4} {stars:>1}  "
        if draft:
            row += f"{short.get(r['projection'], r['projection'])[:9]:<9} "
        row += (f"{name[:26]:<26} {r['position']:<5} {r['height_ft']:>5} "
                f"{r['weight']:>4} {r['ovr']:>4} {r['potential']:>4}  ")
        if teams:
            where = r["team"] + (f" ({r['class_year']})" if r.get("class_year") else "")
            row += f"{where[:24]:<24} "
        else:
            row += f"{r['hometown'][:24]:<24} "
        lines.append(row + r["verdict"])
    if len(recruits) > show:
        lines.append(f"... and {len(recruits) - show} more in the CSV")
    if draft:
        lines.append("* = high school recruiting stars (- = international, wasn't a US recruit)")
    return "\n".join(lines)


def sport_is_draft(sport: str) -> bool:
    """Does this sport's settings file describe a draft class (vs. high school recruits)?"""
    try:
        for d in CONFIG_DIRS:
            if (d / f"{sport}.yaml").exists():
                return load_sport_config(d / f"{sport}.yaml").draft is not None
    except Exception:
        pass
    return False


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
    draft = sport_is_draft(answers["sport"])
    if draft:
        answers["projection"] = ask("Projected round (lottery / 1st / 2nd / undrafted)", default="")
    else:
        answers["stars"] = ask("Stars (5, 4-5, 3+)", default="")
    custom = any(answers.get(k) for k in ("position", "ethnicity", "origin", "stars", "projection"))
    answers["count"] = ask("How many players", default="1" if custom else ("60" if draft else "50"))
    question = "Top of the draft board only? (y = best N, n = realistic mix)" if draft else "Best in the country only? (y/n)"
    answers["top"] = ask(question, default="y" if draft else "n").strip().lower().startswith("y")
    answers["seed"] = ask("Seed (same seed = same class; blank = random)", default="")
    return answers


app = typer.Typer(add_completion=False)


@app.command()
def generate(
    sport: str = typer.Option("basketball", help="Sport (a file in configs/)."),
    count: Optional[int] = typer.Option(None, help="How many recruits (default 50, or 1 if you lock fields)."),
    seed: Optional[int] = typer.Option(None, help="Same seed + same options = same class. Blank = random."),
    top: Optional[bool] = typer.Option(None, "--top/--mix", help="--top = the best N (top of the board); --mix = a realistic spread. Default comes from the sport's settings (draft classes: top; recruits: mix)."),
    position: Optional[str] = typer.Option(None, help="PG, C, 'PG,SG', or a group: guard, wing, forward, big."),
    ethnicity: Optional[str] = typer.Option(None, help="black, white, hispanic, asian, polynesian."),
    origin: Optional[str] = typer.Option(None, "--state", "--country", "--from", help="Home state or country: TX, Georgia, Nigeria."),
    stars: Optional[str] = typer.Option(None, help="Recruiting stars: 5, 4-5, 4+, or 3,4 (high school recruit sports)."),
    projection: Optional[str] = typer.Option(None, help="Draft sports: lottery, 1st, 2nd, undrafted (or '1st,2nd')."),
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
    json_out: bool = typer.Option(True, "--json/--no-json", help="Also save a JSON file for the backstory/lore workflow."),
    report: bool = typer.Option(True, "--report/--no-report", help="Print the class check vs. targets."),
    config: Optional[Path] = typer.Option(None, help="Use this sport config file instead."),
):
    """Generate a class of realistic fictional recruits as a CSV edit sheet."""
    if distinctive not in ("low", "normal", "high"):
        typer.echo("--distinctive must be low, normal or high.")
        raise typer.Exit(1)
    sport_config = load_sport_config(find_config(sport, config))
    if top is None:
        top = sport_config.draft.default_top if sport_config.draft else False
    try:
        if spec_file:
            specs = read_spec_file(spec_file, sport_config)
        else:
            locks = build_locks(sport_config, position=position, ethnicity=ethnicity, origin=origin,
                                stars=stars, archetype=archetype, development=development,
                                height=height, weight=weight, first=first, last=last,
                                projection=projection)
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
    write_csv(recruits, out, sport_config)
    old_rejects = rejects_path(out)
    if result.rejects:
        write_rejects(result.rejects, old_rejects)
    elif old_rejects.exists():
        old_rejects.unlink()          # don't leave a stale rejects file from an earlier run

    json_path = out.with_suffix(".json")
    if json_out:
        requests = [f"{s.count} x {s.locks.describe()}" for s in specs]
        write_json(lore_json(recruits, sport_config, seed, top, requests), json_path)

    table = summary_table(recruits, show)
    if table:
        typer.echo("\n" + table + "\n")
    if report:
        typer.echo(format_report(class_report(recruits, sport_config, top, custom), len(recruits), custom) + "\n")
    if sport_config.draft:
        bands = [b.label for b in sport_config.draft.projection]
        counts = {b: sum(r["projection"] == b for r in recruits) for b in bands}
        what = "prospects"
    else:
        counts = {f"{s}-star": sum(r["stars"] == s for r in recruits) for s in (5, 4, 3, 2)}
        what = "recruits"
    mix = ", ".join(f"{k} {n}" for k, n in counts.items() if n)
    kind = f"top {len(recruits)}" if top else f"{len(recruits)}"
    typer.echo(f"Wrote {kind} {sport_config.sport} {what} to {out}  ({mix})")
    if json_out:
        typer.echo(f"Lore file for backstories: {json_path}")
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
                          ("--stars", "stars"), ("--projection", "projection"), ("--seed", "seed")):
            if a.get(key):
                args += [flag, a[key]]
        args.append("--top" if a["top"] else "--mix")
        sys.argv += args
    app()


if __name__ == "__main__":
    main()
