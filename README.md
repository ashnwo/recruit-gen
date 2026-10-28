# Recruit Generator

Makes realistic fictional high school recruits for sports video games and saves them as
a spreadsheet (CSV) you read while building players in the game:

| Sport | Built for | How you use the sheet |
|---|---|---|
| Basketball | NBA 2K26 (MyNBA draft classes) | Create the players in the draft class editor |
| Football | EA College Football 26/27 (Dynasty), Madden | Re-skin the game's own top recruits to match the sheet (CFB has no import) |

Every player hangs together: his body fits his position, his ratings fit his body and
talent, his name fits his background, and his hometown fits where real recruits come from.

---

## Setup (once)

You need [uv](https://docs.astral.sh/uv/). From the project folder:

```bash
uv sync          # installs everything the project needs
uv run pytest    # optional: runs the checks (should all pass)
```

## Quick start

```bash
uv run python -m recruit_gen.cli                        # asks you questions, one at a time
uv run python -m recruit_gen.cli --count 80             # 80 basketball recruits, a realistic mix
uv run python -m recruit_gen.cli --count 25 --top       # the 25 best in the country
uv run python -m recruit_gen.cli --sport football --count 40 --top
```

Each run prints a quick table, a class check, and where the files went:

```
output/basketball_80_seed123456.csv     the edit sheet (one row per player, one column per rating)
output/basketball_80_seed123456.json    the same class arranged for writing backstories
```

**Same class again:** every run prints its seed. Add `--seed 123456` to get the exact same class.

## Asking for specific players

Lock any field. Everything you leave out is filled in to match.

```bash
uv run python -m recruit_gen.cli --position guard --ethnicity white --count 3
uv run python -m recruit_gen.cli --position C --stars 5 --count 3 --top
uv run python -m recruit_gen.cli --state TX --stars 4-5 --count 10
uv run python -m recruit_gen.cli --height 7-0 --development late
uv run python -m recruit_gen.cli --first Jaylen --last Okafor --position PF
uv run python -m recruit_gen.cli --sport football --position OL --ethnicity polynesian --count 4
```

| Option | Examples |
|---|---|
| `--position` | `PG`, `C`, `PG,SG`, or a group: `guard`, `wing`, `forward`, `big` (football: `OL`, `DL`, `DB`, `skill`...) |
| `--ethnicity` | `black`, `white`, `hispanic`, `asian`, `polynesian` |
| `--state` (or `--country`, `--from`) | `TX`, `Georgia`, `Nigeria` |
| `--stars` | `5`, `4-5`, `4+`, `3,4` |
| `--archetype` | `"Rim Protector"`, `"Speed Rusher"` (see the sport's settings file) |
| `--development` | `Early Bloomer`, `Steady Climber`, `Late Bloomer`, `Raw Project` (or just `late`) |
| `--height` / `--weight` | `6-5` or `77` / `215` |
| `--first` / `--last` | any name |

Locks shape the rest realistically. Lock "white" and you get more big men; lock 7'0" and
he's almost always a center; lock "Polynesian" and he's most likely from Hawaii, Utah,
California or Texas. With `--top`, a position lock gives the best players at that
position, ranked where they'd really rank (the top 3 centers aren't simply #1-#3).

Impossible requests are refused with a reason ("No PG plays as a Rim Protector").
Unusual ones get a heads-up and are made anyway, because you asked.

### Many custom players at once (a "board")

Make a CSV with one request per row; blank cells are filled in. See `examples/board.csv`:

```
count,position,ethnicity,state,stars,archetype,development,height,weight,first,last
1,PG,,TX,5,Floor General,,,,,
2,guard,white,,4-5,,,,,,
1,C,,,5,Rim Protector,,7-2,,,
```

```bash
uv run python -m recruit_gen.cli --spec-file examples/board.csv
```

### Other options

| Option | What it does |
|---|---|
| `--distinctive low/normal/high` | How often creative names (DeAndre, Ja'Marr, Zy'Quez) appear |
| `--out path/file.csv` | Save somewhere else |
| `--show 30` | Rows in the terminal table (`0` = none) |
| `--no-json` | Skip the backstory file |
| `--no-report` | Skip the class check |

`uv run python -m recruit_gen.cli --help` lists everything.

---

## What's in a player

- **Rank, stars, rating:** national rank in a talent pyramid (a few 5-stars, many 2-3 stars), 247-style rating
- **Name:** from real birth records (boys born 2005-2010) and Census surnames, checked to fit his background. Heritage families get matching first names (Okafor, Diallo, Jean-Baptiste, Patel, Khan...). International players get names from their country.
- **Position, archetype, second position**
- **Body:** height and weight that fit the position (no undersized linemen), wingspan (basketball), 40-yard dash (football)
- **Ratings:** every in-game attribute in edit-screen order, plus OVR and potential (his ceiling)
- **Development:** Early Bloomer / Steady Climber / Late Bloomer / Raw Project, and dev trait (football)
- **Background:** hometown, state rank, ethnicity, high school (top recruits often go to national prep schools)
- **The hidden truth:** `true_rank` and `verdict`. Scouts aren't perfect: some players are steals, some are busts. Early bloomers tend to be overrated and late bloomers underrated, and international players are harder to scout.

## The checks that run every time

1. **Clean-up check.** Every player must pass the rules (sane values, believable size for
   his position, no duplicate names). A player who fails is listed in `<file>_rejects.csv`
   with the reason and replaced by a new player at the same rank. If more than 5% fail,
   the run stops and names the problem, because that means a setting is off.
2. **Class check.** Compares the class's ethnicity, positions, stars and international share
   to what your settings promise. A `!` means a difference bigger than chance explains.

## The JSON file (for backstories)

For a separate Claude project that writes player backstories. Each recruit is grouped into
`bio`, `body`, `scouting` and `ratings`, with his top 3 `strengths` and `weaknesses` and
`story_hints` (plain facts like "Ranked #14, but his real talent is closer to #2"). The
`hidden_truth` section is what scouts don't know: good for foreshadowing.

---

## Changing the settings

Everything realistic about the players lives in settings files, not in the code:

| File | What's in it |
|---|---|
| `configs/basketball.yaml` | Positions (share, height, weight, ethnicity, roles, archetypes), talent and star bands, where players come from, prep schools, development, ratings, archetypes, size limits |
| `configs/football.yaml` | Same format for football |
| `configs/shared/places.yaml` | States, countries, hometowns, ethnicity nudges by state, high school name patterns |
| `configs/shared/names/names.yaml` | Name rules, creative-name styling, nicknames, suffixes, heritage lists, international name pools, famous names to avoid |
| `configs/shared/names/*.csv` | The big name tables (from public SSA and Census data) |

Change a number, run again. If a setting is wrong (a typo in a position name, shares that
don't add up to 1), the run stops and tells you what and where. Each file has comments
explaining every setting.

**Don't like a name?** Add it to `never_use` in `configs/shared/names/names.yaml`.

**Adding a sport** means adding a `configs/<sport>.yaml` file in the same format. The
engine never checks which sport it's running (a test makes sure of that).

---

## For developers

```
src/recruit_gen/
  cli.py        command line, question mode, output files
  pipeline.py   runs the stages in order: build_board / build_class / run_pipeline
  stages.py     class shape -> positions -> origins -> archetypes -> physicals ->
                development -> talent -> backgrounds -> ratings -> names
  locks.py      custom requests: parsing, the joint table, request checks
  names.py      name engine         ratings.py   ratings, OVR, potential
  checks.py     clean-up check      report.py    class check vs. targets
  export.py     JSON lore file      models.py    every setting and the Recruit contract
  sampling.py   weighted draws
  test/         the tests (uv run pytest)
```

**Golden files.** `test/basketball_seed42.json` and `test/football_seed42.json` pin down the
exact output for seed 42. If you change the settings or code **on purpose**, rebuild them
and look at the diff before committing:

```bash
uv run python src/recruit_gen/test/update_golden.py
git diff --stat
```

## Troubleshooting

- **`Expected a Python module at: src/recruit_gen/__init__.py`:** that file got deleted
  (usually by dragging a folder in Finder and choosing "Replace"). Restore it with
  `git restore src/recruit_gen/__init__.py`. Unzip updates with `unzip -o file.zip` in
  Terminal instead.
- **`No config for 'basketball'`:** run the command from the project folder, or point to
  the file with `--config path/to/basketball.yaml`.
- **`Stopped by the clean-up check`:** a recent settings change makes too many players
  fail. The message names the most common problem.

## Data sources

- SSA baby names by birth year, 2005-2010 (public domain)
- US Census 2010 surname file with race/ethnicity shares (public domain)
- Rosenman et al. (2023) first-name race shares from US voter files
