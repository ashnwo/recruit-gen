"""JSON export for the backstory / lore workflow (Piece 10).

The CSV is the edit sheet you read while building players in the game. The JSON
is for a separate Claude project that writes backstories: each recruit grouped
into bio, body, scouting and ratings, plus his biggest strengths and weaknesses
and short "story hints" - plain facts a writer can build on. No AI runs here;
this file only arranges the data.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from recruit_gen.models import SportConfig

TOP_N = 3   # strengths / weaknesses listed per player


def key_ratings(r: dict, config: SportConfig) -> dict[str, int]:
    """Ratings that matter at his position: core/major skills plus body ratings that count toward OVR."""
    pos = config.positions[r["position"]]
    group_of = config.attributes.group_of
    default = config.ratings.default_role
    out = {}
    for attr, value in r["ratings"].items():
        if attr in group_of:
            role = pos.overrides.get(attr, pos.roles.get(group_of[attr], default))
            if role in ("core", "major"):
                out[attr] = value
        elif pos.ovr_body.get(attr, 0) > 0:
            out[attr] = value
    return out or dict(r["ratings"])


def story_hints(r: dict, config: SportConfig) -> list[str]:
    """Plain facts a backstory writer can use. Never invents anything."""
    hints = []
    where = r["hometown"]
    if r["origin"] in config.places.countries:
        hints.append(f"International prospect from {where}.")
    prep = config.prep.schools
    if r["high_school"] in prep:
        hints.append(f"Left home ({where}) to play at {r['high_school']}"
                     + (" in high school." if config.draft else "."))
    hints += career_hints(r, config)
    verdict, rank, true_rank = r["verdict"], r["rank"], r["true_rank"]
    if verdict == "Steal":
        hints.append(f"Ranked #{rank}, but his real talent is closer to #{true_rank}: a hidden gem scouts missed.")
    elif verdict == "Underrated":
        hints.append(f"Better than his #{rank} ranking suggests (more like #{true_rank}).")
    elif verdict == "Overrated":
        hints.append(f"Ranked #{rank} on hype; his real level is closer to #{true_rank}.")
    elif verdict == "Bust":
        hints.append(f"Ranked #{rank}, but his real talent is closer to #{true_rank}: a bust waiting to happen.")
    growth = {"Early Bloomer": "Physically ahead of his peers now; less room to grow.",
              "Late Bloomer": "Still filling out; big room to grow.",
              "Raw Project": "Raw and unpolished, with a high ceiling if he develops.",
              "Steady Climber": "Improving at a steady pace."}
    if r["development"] in growth:
        hints.append(growth[r["development"]])
    best = (r.get("projected_pick") or 999) <= 30 if config.draft else r["stars"] == 5
    if best and r["pos_rank"] == 1:
        hints.append(f"The top-ranked {r['position']} in this class.")
    if r["state_rank"] == 1 and r["origin"] not in config.places.countries:
        hints.append(f"The top-ranked {'prospect' if config.draft else 'recruit'} from {r['origin']} in this class.")
    if r.get("suffix"):
        hints.append(f"Named after a family member ({r['first_name']} {r['last_name']} {r['suffix']}).")
    return hints


def career_hints(r: dict, config: SportConfig) -> list[str]:
    """Draft classes: how his high school ranking compares to now, and where he plays."""
    if config.draft is None:
        return []
    hints, proj, stars = [], r["projection"], r["stars"]
    now = {"Lottery": "as a lottery pick", "1st round": "as a 1st-round pick", "2nd round": "as a 2nd-round pick",
           "Undrafted": "to go undrafted"}.get(proj, proj.lower())
    early = proj in [b.label for b in config.draft.projection[:2]]          # lottery / 1st round
    if stars is not None and stars <= 3 and early:
        hints.append(f"Only a {stars}-star recruit in high school; now projected {now}. A big riser.")
    elif stars == 5 and not early:
        hints.append(f"A 5-star recruit in high school, but now projected {now}. His stock has slipped.")
    elif stars is not None:
        hints.append(f"A {stars}-star recruit in high school; now projected {now}.")
    level, team, year = r.get("level"), r.get("team"), r.get("class_year")
    if level == "College":
        if year == "Fr" and early:
            hints.append(f"Freshman at {team}, expected to be one-and-done.")
        elif year == "Sr":
            hints.append(f"Senior at {team}: a four-year college player.")
        else:
            hints.append(f"{ {'Fr': 'Freshman', 'So': 'Sophomore', 'Jr': 'Junior'}.get(year, year)} at {team}.")
    elif level == "Overseas pro":
        hints.append(f"Already a pro, playing for {team} (age {r['age']}).")
    elif level:
        hints.append(f"Skipped college to play for {team} ({level}).")
    return hints


def lore_record(r: dict, config: SportConfig) -> dict:
    keys = key_ratings(r, config)
    ordered = sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "rank": r["rank"],
        "stars": r["stars"],             # draft classes: high school recruiting stars (background)
        "rating": r["rating"],
        "name": {"first": r["first_name"], "last": r["last_name"], "suffix": r["suffix"],
                 "full": " ".join(x for x in (r["first_name"], r["last_name"], r["suffix"]) if x)},
        "position": r["position"],
        "secondary_position": r["secondary_position"],
        "archetype": r["archetype"],
        "bio": {"hometown": r["hometown"], "origin": r["origin"], "ethnicity": r["ethnicity"],
                "high_school": r["high_school"]},
        "body": {"height_in": r["height"], "height": r["height_ft"], "weight_lbs": r["weight"],
                 "wingspan_in": r["wingspan"], "wingspan": r["wingspan_ft"] or None, "forty": r["forty"]},
        "career": ({"team": r["team"], "level": r["level"], "class_year": r["class_year"] or None,
                    "age": r["age"]} if r.get("team") else
                   {"age": r["age"]} if r.get("age") else None),
        "scouting": {"national_rank": r["rank"], "position_rank": r["pos_rank"],
                     "projection": r.get("projection") or None, "projected_pick": r.get("projected_pick"),
                     "state_rank": r["state_rank"],
                     "development": r["development"], "growth": r["growth"], "dev_trait": r["dev_trait"] or None,
                     "hidden_truth": {"true_rank": r["true_rank"], "verdict": r["verdict"]}},
        "ratings": {"ovr": r["ovr"], "potential": r["potential"],
                    "strengths": [{"rating": k, "value": v} for k, v in ordered[:TOP_N]],
                    "weaknesses": [{"rating": k, "value": v} for k, v in ordered[::-1][:TOP_N]],
                    "all": r["ratings"]},
        "story_hints": story_hints(r, config),
    }


def lore_json(recruits: list[dict], config: SportConfig, seed: int, top: bool, requests: list[str]) -> dict:
    return {
        "class": {
            "sport": config.sport,
            "game": config.game,
            "seed": seed,
            "count": len(recruits),
            "mode": "top" if top else "realistic mix",
            "requests": requests,
            "made_on": date.today().isoformat(),
            "note": "hidden_truth is what scouts don't know; use it for foreshadowing, not as public info."
                    + (" These are draft prospects: rank is the big board, stars are from high school."
                       if config.draft else " These are high school recruits."),
        },
        "recruits": [lore_record(r, config) for r in recruits],
    }


def write_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
