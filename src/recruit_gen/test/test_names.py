"""Piece 5: names that fit each player."""
import re
from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from recruit_gen.models import SportConfig
from recruit_gen.names import name_book, name_class, plain, repeat_limits, to_ascii
from recruit_gen.pipeline import run_pipeline

US_POOLS = {"Black": "us_black", "White": "us_white", "Hispanic": "us_hispanic", "Asian": "us_asian"}


def named(basketball, ethnicity, n=3000, seed=1, origin="TX", international=False, distinctive="normal"):
    """Run the name step on players we pin to one background."""
    recruits = [{"rank": i + 1, "ethnicity": ethnicity, "origin": origin, "international": international}
                for i in range(n)]
    name_class(recruits, np.random.default_rng(seed), name_book(basketball.names, distinctive), basketball.places)
    return recruits


# ---- config rules ----------------------------------------------------------------
def test_names_config_loaded(basketball):
    assert basketball.names is not None
    assert (basketball.names.folder / "first_names.csv").exists()
    assert {"polynesian", "nigeria", "serbia"} <= set(basketball.names.pools)


def test_country_with_unknown_name_pool_is_rejected(basketball):
    d = basketball.model_dump()
    d["places"]["countries"]["NGA"]["name_pools"] = {"Black": {"atlantis": 1.0}}
    with pytest.raises(ValidationError, match="unknown name pools"):
        SportConfig.model_validate(d)


def test_heritage_with_unknown_pool_is_rejected(basketball):
    d = basketball.model_dump()
    d["names"]["heritage"]["west_african"]["pools"] = ["atlantis"]
    with pytest.raises(ValidationError, match="unknown pools"):
        SportConfig.model_validate(d)


# ---- every name ------------------------------------------------------------------
def test_every_recruit_has_a_clean_name(basketball):
    for r in run_pipeline(1000, seed=2, config=basketball):
        assert r["first_name"] and r["last_name"]
        assert r["first_name"].isascii() and r["last_name"].isascii()   # game editors want plain letters
        assert r["suffix"] in ("", "Jr.", "II", "III", "IV")


def test_no_repeated_names_in_a_class(basketball):
    recruits = run_pipeline(1000, seed=3, config=basketball)
    full = [plain(f'{r["first_name"]} {r["last_name"]}') for r in recruits]
    assert len(full) == len(set(full))
    max_first, max_last = repeat_limits(1000)
    assert max(Counter(r["last_name"] for r in recruits).values()) <= max_last


def test_first_name_never_matches_last_name(basketball):
    for eth in ("Black", "Asian"):
        for r in named(basketball, eth, n=3000, seed=4):
            assert plain(r["legal_first"]) != plain(r["last_name"])


def test_no_famous_athlete_names(basketball, book):
    for r in run_pipeline(1000, seed=5, config=basketball):
        assert plain(f'{r["first_name"]} {r["last_name"]}') not in book.famous


def test_initials_are_real_ones(basketball, book):
    allowed = set(basketball.names.nicknames.initials)
    for r in named(basketball, "Black", n=4000, seed=6):
        short = r["first_name"].replace(".", "")
        if len(short) == 2 and short.isupper():
            assert short in allowed and short[0] == r["legal_first"][0].upper()


def test_never_use_list_is_honored(basketball):
    cfg = basketball.model_copy(update={"names": basketball.names.model_copy(update={"never_use": ["Jayden", "Smith"]})})
    for r in named(cfg, "Black", n=3000, seed=7):
        assert r["legal_first"] != "Jayden" and r["last_name"] != "Smith"


# ---- names fit the player --------------------------------------------------------
@pytest.mark.parametrize("ethnicity", ["Black", "White", "Hispanic"])
def test_first_names_fit_the_group(basketball, book, ethnicity):
    for r in named(basketball, ethnicity, n=3000, seed=8):
        if r["heritage"] is None:
            assert book.fits(r["legal_first"], ethnicity), r["legal_first"]


def test_creative_names_show_up_and_follow_the_setting(basketball, book):
    def creative_share(level):
        out = [r for r in named(basketball, "Black", n=4000, seed=9, distinctive=level) if r["heritage"] is None]
        return sum(book.black_share(r["legal_first"].replace("'", "").capitalize()) >= 0.6 for r in out) / len(out)
    low, normal, high = creative_share("low"), creative_share("normal"), creative_share("high")
    assert low < normal < high
    assert normal > 0.2


def test_styled_names_look_right(basketball):
    styled = [r["legal_first"] for r in named(basketball, "Black", n=6000, seed=10)
              if "'" in r["legal_first"] or re.search(r"[a-z][A-Z]", r["legal_first"])]
    assert styled                                           # DeAndre, Ja'Marr... do appear
    for name in styled:
        assert re.fullmatch(r"[A-Z][a-z]*'?[A-Z][a-z]+", name), name


# ---- heritage families -----------------------------------------------------------
def test_common_white_surnames_are_not_treated_as_african(book):
    for last in ("Wade", "Barry", "Kane"):
        assert book.heritage_for("Black", last) is None
    assert book.heritage_for("Black", "Diallo") == "francophone_west_african"
    assert book.heritage_for("Black", "Okafor") == "west_african"
    assert book.heritage_for("Black", "Jean-Baptiste") == "haitian"


def test_heritage_families_get_no_jr(basketball):
    allows = {n for n, h in basketball.names.heritage.items() if h.suffixes}
    for r in named(basketball, "Black", n=20_000, seed=11):
        if r["heritage"] and r["heritage"] not in allows:
            assert r["suffix"] == "", f'{r["first_name"]} {r["last_name"]} {r["suffix"]}'


def test_heritage_families_often_get_heritage_first_names(basketball, book):
    out = [r for r in named(basketball, "Black", n=30_000, seed=12) if r["heritage"] == "francophone_west_african"]
    assert len(out) > 10
    share = sum(r["legal_first"] in book.heritage_first["francophone_west_african"] for r in out) / len(out)
    assert share > 0.5


def test_east_asian_surnames_never_get_south_asian_first_names(basketball, book):
    south = book.heritage_first["south_asian"] | book.heritage_first["south_asian_muslim"]
    for r in named(basketball, "Asian", n=6000, seed=13):
        if r["heritage"] == "east_asian":
            assert r["legal_first"] not in south, f'{r["legal_first"]} {r["last_name"]}'


def test_south_asian_surnames_mostly_get_south_asian_first_names(basketball, book):
    out = [r for r in named(basketball, "Asian", n=6000, seed=14) if r["heritage"] == "south_asian"]
    share = sum(r["legal_first"] in book.heritage_first["south_asian"] for r in out) / len(out)
    assert share > 0.6


def test_census_glued_surnames_are_fixed(book):
    black_lasts = set(book.last_tables["black"].keys)
    assert "Jean-Baptiste" in black_lasts and "Jeanbaptiste" not in black_lasts
    assert "St. Louis" in black_lasts or "Saint-Louis" in black_lasts


# ---- Polynesian and international ------------------------------------------------
def test_polynesian_players_get_polynesian_last_names(basketball):
    pool = set(basketball.names.pools["polynesian"].last)
    out = named(basketball, "Polynesian", n=2000, seed=15, origin="HI")
    assert sum(r["last_name"] in pool for r in out) / len(out) > 0.9


@pytest.mark.parametrize("country, ethnicity, pool", [("NGA", "Black", "nigeria"), ("SRB", "White", "serbia"),
                                                      ("SSD", "Black", "south_sudan")])
def test_international_players_use_their_country_names(basketball, country, ethnicity, pool):
    lasts = {to_ascii(n) for n in basketball.names.pools[pool].last}   # Stanković -> Stankovic
    for r in named(basketball, ethnicity, n=300, seed=16, origin=country, international=True):
        assert r["last_name"] in lasts
