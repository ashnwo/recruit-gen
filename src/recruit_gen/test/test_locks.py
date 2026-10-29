"""Piece 8: custom requests - lock any field, the rest fills in to match."""
from collections import Counter
from pathlib import Path

import pytest

from recruit_gen.checks import realism_problems
from recruit_gen.cli import default_out, find_config, summary_table, wizard
from recruit_gen.locks import (
    LockError, Locks, Spec, build_locks, check_request, parse_height, parse_stars, read_spec_file,
)
from recruit_gen.names import plain
from recruit_gen.pipeline import build_board, run_pipeline


def make(basketball, n=300, seed=1, top=False, **fields):
    return run_pipeline(n, seed, basketball, top=top, locks=build_locks(basketball, **fields))


# ---- reading what you type --------------------------------------------------------
@pytest.mark.parametrize("text, inches", [("6-5", 77), ("6'5\"", 77), ("6 5", 77), ("77", 77), ("7-0", 84)])
def test_heights(text, inches):
    assert parse_height(text) == inches


@pytest.mark.parametrize("text, stars", [("5", (5,)), ("4-5", (4, 5)), ("4+", (4, 5)), ("3,4", (3, 4)), ("5 stars", (5,))])
def test_stars(text, stars):
    assert parse_stars(text) == stars


def test_words_people_type(basketball):
    lk = build_locks(basketball, position="guard", ethnicity="Caucasian", origin="texas",
                     archetype="rim prot", development="late")
    assert lk.position == ("PG", "SG") and lk.ethnicity == "White" and lk.origin == "TX"
    assert lk.archetype == "Rim Protector" and lk.development == "Late Bloomer"
    assert build_locks(basketball, origin="Nigeria").origin == "NGA"


@pytest.mark.parametrize("field, value, hint", [
    ("position", "QB", "PG"), ("ethnicity", "martian", "black"), ("origin", "Atlantis", "TX"),
    ("projection", "top ten", "Lottery"), ("archetype", "Wizard", "Rim Protector"), ("height", "tall", "6-5"),
])
def test_bad_input_explains_itself(basketball, field, value, hint):
    with pytest.raises(LockError, match=hint):
        build_locks(basketball, **{field: value})


def test_bad_stars_explain_themselves(football):
    with pytest.raises(LockError, match="4-5"):
        build_locks(football, stars="seven")


# ---- requests that can't be met ---------------------------------------------------
def test_style_that_doesnt_fit_the_position(basketball):
    with pytest.raises(LockError, match="Rim Protector"):
        check_request(Spec(1, build_locks(basketball, position="PG", archetype="Rim Protector")), basketball)


def test_impossible_combination(basketball):
    with pytest.raises(LockError, match="No realistic player"):
        check_request(Spec(1, build_locks(basketball, origin="Serbia", ethnicity="polynesian")), basketball)


def test_more_five_stars_than_exist(football):
    with pytest.raises(LockError, match="Only 32 5-star"):
        check_request(Spec(40, build_locks(football, stars="5")), football)


def test_more_lottery_picks_than_exist(basketball):
    with pytest.raises(LockError, match="Only 14 Lottery"):
        check_request(Spec(20, build_locks(basketball, projection="lottery")), basketball)


def test_draft_sports_refuse_star_locks(basketball):
    with pytest.raises(LockError, match="--projection"):
        build_locks(basketball, stars="5")


def test_full_name_locked_means_one_player(basketball):
    with pytest.raises(LockError, match="only 1 player"):
        check_request(Spec(3, build_locks(basketball, first="Jaylen", last="Okafor")), basketball)


def test_unusual_lock_gets_a_warning_not_a_refusal(basketball):
    warnings = check_request(Spec(1, build_locks(basketball, position="C", height="5-10")), basketball)
    assert warnings and "unusual" in warnings[0]
    r = make(basketball, n=1, position="C", height="5-10")[0]
    assert r["height"] == 70 and r["position"] == "C"          # you asked, you got him
    assert realism_problems({**r, "locks": build_locks(basketball, position="C", height="5-10")}, basketball) == []


# ---- every lock is honored ----------------------------------------------------------
@pytest.mark.parametrize("fields, check", [
    ({"position": "guard"}, lambda r: r["position"] in ("PG", "SG")),
    ({"ethnicity": "white"}, lambda r: r["ethnicity"] == "White"),
    ({"origin": "GA"}, lambda r: r["origin"] == "GA" and r["hometown"].endswith(", GA")),
    ({"origin": "Nigeria"}, lambda r: r["origin"] == "NGA" and r["hometown"].endswith("Nigeria")),
    ({"projection": "1st,2nd"}, lambda r: r["projection"] in ("1st round", "2nd round")),
    ({"archetype": "Point Forward"}, lambda r: r["archetype"] == "Point Forward" and r["position"] in ("SF", "PF")),
    ({"development": "Raw Project"}, lambda r: r["development"] == "Raw Project"),
    ({"height": "6-9"}, lambda r: r["height"] == 81),
    ({"weight": "240"}, lambda r: r["weight"] == 240),
    ({"last": "Okafor"}, lambda r: r["last_name"] == "Okafor"),
])
def test_locks_are_honored(basketball, fields, check):
    recruits = make(basketball, n=40, seed=2, **fields)
    assert len(recruits) == 40 and all(check(r) for r in recruits)


def test_locks_combine(basketball, book):
    for r in make(basketball, n=50, seed=3, position="guard", ethnicity="white", origin="TX"):
        assert r["position"] in ("PG", "SG") and r["ethnicity"] == "White" and r["origin"] == "TX"
        assert book.fits(r["first_name"].replace("'", ""), "White") or len(r["first_name"].replace(".", "")) == 2


def test_exact_name(basketball):
    r = make(basketball, n=1, seed=4, first="Jaylen", last="Okafor", ethnicity="black")[0]
    assert (r["first_name"], r["last_name"], r["suffix"]) == ("Jaylen", "Okafor", "")


# ---- locks shape the rest realistically ---------------------------------------------
def test_locking_white_makes_bigs_more_likely(basketball):
    white = Counter(r["position"] for r in make(basketball, n=1000, seed=5, ethnicity="white"))
    anyone = Counter(r["position"] for r in run_pipeline(1000, 5, basketball))
    assert white["C"] + white["PF"] > anyone["C"] + anyone["PF"] + 60
    assert white["PG"] < anyone["PG"]


def test_locking_height_picks_fitting_positions(basketball):
    seven = Counter(r["position"] for r in make(basketball, n=300, seed=6, height="7-0"))
    assert seven["C"] > 150 and seven["PG"] == 0
    small = Counter(r["position"] for r in make(basketball, n=300, seed=6, height="6-0"))
    assert small["PG"] > 200 and small["C"] == 0


def test_locking_polynesian_picks_fitting_homes(basketball):
    homes = Counter(r["origin"] for r in make(basketball, n=300, seed=7, ethnicity="polynesian"))
    assert sum(homes[c] for c in ("HI", "UT", "CA", "AUS")) > 120


def test_top_centers_are_spread_through_the_rankings(basketball):
    ranks = [r["rank"] for r in make(basketball, n=5, seed=8, top=True, position="C")]
    assert ranks == sorted(ranks) and ranks[-1] > 5        # not simply #1-#5
    assert ranks[0] <= 25                                   # but the best center is near the top


def test_star_lock_in_top_mode(football):
    locks = build_locks(football, stars="4")
    ranks = [r["rank"] for r in run_pipeline(10, 9, football, top=True, locks=locks)]
    assert ranks == list(range(33, 43))                     # the best 4-stars are #33-#42


def test_projection_lock_in_top_mode(basketball):
    ranks = [r["rank"] for r in make(basketball, n=10, seed=9, top=True, projection="1st")]
    assert ranks == list(range(15, 25))                     # the best non-lottery 1st-rounders: #15-#24


def test_same_seed_same_custom_class(basketball):
    a = make(basketball, n=20, seed=10, position="big", origin="CA")
    b = make(basketball, n=20, seed=10, position="big", origin="CA")
    assert a == b


# ---- boards: many requests in one class --------------------------------------------
def test_board_file(basketball):
    specs = read_spec_file(find_config("basketball").parent.parent / "examples" / "board.csv", basketball)
    result = build_board(specs, seed=11, config=basketball)
    recruits = result.recruits
    assert len(recruits) == sum(s.count for s in specs)
    assert len({r["rank"] for r in recruits}) == len(recruits)
    names = [plain(f'{r["first_name"]} {r["last_name"]}') for r in recruits]
    assert len(names) == len(set(names))
    assert any(r["first_name"] == "Jaylen" and r["last_name"] == "Okafor" for r in recruits)
    assert any(r["position"] == "C" and r["height"] == 86 and r["archetype"] == "Rim Protector" for r in recruits)
    assert sum(r["origin"] == "NGA" for r in recruits) >= 1


def test_bad_board_row_names_the_line(basketball, tmp_path):
    f = tmp_path / "board.csv"
    f.write_text("count,position,projection\n1,PG,lottery\n2,QB,1st\n")
    with pytest.raises(LockError, match="line 3"):
        read_spec_file(f, basketball)


# ---- command line helpers ------------------------------------------------------------
def test_default_output_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_out("basketball", 80, 42, False, False) == tmp_path / "output" / "basketball_80_seed42.csv"
    assert default_out("basketball", 25, 7, True, False).name == "basketball_25_top_seed7.csv"
    assert default_out("basketball", 3, 9, False, True).name == "basketball_3_custom_seed9.csv"


def test_summary_table(basketball):
    table = summary_table(run_pipeline(20, 12, basketball), show=5)
    lines = table.split("\n")
    assert len(lines) == 2 + 5 + 1 + 1                    # header, rule, 5 rows, "more", star note
    assert "more in the CSV" in lines[-2] and "high school" in lines[-1]
    assert "Proj" in lines[0] and "Team" in lines[0]


def test_wizard_turns_answers_into_options():
    answers = iter(["basketball", "guard", "white", "TX", "lottery", "3", "", "77"])
    a = wizard(ask=lambda text, default="": next(answers) or default)
    assert a == {"sport": "basketball", "position": "guard", "ethnicity": "white", "origin": "TX",
                 "projection": "lottery", "count": "3", "top": True, "seed": "77"}   # draft: top by default
    answers = iter(["football", "QB", "", "", "5", "", "", ""])
    a = wizard(ask=lambda text, default="": next(answers) or default)
    assert a["stars"] == "5" and a["count"] == "1" and a["top"] is False
