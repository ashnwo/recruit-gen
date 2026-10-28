"""Realistic names (Piece 5).

US-born players
  first names  SSA baby names for boys born 2005-2010, split by group
  last names   2010 US Census surnames, split by group
  Both must "fit" the player's group (see fit in names.yaml), so no Viggo from
  Atlanta or Gottlieb as a Hispanic surname.

The last name is drawn FIRST, because it tells us the most about the family:
  - heritage surnames (Okafor, Diallo, Jean-Baptiste, Patel, Khan, Nguyen...)
    get a first name that fits that heritage, and no Jr./II unless allowed
  - everyone else gets a first name from his group's table; Black players'
    names can get creative styling (DeAndre, Ja'Marr)

Polynesian and international players use the hand-built pools in names.yaml.
Extras: Jr./II/III, "Trey" for IIIs, initials (CJ, DJ), no repeats, and no
names that match famous athletes.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter

from recruit_gen.models import NamesConfig
from recruit_gen.sampling import WeightedPicker

GROUP_KEY = {"White": "white", "Black": "black", "Hispanic": "hispanic", "Asian": "asian"}
US_POOL = {"Black": "us_black", "White": "us_white", "Hispanic": "us_hispanic",
           "Asian": "us_asian", "Polynesian": "polynesian"}
CROSS_SOURCES = {"white": 0.45, "black": 0.35, "hispanic": 0.15, "asian": 0.05}
_SPECIAL = str.maketrans({"ß": "ss", "Đ": "Dj", "đ": "dj", "ı": "i", "ø": "o", "Ø": "O",
                          "ł": "l", "Ł": "L", "æ": "ae", "Æ": "Ae"})


def to_ascii(text: str) -> str:
    """Game editors want plain letters: Đorđević -> Djordjevic."""
    text = text.translate(_SPECIAL)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def plain(name: str) -> str:
    """Lowercase, no apostrophes/periods/spaces: for comparing names."""
    return "".join(c for c in name.lower() if c.isalnum())


class NameBook:
    """Loads the name tables once and builds every weighted list the names stage needs."""

    def __init__(self, cfg: NamesConfig, distinctive: str = "normal"):
        self.cfg = cfg
        self.never = {plain(n) for n in cfg.never_use}
        self.famous = {plain(n) for n in cfg.famous_names}
        self.boost = cfg.distinctive.distinctive_boost.get(distinctive, 1.0)
        self._load_first_names()
        self._load_surnames()
        self._index_heritage()
        self._cache: dict[tuple, WeightedPicker] = {}

    # ------------------------------------------------------------------ loading
    def _load_first_names(self) -> None:
        self.first_rows: dict[str, dict[str, float]] = {}
        with open(self.cfg.folder / "first_names.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if plain(row["name"]) in self.never:
                    continue
                self.first_rows[row["name"]] = {k: float(v) for k, v in row.items() if k != "name"}

    def _rejoin(self, name: str) -> str:
        """Census glues Jean-Baptiste into Jeanbaptiste and St. Louis into Stlouis. Fix both."""
        for prefix, sep in self.cfg.rejoin_prefixes.items():
            rest = name[len(prefix):]
            if name.startswith(prefix) and rest.capitalize() in self.cfg.rejoin_remainders:
                return f"{prefix}{sep}{rest.capitalize()}"
        return name

    def _load_surnames(self) -> None:
        fit, sharpen = self.cfg.fit, self.cfg.sharpen
        weights = {g: {} for g in GROUP_KEY.values()}
        self.white_share: dict[str, float] = {}
        with open(self.cfg.folder / "surnames.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if plain(row["name"]) in self.never:
                    continue
                name, count = self._rejoin(row["name"]), float(row["count"])
                self.white_share[plain(name)] = float(row["white"]) / count if count else 0.0
                for g in weights:
                    w = float(row[g])
                    if w >= fit.surname_min_group_people or (count and w / count >= fit.surname_min_group_share):
                        weights[g][name] = w * min(1.0, w / count) ** sharpen[g]
        self.last_tables = {g: WeightedPicker(w) for g, w in weights.items()}

    def _index_heritage(self) -> None:
        """surname -> heritage, per group; plus which first names each group reserves."""
        self.heritage_of: dict[tuple[str, str], str] = {}     # (group, plain surname) -> heritage
        self.prefixes: list[tuple[str, str, str]] = []       # (group, prefix, heritage)
        self.patterns: list[tuple[str, re.Pattern, str]] = []
        self.default_heritage: dict[str, str] = {}
        self.heritage_first: dict[str, set[str]] = {}
        self.reserved: dict[str, dict[str, set[str]]] = {}   # group -> heritage -> names
        listed = {(g, to_ascii(f)) for h in self.cfg.heritage.values() for g in h.groups
                  for f in list(h.first) + [x for p in h.pools for x in self.cfg.pools[p].first]}
        pool_firsts = {to_ascii(f) for p in self.cfg.pools.values() for f in p.first}   # Kekoa, Sione...
        for h in self.cfg.heritage.values():                 # names matching a heritage's patterns
            pats = [re.compile(p, re.IGNORECASE) for p in h.first_patterns]
            if pats:
                listed |= {(g, n) for g in h.groups for n in self.first_rows if any(p.search(n) for p in pats)}
        for name, h in self.cfg.heritage.items():
            surnames = list(h.surnames) + [s for p in h.pools for s in self.cfg.pools[p].last]
            firsts = {to_ascii(f) for f in set(h.first) | {f for p in h.pools for f in self.cfg.pools[p].first}}
            pats = [re.compile(p, re.IGNORECASE) for p in h.first_patterns]
            for group in h.groups:
                g = GROUP_KEY.get(group)
                if pats and g:
                    firsts |= {n for n in self.first_rows if any(p.search(n) for p in pats) and self.fits(n, group)}
                if h.auto_first_min_share is not None and g:
                    # e.g. every first name that's 80%+ Asian and not on another heritage's list
                    firsts |= {n for n in self.first_rows if (group, n) not in listed
                               and n not in pool_firsts and self.share(n, g) >= h.auto_first_min_share}
                for s in surnames:
                    key = plain(to_ascii(s))
                    if h.max_white_share is not None and self.white_share.get(key, 0.0) > h.max_white_share:
                        continue                     # Wade, Barry, Kane: common white names too
                    self.heritage_of[(group, key)] = name
                for p in h.surname_prefixes:
                    self.prefixes.append((group, p.lower(), name))
                for pat in h.surname_patterns:
                    self.patterns.append((group, re.compile(pat, re.IGNORECASE), name))
                if h.reserve_first:
                    self.reserved.setdefault(group, {})[name] = firsts
            self.heritage_first[name] = firsts
            if h.default_for:
                self.default_heritage[h.default_for] = name

    def share(self, name: str, g: str) -> float:
        """Share of all boys with this first name who are in group key g (white/black/...)."""
        row = self.first_rows.get(name)
        return row[g] / row["births"] if row and row["births"] else 0.0

    # ------------------------------------------------------------------ fit + tables
    def fits(self, first: str, group: str) -> bool:
        """Is this first name believable for a boy from this group (Black, White...)?"""
        row = self.first_rows.get(first) or self.first_rows.get(first.replace("'", "").capitalize())
        if row is None or group not in GROUP_KEY:
            return False
        own, births = row[GROUP_KEY[group]], row["births"]
        f = self.cfg.fit
        return own >= f.min_group_births or (births > 0 and own / births >= f.min_group_share)

    def _first_weight(self, name: str, g: str, boost: bool) -> float:
        row = self.first_rows[name]
        w = row[g] * ((row[g] / row["births"]) ** self.cfg.sharpen[g] if row["births"] else 0.0)
        if boost and g == "black" and self.black_share(name) >= self.cfg.distinctive.distinctive_share:
            w *= self.boost
        return w

    def black_share(self, name: str) -> float:
        return self.share(name, "black")

    def _reserved_except(self, group: str, keep: str | None) -> set[str]:
        return {n for h, names in self.reserved.get(group, {}).items() if h != keep for n in names}

    def table(self, kind: str, group: str, extra: str = "") -> WeightedPicker:
        """Build (once) and return a weighted list of first names.
        kind: default    - the group's normal names (creative names boosted for Black players)
              mainstream - no creative-spelling names (for heritage families)
              cross      - names from group `extra` that still fit `group`
              heritage   - heritage `extra`'s own first names
              popular    - common US names only (Polynesian players), from table `extra`"""
        key = (kind, group, extra)
        if key in self._cache:
            return self._cache[key]
        g = GROUP_KEY.get(group)
        weights: dict[str, float] = {}
        if kind == "heritage":
            # Table weight, but squeezed between 30 and 300 so a hugely common name
            # (Samuel, Emmanuel) can't crowd out the rest of the list.
            for name in self.heritage_first[extra]:
                in_table = name in self.first_rows and g
                w = self._first_weight(name, g, False) if in_table else 30.0
                weights[name] = min(max(w, 30.0), 300.0)
        elif kind == "popular":
            for name, row in self.first_rows.items():
                if row["births"] >= self.cfg.fit.polynesian_us_min_births:
                    weights[name] = self._first_weight(name, extra, False)
        else:
            blocked = self._reserved_except(group, extra if kind == "mainstream" else None)
            source = extra if kind == "cross" else g
            for name in self.first_rows:
                if name in blocked or not self.fits(name, group):
                    continue
                if kind == "cross" and not self.fits(name, next(k for k, v in GROUP_KEY.items() if v == source)):
                    continue
                if kind == "mainstream":
                    h = self.cfg.heritage[extra]
                    if self.share(name, g) >= h.mainstream_max_share or self.first_rows[name]["births"] < h.mainstream_min_births:
                        continue
                    if h.mainstream_max_hispanic is not None and self.share(name, "hispanic") >= h.mainstream_max_hispanic:
                        continue
                weights[name] = self._first_weight(name, source, kind == "default")
        self._cache[key] = WeightedPicker(weights)
        return self._cache[key]

    def heritage_for(self, group: str, last: str) -> str | None:
        found = self.heritage_of.get((group, plain(last)))
        if found:
            return found
        low = last.lower()
        for g, prefix, name in self.prefixes:
            if g == group and low.startswith(prefix):
                return name
        for g, pattern, name in self.patterns:
            if g == group and pattern.search(low):
                return name
        return self.default_heritage.get(group)

    # ------------------------------------------------------------------ styling + extras
    def style(self, rng, name: str) -> str:
        """Deandre -> DeAndre or De'Andre (only for creative names that fit the pattern)."""
        d = self.cfg.distinctive
        if self.black_share(name) < d.distinctive_share or rng.random() >= d.style_rate:
            return name
        low = name.lower()
        for p in sorted(d.prefixes + ["D"], key=len, reverse=True):
            rest = low[len(p):]
            if low.startswith(p.lower()) and rest in d.roots and rest[:1] != p[-1:].lower():
                if len(p) == 1 or rng.random() >= d.camel_share:
                    return f"{p}'{rest.capitalize()}"
                return f"{p}{rest.capitalize()}"
        return name

    def suffix(self, rng, rates_key: str) -> str:
        rates = self.cfg.suffix_rates.get(rates_key, self.cfg.suffix_rates["default"])
        x, acc = rng.random(), 0.0
        for suf, p in rates.items():
            acc += p
            if x < acc:
                return suf
        return ""

    def initials(self, rng, first: str) -> str | None:
        n = self.cfg.nicknames
        options = {k: v for k, v in n.initials.items() if k[0] == first[0].upper()}
        if not options:
            return None
        pair = WeightedPicker(options).pick(rng)
        return f"{pair[0]}.{pair[1]}." if rng.random() < n.dotted_share else pair

    # ------------------------------------------------------------------ one name
    def draw(self, rng, pool: str, group: str) -> dict:
        """One raw name (no repeat checks). Returns first, legal_first, last, suffix, heritage."""
        cfg = self.cfg
        heritage, styled_ok, suffix_key = None, False, "default"
        if pool in ("us_black", "us_white", "us_hispanic", "us_asian"):
            g = pool[3:]
            last = self.last_tables[g].pick(rng)
            heritage = self.heritage_for(group, last)
            suffix_key = g
            if heritage:
                h = cfg.heritage[heritage]
                pick_heritage = rng.random() < h.heritage_first_share and self.table("heritage", group, heritage)
                first = (self.table("heritage", group, heritage) if pick_heritage
                         else self.table("mainstream", group, heritage)).pick(rng)
                if not h.suffixes:
                    suffix_key = None
            else:
                if rng.random() < cfg.fit.cross_mix:
                    src = WeightedPicker({k: v for k, v in CROSS_SOURCES.items() if k != g}).pick(rng)
                    cross = self.table("cross", group, src)
                    first = cross.pick(rng) if cross else self.table("default", group).pick(rng)
                else:
                    first = self.table("default", group).pick(rng)
                styled_ok = g == "black"
        elif pool == "polynesian":
            poly = cfg.pools["polynesian"]
            x = rng.random()
            if x < 0.5:
                first = poly.first[int(rng.integers(len(poly.first)))]
            else:
                first = self.table("popular", group, "white" if x < 0.85 else "black").pick(rng)
            last = (poly.last[int(rng.integers(len(poly.last)))] if rng.random() < 0.95
                    else self.last_tables["white"].pick(rng))
            suffix_key = "polynesian"
        else:
            p = cfg.pools[pool]
            first = p.first[int(rng.integers(len(p.first)))]
            last = p.last[int(rng.integers(len(p.last)))]
        legal = self.style(rng, first) if styled_ok else first
        legal, last = to_ascii(legal), to_ascii(last)
        suffix = self.suffix(rng, suffix_key) if suffix_key else ""
        goes_by = legal
        if suffix == "III" and rng.random() < cfg.nicknames.trey_rate:
            goes_by = "Trey"
        elif rng.random() < cfg.nicknames.initials_rate:
            goes_by = self.initials(rng, legal) or legal
        return {"first": goes_by, "legal_first": legal, "last": last, "suffix": suffix, "heritage": heritage}


_BOOKS: dict[tuple, NameBook] = {}


def name_book(cfg: NamesConfig, distinctive: str = "normal") -> NameBook:
    """Loading the tables takes about a second, so keep one NameBook per settings."""
    key = (cfg.model_dump_json(), distinctive)
    if key not in _BOOKS:
        _BOOKS[key] = NameBook(cfg, distinctive)
    return _BOOKS[key]


def pool_for(r: dict, places, rng) -> str:
    """Which name pool a player draws from: his country's pools if he's international,
    otherwise the US table for his group."""
    if r["international"]:
        pools = places.countries[r["origin"]].name_pools.get(r["ethnicity"])
        if pools:
            return WeightedPicker(pools).pick(rng)
    return US_POOL[r["ethnicity"]]


def repeat_limits(count: int) -> tuple[int, int]:
    """How many times one first / last name may appear in a class of this size."""
    return 1 + count // 45, 1 + count // 30


def name_class(recruits: list[dict], rng, book: NameBook, places) -> None:
    """Give every recruit a unique, believable name."""
    max_first, max_last = repeat_limits(len(recruits))
    used_first, used_last, used_full = Counter(), Counter(), set()
    for r in recruits:
        pool = pool_for(r, places, rng)
        for attempt in range(80):
            n = book.draw(rng, pool, r["ethnicity"])
            relax = attempt >= 50          # struggling (tiny pools)? loosen the repeat limits
            full = plain(f"{n['first']} {n['last']}")
            if plain(n["legal_first"]) == plain(n["last"]):
                continue                   # "Yusuf Yusuf" happens, but reads like a bug
            if not relax and (used_first[n["legal_first"]] >= max_first or used_last[n["last"]] >= max_last):
                continue
            if full in used_full or full in book.famous or plain(f"{n['legal_first']} {n['last']}") in book.famous:
                continue
            break
        else:
            raise RuntimeError(f"Couldn't find a unique name for recruit #{r['rank']}")
        used_first[n["legal_first"]] += 1
        used_last[n["last"]] += 1
        used_full.add(full)
        r["first_name"], r["last_name"], r["suffix"] = n["first"], n["last"], n["suffix"]
        r["legal_first"], r["name_pool"], r["heritage"] = n["legal_first"], pool, n["heritage"]
