from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

Ethnicity = Literal["Black", "White", "Hispanic", "Asian", "Polynesian"]
ETHNICITIES: tuple[str, ...] = Ethnicity.__args__
Role = Literal["core", "major", "minor", "low", "none"]   # how important a skill group is at a position
Weight = Annotated[float, Field(gt=0)]          # a relative weight: any positive number
Share = Annotated[float, Field(ge=0, le=1)]     # a fraction from 0 to 1


def check_adds_to_one(mix: dict, what: str) -> dict:
    total = sum(mix.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"{what} adds up to {total:.3f}; it must add up to 1")
    return mix


Rating = Annotated[int, Field(ge=1, le=99)]


class Recruit(BaseModel):
    """The contract every finished recruit must meet.
    Field order = CSV column order (the 2K ratings get flattened onto the end)."""
    rank: int = Field(ge=1)              # national rank (scouted); for draft sports, the big board
    projection: str = ""                 # draft sports: Lottery / 1st round / 2nd round / Undrafted
    projected_pick: Optional[int] = Field(default=None, ge=1)   # draft sports: projected pick number
    stars: Optional[int] = Field(default=None, ge=1, le=5)      # draft sports: high school stars (background)
    rating: Optional[float] = Field(default=None, ge=0, le=101) # 247-style rating, e.g. 98.6
    first_name: str = Field(min_length=1)   # the name he goes by (CJ, Trey, DeAndre...)
    last_name: str = Field(min_length=1)
    suffix: str                          # "", Jr., II, III, IV
    position: str
    secondary_position: str              # "" if none
    pos_rank: int = Field(ge=1)          # rank among this class's players at his position
    team: str = ""                       # where he plays now (draft sports): Duke, Real Madrid...
    level: str = ""                      # College / Overseas pro / NBL Next Stars / Overtime Elite / G League
    class_year: str = ""                 # college players: Fr / So / Jr / Sr
    age: Optional[int] = Field(default=None, ge=14, le=30)
    archetype: str
    ovr: Rating                          # current overall
    potential: Rating                    # "Potential": his ceiling
    dev_trait: str = ""                  # Normal / Impact / Star / Elite (sports that use them)
    development: str                     # Early Bloomer / Steady Climber / Late Bloomer / Raw Project
    growth: str                          # Low / Medium / High / Very High
    height: int = Field(ge=60, le=96)    # inches; outside 5'0"-8'0" = something broke
    height_ft: str                       # same height written like 6'5"
    weight: int = Field(ge=100, le=400)  # lbs; outside this = something broke
    wingspan: Optional[float] = Field(default=None, ge=60, le=104)   # inches
    wingspan_ft: str = ""
    forty: Optional[float] = Field(default=None, ge=4.0, le=7.0)   # 40-yard dash (sports that use it)
    ethnicity: Ethnicity
    hometown: str                        # "Atlanta, GA" or "Lagos, Nigeria"
    origin: str                          # state code (GA) or country code (NGA)
    state_rank: int = Field(ge=1)        # rank among this class's players from the same state/country
    high_school: str
    true_rank: int = Field(ge=1)         # where his REAL talent would rank him (hidden from scouts)
    verdict: str                         # Steal / Underrated / As Advertised / Overrated / Bust
    ratings: dict[str, Rating]           # 2K attributes, e.g. {"Three-Point Shot": 74, ...}

    @model_validator(mode="after")
    def ceiling_not_below_current(self):
        if self.potential < self.ovr:
            raise ValueError(f"potential {self.potential} is below ovr {self.ovr}")
        return self


# ---- height distributions (Days 8-14) ------------------------------------------
class NormalParams(BaseModel):
    loc: float
    scale: float = Field(gt=0)
    distribution: Literal["normal"]


class SkewNormalParams(BaseModel):
    loc: float
    scale: float = Field(gt=0)
    distribution: Literal["skewnormal"]
    a: float


PhysicalsConfig = Annotated[
    NormalParams | SkewNormalParams,
    Field(discriminator="distribution"),
]

# PhysicalsConfig is a *label* for "one of these two classes", not a class itself,
# so you can't call PhysicalsConfig(...). A TypeAdapter is the tool that reads the
# "distribution" field and builds the right class.
_physicals_adapter = TypeAdapter(PhysicalsConfig)


def parse_physicals_config(data: dict) -> NormalParams | SkewNormalParams:
    """Plain dict -> NormalParams or SkewNormalParams. Raises ValidationError if wrong."""
    return _physicals_adapter.validate_python(data)


# ---- per-position bodies (Piece 2) ---------------------------------------------
class WeightParams(BaseModel):
    """Weight moves with height: center = at_ref + lbs_per_inch * (height - ref_height)."""
    ref_height: float = Field(gt=0)
    at_ref: float = Field(gt=0)
    lbs_per_inch: float = Field(ge=0)
    sd: float = Field(gt=0)


class OriginOverride(BaseModel):
    """rate of players at this position come from these places instead (Australian punters)."""
    rate: Share
    states: dict[str, Weight] = {}
    countries: dict[str, Weight] = {}


class PositionConfig(BaseModel):
    share: float = Field(gt=0, le=1)
    height: PhysicalsConfig
    weight: WeightParams
    ethnicity: dict[Ethnicity, Share]    # mix for US-born players at this position
    secondary: dict[str, Share] = {}     # second position -> chance ("" = none)
    wingspan: Optional[tuple[float, Annotated[float, Field(ge=0)]]] = None   # [inches past height, spread]
    roles: dict[str, Role] = {}          # skill group -> role
    overrides: dict[str, Role] = {}      # single rating -> role
    body: dict[str, float] = {}          # body rating starting points
    ovr_body: dict[str, Annotated[float, Field(ge=0)]] = {}
    archetypes: dict[str, Weight] = {}
    elite_tilt: float = Field(default=0.0, le=0)   # below 0 = rarely ranked near the top
    origin_override: Optional["OriginOverride"] = None   # some positions come from special places

    @field_validator("ethnicity")
    @classmethod
    def ethnicity_adds_to_one(cls, mix):
        return check_adds_to_one(mix, "ethnicity mix")


# ---- places (Piece 4): configs/shared/places.yaml -------------------------------
class StateInfo(BaseModel):
    name: str
    cities: dict[str, Weight] = Field(min_length=1)
    eth_mult: dict[Ethnicity, Annotated[float, Field(ge=0)]] = {}   # nudges the ethnicity odds


class CountryInfo(BaseModel):
    name: str
    cities: dict[str, Weight] = Field(min_length=1)
    ethnicity: dict[Ethnicity, Share]
    name_pools: dict[Ethnicity, dict[str, Weight]] = {}   # group -> {pool: weight}

    @field_validator("ethnicity")
    @classmethod
    def ethnicity_adds_to_one(cls, mix):
        return check_adds_to_one(mix, "country ethnicity mix")


class SchoolPattern(BaseModel):
    text: str
    weight: Weight


class SchoolNames(BaseModel):
    patterns: list[SchoolPattern] = Field(min_length=1)
    international: str
    directions: list[str] = Field(min_length=1)
    namesakes: list[str] = Field(min_length=1)
    saints: list[str] = Field(min_length=1)


class PlacesData(BaseModel):
    default_eth_mult: dict[Ethnicity, Annotated[float, Field(ge=0)]] = {}   # for states that don't say
    states: dict[str, StateInfo] = Field(min_length=1)
    countries: dict[str, CountryInfo] = {}
    schools: SchoolNames


# ---- names (Piece 5): configs/shared/names/names.yaml ------------------------------
US_NAME_POOLS = {"us_white", "us_black", "us_hispanic", "us_asian"}


class FitRules(BaseModel):
    min_group_births: float = Field(ge=0)
    min_group_share: Share
    surname_min_group_people: float = Field(ge=0)
    surname_min_group_share: Share
    cross_mix: Share
    polynesian_us_min_births: float = Field(ge=0)


class DistinctiveRules(BaseModel):
    distinctive_share: Share
    distinctive_boost: dict[Literal["low", "normal", "high"], Annotated[float, Field(ge=0)]]
    style_rate: Share
    camel_share: Share
    prefixes: list[str]
    roots: list[str]


class NicknameRules(BaseModel):
    initials_rate: Share
    initials: dict[str, Weight]
    dotted_share: Share
    trey_rate: Share


class NamePool(BaseModel):
    first: list[str] = Field(min_length=1)
    last: list[str] = Field(min_length=1)


class Heritage(BaseModel):
    groups: list[Ethnicity] = Field(min_length=1)
    heritage_first_share: Share
    pools: list[str] = []                # also use these hand-built pools' names
    surnames: list[str] = []
    surname_prefixes: list[str] = []
    first: list[str] = []
    reserve_first: bool = False          # these first names are ONLY for this heritage
    suffixes: bool = False               # allow Jr./II/III
    default_for: Optional[Ethnicity] = None
    max_white_share: Optional[Share] = None        # skip listed surnames that are common white names
    mainstream_max_share: Share = 0.6              # "mainstream" first names stay below this group share
    mainstream_min_births: float = Field(default=1000, ge=0)   # ...and are at least this common
    mainstream_max_hispanic: Optional[Share] = None            # ...and aren't this Hispanic
    auto_first_min_share: Optional[Share] = None   # also count first names at least this typical
    surname_patterns: list[str] = []               # regexes for unlisted surnames
    first_patterns: list[str] = []                 # regexes for first names that belong here


class NamesConfig(BaseModel):
    folder: Path                         # where first_names.csv / surnames.csv live (set by the loader)
    fit: FitRules
    distinctive: DistinctiveRules
    sharpen: dict[Literal["white", "black", "hispanic", "asian"], Annotated[float, Field(ge=0)]]
    nicknames: NicknameRules
    suffix_rates: dict[str, dict[str, Share]]
    never_use: list[str] = []
    famous_names: list[str] = []
    rejoin_prefixes: dict[str, str] = {}   # prefix -> separator ("-" or ". ")
    rejoin_remainders: list[str] = []
    pools: dict[str, NamePool] = {}
    heritage: dict[str, Heritage] = {}

    @model_validator(mode="after")
    def heritage_pools_exist(self):
        for name, h in self.heritage.items():
            missing = [p for p in h.pools if p not in self.pools]
            if missing:
                raise ValueError(f"heritage '{name}' uses unknown pools: {', '.join(missing)}")
        if "default" not in self.suffix_rates:
            raise ValueError("suffix_rates needs a 'default' entry")
        return self


# ---- where a sport's players come from (Piece 4) ---------------------------------
class OriginsConfig(BaseModel):
    international_rate: Share
    states: dict[str, Weight] = Field(min_length=1)
    countries: dict[str, Weight] = {}

    @model_validator(mode="after")
    def countries_if_international(self):
        if self.international_rate > 0 and not self.countries:
            raise ValueError("international_rate is above 0 but no countries are listed")
        return self


class PrepConfig(BaseModel):
    rate: Share
    star_boost: dict[int, Annotated[float, Field(ge=0)]] = {}
    international_rate: Share
    schools: dict[str, Weight] = Field(min_length=1)


class StarBand(BaseModel):
    stars: int = Field(ge=1, le=5)
    last_rank: int = Field(ge=1)
    rating: tuple[float, float]          # (best, worst) rating inside this band

    @model_validator(mode="after")
    def best_rating_first(self):
        if self.rating[0] < self.rating[1]:
            raise ValueError(f"{self.stars}-star rating range must go best to worst, got {list(self.rating)}")
        return self


class VerdictCutoffs(BaseModel):
    """How many times better/worse the true rank must be than the scouted rank."""
    steal: float = Field(gt=1)
    underrated: float = Field(gt=1)
    overrated: float = Field(gt=1)
    bust: float = Field(gt=1)
    min_gap: int = Field(ge=0)        # and at least this many rank spots apart

    @model_validator(mode="after")
    def extremes_are_bigger(self):
        if self.steal < self.underrated or self.bust < self.overrated:
            raise ValueError("steal must be >= underrated and bust must be >= overrated")
        return self


class TalentConfig(BaseModel):
    international_error_mult: float = Field(default=1.0, ge=1)
    population: int = Field(gt=0)
    pool_size: int = Field(gt=0)
    scout_error: float = Field(ge=0)
    stars: list[StarBand] = Field(min_length=1)
    verdicts: VerdictCutoffs

    @model_validator(mode="after")
    def bands_make_sense(self):
        if self.pool_size > self.population:
            raise ValueError("pool_size can't be bigger than population")
        last = [b.last_rank for b in self.stars]
        if last != sorted(last) or len(set(last)) != len(last):
            raise ValueError(f"star bands must go best to worst with growing last_rank, got {last}")
        if last[-1] < self.pool_size:
            raise ValueError(f"the last star band must cover the whole pool (last_rank >= {self.pool_size})")
        return self


# ---- development, ratings, archetypes (Piece 6) ----------------------------------
class Trajectory(BaseModel):
    weight: Weight
    gap: float = Field(ge=0)             # current ability below ceiling, in talent units
    bias: float                          # scouting bias, in scouting errors (+ = overrated)
    noise: float = Field(ge=0)           # extra scouting uncertainty
    growth: str


class DevelopmentConfig(BaseModel):
    gap_spread: float = Field(ge=0)
    trajectories: dict[str, Trajectory] = Field(min_length=1)


class PotentialRules(BaseModel):
    center: float
    per_sd: float
    noise: float = Field(ge=0)
    min: int
    max: int


class RatingsConfig(BaseModel):
    default_role: Role = "low"           # role for skill groups a position doesn't list
    min: int = Field(ge=1)
    max: int = Field(le=99)
    levels: dict[Role, tuple[float, Annotated[float, Field(ge=0)]]]   # role -> [middle, spread]
    group_spread: float = Field(ge=0)
    attr_noise: float = Field(ge=0)
    ovr_weights: dict[Role, Annotated[float, Field(ge=0)]]
    potential: PotentialRules
    athletic_from_talent: float = Field(ge=0, lt=1)


class DevTraitsConfig(BaseModel):
    """Trait levels from TRUE talent (+ noise), checked top-down: first one he reaches."""
    noise: float = Field(ge=0)
    levels: dict[str, float] = Field(min_length=1)


class FortyConfig(BaseModel):
    """40-yard dash from a speed rating: at_95 + (95 - rating) x per_point (+ noise)."""
    rating: str
    at_95: float
    per_point: float
    noise: float = Field(ge=0)
    min: float
    max: float


class BodyLoad(BaseModel):
    athletic: float = 0.0
    height: float = 0.0
    weight: float = 0.0
    talent: float = 0.0


class AttributesConfig(BaseModel):
    order: list[str] = Field(min_length=1)
    groups: dict[str, list[str]] = Field(min_length=1)
    body: dict[str, BodyLoad] = {}
    fixed: dict[str, tuple[float, Annotated[float, Field(ge=0)]]] = {}

    @model_validator(mode="after")
    def every_attribute_has_one_home(self):
        homes = Counter([a for lst in self.groups.values() for a in lst] + list(self.body) + list(self.fixed))
        doubled = [a for a, n in homes.items() if n > 1]
        missing = [a for a in self.order if a not in homes]
        extra = [a for a in homes if a not in self.order]
        if doubled or missing or extra:
            raise ValueError(f"attributes don't line up: in two places {doubled}, "
                             f"not in any group {missing}, not in order {extra}")
        return self

    @property
    def group_of(self) -> dict[str, str]:
        return {a: g for g, lst in self.groups.items() for a in lst}


class ArchetypeConfig(BaseModel):
    mods: dict[str, float] = {}
    shift: dict[Literal["height", "weight", "athletic", "wingspan"], float] = {}


# ---- clean-up check (Piece 7) -----------------------------------------------------
Range = tuple[float, float]


class BodyBounds(BaseModel):
    height: Optional[Range] = None
    weight: Optional[Range] = None


class ChecksConfig(BaseModel):
    max_reject_share: Share
    always_allow: int = Field(default=2, ge=0)
    max_rebuilds: int = Field(ge=0)
    body: dict[str, BodyBounds] = {}
    wingspan_vs_height: Optional[Range] = None


# ---- draft prospects (Piece 11) ---------------------------------------------------
class ProjectionBand(BaseModel):
    label: str
    last_rank: int = Field(ge=1)


class HighSchoolHistory(BaseModel):
    shift: dict[str, float] = {}         # development type -> how his old HS ranking compared to now
    noise: float = Field(ge=0)


class DraftConfig(BaseModel):
    """Makes a sport's class a draft class: rank = big board, stars = high school background."""
    default_top: bool = True
    picks: int = Field(ge=1)
    projection: list[ProjectionBand] = Field(min_length=1)
    high_school: HighSchoolHistory

    @model_validator(mode="after")
    def bands_in_order(self):
        last = [b.last_rank for b in self.projection]
        if last != sorted(last) or len(set(last)) != len(last):
            raise ValueError(f"projection bands must go best to worst with growing last_rank, got {last}")
        return self


class ClassYear(BaseModel):
    share: Weight
    tilt: float = 0.0


class CollegeTier(BaseModel):
    share: Weight
    tilt: float = 0.0
    teams: list[str] = Field(min_length=1)


COLLEGE, OVERSEAS = "College", "Overseas pro"


class TeamsConfig(BaseModel):
    """Where draft prospects play now."""
    paths: dict[str, dict[str, Weight]] = Field(min_length=1)   # "US" / country / "abroad" -> {path: weight}
    college_tiers: dict[str, CollegeTier] = Field(min_length=1)
    class_years: dict[str, ClassYear] = Field(min_length=1)
    development_nudge: dict[str, dict[str, Annotated[float, Field(ge=0)]]] = {}
    college_age: dict[str, int] = Field(min_length=1)
    pro_ages: dict[int, Weight] = Field(min_length=1)
    pro_teams: dict[str, list[str]] = {}                 # path -> teams
    overseas_clubs: dict[str, list[str]] = {}            # country (or "other") -> clubs

    @model_validator(mode="after")
    def everything_has_a_home(self):
        problems = []
        for where, paths in self.paths.items():
            for path in paths:
                if path not in (COLLEGE, OVERSEAS) and path not in self.pro_teams:
                    problems.append(f"path '{path}' (in {where}) has no teams in pro_teams")
                if path == OVERSEAS and "other" not in self.overseas_clubs:
                    problems.append("overseas_clubs needs an 'other' list")
        if "US" not in self.paths or "abroad" not in self.paths:
            problems.append("paths needs both 'US' and 'abroad'")
        missing_age = [y for y in self.class_years if y not in self.college_age]
        if missing_age:
            problems.append(f"college_age is missing {missing_age}")
        for dev, nudges in self.development_nudge.items():
            problems += [f"development_nudge {dev} year '{y}'" for y in nudges if y not in self.class_years]
        if problems:
            raise ValueError("; ".join(problems))
        return self


class SportConfig(BaseModel):
    sport: str
    game: str = ""                       # which video game this sport's settings are built for
    positions: dict[str, PositionConfig] = Field(min_length=1)
    talent: TalentConfig
    origins: OriginsConfig
    prep: PrepConfig
    places: PlacesData                   # filled in from configs/shared/places.yaml by the loader
    names: Optional[NamesConfig] = None  # filled in from configs/shared/names/names.yaml by the loader
    development: Optional[DevelopmentConfig] = None
    ratings: Optional[RatingsConfig] = None
    attributes: Optional[AttributesConfig] = None
    archetypes: dict[str, ArchetypeConfig] = {}
    checks: Optional[ChecksConfig] = None
    position_groups: dict[str, list[str]] = {}   # words you can type for --position (guard, big...)
    dev_traits: Optional[DevTraitsConfig] = None
    forty: Optional[FortyConfig] = None
    draft: Optional[DraftConfig] = None          # draft sports (NBA prospects)
    teams: Optional[TeamsConfig] = None          # where draft prospects play now
    ages: Optional[dict[int, Weight]] = None     # ages for sports without teams (HS recruits)

    @model_validator(mode="after")
    def rating_names_line_up(self):
        """Catch typos: every attribute, group, archetype and position named anywhere must exist."""
        problems = []
        known_pos = set(self.positions) | {""}
        if self.checks is not None:
            problems += [f"checks body position '{x}'" for x in self.checks.body if x not in self.positions]
        for group, members in self.position_groups.items():
            problems += [f"position group {group} '{x}'" for x in members if x not in self.positions]
        for pos, p in self.positions.items():
            ov = p.origin_override
            if ov is not None:
                problems += [f"{pos} origin_override '{x}'" for x in ov.states if x not in self.places.states]
                problems += [f"{pos} origin_override '{x}'" for x in ov.countries if x not in self.places.countries]
        if self.forty is not None and self.attributes is not None and self.forty.rating not in self.attributes.order:
            problems.append(f"forty rating '{self.forty.rating}'")
        trajectories = set(self.development.trajectories) if self.development else set()
        if self.draft is not None:
            if self.draft.projection[-1].last_rank < self.talent.pool_size:
                problems.append(f"draft projection must cover the whole pool (last_rank >= {self.talent.pool_size})")
            problems += [f"draft high_school shift '{x}'" for x in self.draft.high_school.shift if x not in trajectories]
        if self.teams is not None:
            problems += [f"teams paths country '{x}'" for x in self.teams.paths
                         if x not in ("US", "abroad") and x not in self.places.countries]
            problems += [f"teams overseas_clubs country '{x}'" for x in self.teams.overseas_clubs
                         if x != "other" and x not in self.places.countries]
            problems += [f"teams development_nudge '{x}'" for x in self.teams.development_nudge if x not in trajectories]
        for pos, p in self.positions.items():
            problems += [f"{pos} secondary '{x}'" for x in p.secondary if x not in known_pos]
            problems += [f"{pos} archetype '{x}'" for x in p.archetypes if x not in self.archetypes]
        if self.attributes is not None:
            a = self.attributes
            skills, body = set(a.group_of), set(a.body)
            for pos, p in self.positions.items():
                problems += [f"{pos} role group '{x}'" for x in p.roles if x not in a.groups]
                problems += [f"{pos} override '{x}'" for x in p.overrides if x not in skills]
                problems += [f"{pos} body '{x}'" for x in list(p.body) + list(p.ovr_body) if x not in body]
            for name, arch in self.archetypes.items():
                problems += [f"archetype {name} mod '{x}'" for x in arch.mods if x not in a.order]
        if problems:
            raise ValueError("unknown names: " + "; ".join(problems))
        return self

    @model_validator(mode="after")
    def origins_exist_in_places(self):
        missing = [c for c in self.origins.states if c not in self.places.states]
        missing += [c for c in self.origins.countries if c not in self.places.countries]
        if missing:
            raise ValueError(f"origin codes not found in places.yaml: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def name_pools_exist(self):
        if self.names is None:
            return self
        known = US_NAME_POOLS | {"polynesian"} | set(self.names.pools)
        for code, country in self.places.countries.items():
            for group, pools in country.name_pools.items():
                missing = [p for p in pools if p not in known]
                if missing:
                    raise ValueError(f"{code} {group} uses unknown name pools: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def shares_add_up_to_one(self):
        total = sum(p.share for p in self.positions.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"position shares add up to {total:.2f}; they must add up to 1")
        return self


PositionConfig.model_rebuild()
