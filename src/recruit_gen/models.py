from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class Recruit(BaseModel):
    """The contract every finished recruit must meet.
    Field order = CSV column order. It grows as each stage becomes real."""
    rank: int = Field(ge=1)              # national rank (scouted)
    stars: int = Field(ge=1, le=5)
    rating: float = Field(ge=0, le=101)  # 247-style rating, e.g. 98.6
    position: str
    pos_rank: int = Field(ge=1)          # rank among this class's players at his position
    height: int = Field(ge=60, le=96)    # inches; outside 5'0"-8'0" = something broke
    height_ft: str                       # same height written like 6'5"
    weight: int = Field(ge=100, le=400)  # lbs; outside this = something broke
    origin: str
    development: str
    name: str
    true_rank: int = Field(ge=1)         # where his REAL talent would rank him (hidden from scouts)
    verdict: str                         # Steal / Underrated / As Advertised / Overrated / Bust


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


class PositionConfig(BaseModel):
    share: float = Field(gt=0, le=1)
    height: PhysicalsConfig
    weight: WeightParams


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


class SportConfig(BaseModel):
    sport: str
    positions: dict[str, PositionConfig] = Field(min_length=1)
    talent: TalentConfig

    @model_validator(mode="after")
    def shares_add_up_to_one(self):
        total = sum(p.share for p in self.positions.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"position shares add up to {total:.2f}; they must add up to 1")
        return self
