from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class Recruit(BaseModel):
    """The contract every finished recruit must meet.
    It grows as each stage becomes real (ratings, names...)."""
    origin: str
    position: str
    height: int = Field(ge=60, le=96)    # inches; outside 5'0"-8'0" = something broke
    height_ft: str                       # same height written like 6'5"
    weight: int = Field(ge=100, le=400)  # lbs; outside this = something broke
    talent: float = Field(ge=0, lt=1)
    development: str
    name: str


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


class SportConfig(BaseModel):
    sport: str
    positions: dict[str, PositionConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def shares_add_up_to_one(self):
        total = sum(p.share for p in self.positions.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"position shares add up to {total:.2f}; they must add up to 1")
        return self
