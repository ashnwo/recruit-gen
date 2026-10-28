from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class Recruit(BaseModel):
    """The contract every finished recruit must meet.
    It grows as each stage becomes real (position, ratings, names...)."""
    origin: str
    position: str
    height: int = Field(gt=0)            # inches
    weight: int = Field(gt=0)            # lbs
    talent: float = Field(ge=0, lt=1)
    development: str
    name: str


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
    """Plain dict (e.g. from YAML) -> NormalParams or SkewNormalParams.
    Raises pydantic.ValidationError if anything is wrong."""
    return _physicals_adapter.validate_python(data)
