from pydantic import BaseModel, Field
from typing import Literal, Annotated

class Recruit(BaseModel):
    height: int = Field(gt=0)   # inches
    weight: int = Field(gt=0)   # lbs

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