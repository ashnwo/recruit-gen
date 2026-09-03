from pydantic import BaseModel, Field
from typing import Literal

class Recruit(BaseModel):
    height: int = Field(gt=0)   # inches
    weight: int = Field(gt=0)   # lbs

class PhysicalsConfig(BaseModel):
    loc: float
    scale: float = Field(gt=0)
    distribution: Literal["normal", "skewnormal"]