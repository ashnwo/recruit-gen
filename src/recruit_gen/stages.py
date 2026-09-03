from recruit_gen.models import Recruit, PhysicalsConfig
import numpy as np

def class_shape(count: int, rng) -> list[dict]:
    return [{} for _ in range(count)]   # stub: N blank recruits

def origins(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["origin"] = "USA"   # stub
    return recruits

def positions(recruits: list[dict], rng) -> list[dict]:
    for r in recruits:
        r["position"] = "PG"   # stub: everyone's a point guard for now
    return recruits

def physicals(recruits: list[dict], rng, config: PhysicalsConfig) -> list[Recruit]:
    recruits_out = []

    for r in recruits:
        height = rng.normal(loc=config.loc, scale=config.scale)
        height = round(height)
        
        recruit = Recruit(
            height = height,
            weight = 190
        )
        recruits_out.append(recruit)
    return recruits_out

def talent(recruits, rng, exponent=3):
    for r in recruits:
        random_number = rng.random()
        talent_score = random_number ** exponent
        r["talent"] = talent_score 
    return recruits

def development(recruits: int, rng) -> list[dict]:
    for r in recruits:
        r["development"] = "star"   # stub
    return recruits

def names(recruits: int, rng) -> list[dict]:
    for r in recruits:
        r["names"] = "bobby smith"   # stub
    return recruits

def validate_dedup(recruits: int, rng) -> list[dict]:
    for r in recruits:
        r["validate_dedup"] = ""   # stub
    return recruits


