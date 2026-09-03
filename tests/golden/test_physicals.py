import numpy as np
from recruit_gen.stages import (
    physicals
)
from recruit_gen.models import PhysicalsConfig
import pytest
from pydantic import ValidationError



# def test_physicals_returns_valid_recruits():
#     seed = 22
#     rng = np.random.default_rng(seed)
#     recruits = [
#         {"height" : 19}, {"height" : 0}
#     ]
#     recruits = physicals(recruits, rng)

#     for r in recruits:
#         assert r.height > 0


# def test_height_distribution_shape():
#     rng = np.random.default_rng(42)
#     dummy = [{} for _ in range(10000)]
#     out = physicals(dummy, rng)
#     heights = [r.height for r in out]

#     sd = np.std(heights)
#     assert 1.5 < sd < 3.5    # brick fails low, smear fails high



def test_good_config_passes():
    cfg = PhysicalsConfig(loc=79.0, scale=2.5, distribution="normal")
    assert cfg.scale == 2.5

def test_negative_scale_rejected():          # value wrong
    with pytest.raises(ValidationError):
        PhysicalsConfig(loc=79.0, scale=-2, distribution="normal")

def test_string_scale_rejected():            # type wrong
    with pytest.raises(ValidationError):
        PhysicalsConfig(loc=79.0, scale="tall", distribution="normal")

def test_unknown_distribution_rejected():    # name wrong
    with pytest.raises(ValidationError):
        PhysicalsConfig(loc=79.0, scale=2.5, distribution="banana")