"""Shared test setup. pytest loads this file automatically."""
import pytest

from recruit_gen.cli import find_config
from recruit_gen.names import name_book
from recruit_gen.stages import load_sport_config


@pytest.fixture(scope="session")
def basketball():
    """Your real configs/basketball.yaml, loaded once for the whole test run."""
    return load_sport_config(find_config("basketball"))


@pytest.fixture(scope="session")
def book(basketball):
    """The loaded name tables for basketball's name settings."""
    return name_book(basketball.names)
