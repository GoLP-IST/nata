from pathlib import Path

import pytest

from nata.containers.helpers import location_exist


@pytest.fixture(name="patch_Path_exists_True")
def _patch_Path_exists_True(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda _: True)


@pytest.fixture(name="patch_Path_exists_False")
def _patch_Path_exists_False(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda _: False)


def test_location_exist(patch_Path_exists_True):
    location_exist(..., ..., Path("."))


def test_location_exist_raises(patch_Path_exists_False):
    with pytest.raises(ValueError, match="does not exist"):
        location_exist(..., ..., Path("."))
