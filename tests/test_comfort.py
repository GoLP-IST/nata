# -*- coding: utf-8 -*-
import pytest

from nata import comfort
from nata.comfort import load


@pytest.fixture(name="filelist", scope="function")
def dummy_filelist(tmp_path):
    """Creates a dummy file tree"""
    root_file = tmp_path / "root_file.txt"
    root_file.touch()

    first_dir = tmp_path / "first"
    first_dir.mkdir()

    second_dir = first_dir / "second"
    second_dir.mkdir()

    third_dir = second_dir / "third"
    third_dir.mkdir()

    for i in range(10):
        f = first_dir / f"first_{i:02d}.txt"
        f.touch()
        f = second_dir / f"second_{i:02d}.txt"
        f.touch()
        f = third_dir / f"third_{i:02d}.txt"
        f.touch()

    return tmp_path


@pytest.fixture(name="mocked_collection")
def mock_DatasetCollection(filelist, monkeypatch):
    class DummyCollection:
        def __init__(self, root_path=None):
            assert root_path == filelist
            self.paths = []

        def append(self, path):
            self.paths.append(path.name)

    monkeypatch.setattr(comfort, "DatasetCollection", DummyCollection)


@pytest.mark.usefixtures("mocked_collection")
def test_load(filelist):
    basepath = filelist
    test_collection = load(basepath / "*05*")

    for expected in ("first_05.txt", "second_05.txt", "third_05.txt"):
        assert expected in test_collection.paths
        test_collection.paths.remove(expected)
    assert len(test_collection.paths) == 0
