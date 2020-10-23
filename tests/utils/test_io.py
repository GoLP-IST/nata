# -*- coding: utf-8 -*-
import pytest

from nata.comfort import FileList


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


def test_FileList_nonExisting():
    with pytest.raises(ValueError, match="Passed a non-existing path!"):
        FileList("some/non/existing/path")


def test_FileList_singleFile_entry(filelist):
    basepath = filelist
    path_entry = FileList(basepath / "root_file.txt")
    assert path_entry.entrypoint == basepath / "root_file.txt"
    assert path_entry.recursive is True
    assert path_entry.search_pattern == "*"
    assert path_entry.is_single_file is True
    assert all(p == (basepath / "root_file.txt") for p in path_entry.paths)
    assert path_entry.parent_directory == basepath


def test_FileList_wildCards_recursive(filelist):
    basepath = filelist
    wildcard_file = FileList(basepath / "second_*")
    expected_files = [f"second_{i:02d}.txt" for i in range(10)]

    assert wildcard_file.entrypoint == basepath
    assert wildcard_file.parent_directory == basepath
    assert wildcard_file.recursive is True
    assert wildcard_file.search_pattern == "second_*"
    for fp in wildcard_file.paths:
        assert fp.name in expected_files
        expected_files.remove(fp.name)
    assert len(expected_files) == 0


def test_FileList_wildCards_nonrecursive(filelist):
    basepath = filelist
    wildcard_file = FileList(basepath / "first" / "second" / "*.txt", recursive=False)
    expected_files = [f"second_{i:02d}.txt" for i in range(10)]

    assert wildcard_file.entrypoint == basepath / "first" / "second"
    assert wildcard_file.parent_directory == basepath / "first" / "second"
    assert wildcard_file.recursive is False
    for fp in wildcard_file.paths:
        assert fp.name in expected_files
        expected_files.remove(fp.name)
    assert len(expected_files) == 0
