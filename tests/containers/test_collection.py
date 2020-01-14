from pathlib import Path

import pytest
import numpy as np
import attr

from nata.containers import DatasetCollection
from nata.utils.exceptions import NataInvalidContainer

@pytest.fixture(name="mocked_container_set")
def _mocked_containerset(monkeypatch):
    @attr.s
    class DummyGridDataset:
        type_ = attr.ib(converter=str)
        name = attr.ib(converter=str, init=False)
        appendable = True
        items = attr.ib(init=False)

        def __attrs_post_init__(self):
            self.name = "DummyGridDataset_" + self.type_
            self.items = 1

        @type_.validator
        def _check_correct_dummy(self, attribute, value):
            if not "grid" in value:
                raise NataInvalidContainer

        def append(self, obj):
            self.items += 1


    @attr.s
    class DummyParticleDataset:
        type_ = attr.ib(converter=str)
        name = attr.ib(converter=str, init=False)
        appendable = False
        items = attr.ib(init=False)

        def __attrs_post_init__(self):
            self.name = "DummyParticleDataset_" + self.type_
            self.items = 1

        @type_.validator
        def _check_correct_dummy(self, attribute, value):
            if not "particles" in value:
                raise NataInvalidContainer

    monkeypatch.setattr(
        "nata.containers.collection.DatasetCollection._container_set",
        set([DummyGridDataset, DummyParticleDataset])
    )

def test_DatasetCollection_init():
    empty_collection = DatasetCollection(".")
    assert isinstance(empty_collection.root_path, Path)
    assert empty_collection.root_path == Path(".")



def test_DatasetCollection_datasets():
    empty_collection = DatasetCollection(".")
    assert isinstance(empty_collection.datasets, np.ndarray)
    assert empty_collection.datasets.dtype.type is np.str_
    assert np.alltrue(empty_collection.datasets == np.array([], dtype=str))


def test_DatasetCollection_append_raisesError():
    empty_collection = DatasetCollection(".")

    with pytest.raises(
        ValueError,
        match=f"Can not append object of type*"
    ):
        empty_collection.append(123)
        empty_collection.append("test")
        empty_collection.append([1, 2, 3])


def test_DatasetCollection_append_paths(mocked_container_set):
    collection = DatasetCollection(".")
    assert collection.root_path == Path(".")

    collection.append("grid")
    collection.append("particles")

    assert len(collection.datasets) == 2
    assert "DummyGridDataset" in collection.datasets
    assert "DummyParticleDataset" in collection.datasets

def test_DatasetCollection_append_paths(mocked_container_set):
    subcol_1 = DatasetCollection(".")
    assert subcol_1.root_path == Path(".")

    subcol_1.append("grid_1")
    subcol_1.append("particles_1")

    assert len(subcol_1.datasets) == 2
    assert "DummyGridDataset_grid_1" in subcol_1.datasets
    assert "DummyParticleDataset_particles_1" in subcol_1.datasets

    subcol_2 = DatasetCollection(".")
    assert subcol_2.root_path == Path(".")

    subcol_2.append("grid_2")
    subcol_2.append("particles_2")

    assert len(subcol_2.datasets) == 2
    assert "DummyGridDataset_grid_2" in subcol_2.datasets
    assert "DummyParticleDataset_particles_2" in subcol_2.datasets

    subcol_1.append(subcol_2)
    total = subcol_1

    assert len(total.datasets) == 4

    for s in [
        "DummyGridDataset_grid_1",
        "DummyParticleDataset_particles_1",
        "DummyGridDataset_grid_2",
        "DummyParticleDataset_particles_2"
    ]:
        assert s in total.datasets


def test_DatasetCollection_empty():
    empty = DatasetCollection(".")
    empty.append(Path(__file__))
    assert len(empty.datasets) == 0


def test_DatasetCollection_append_existingDataset(mocked_container_set):
    collection = DatasetCollection(".")

    # appendable object
    collection.append("grid")
    collection.append("grid")
    assert collection["DummyGridDataset_grid"].items == 2

    # non-appendable object
    collection.append("particles")
    with pytest.raises(ValueError, match="not appendable"):
        collection.append("particles")
    assert collection["DummyParticleDataset_particles"].items == 1