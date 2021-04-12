# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import pytest
from numpy.typing import ArrayLike

from nata.containers import GridDataset
from nata.types import BasicIndexing
from nata.types import FileLocation
from nata.utils.container_tools import register_backend


@pytest.fixture(name="grid_files")
def custom_grid_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # make checking for 'is_valid_backend' always succeed
    monkeypatch.setattr(GridDataset, "is_valid_backend", lambda _: True)

    # make 'basedir'
    basedir = tmp_path / "grid_files"
    basedir.mkdir(parents=True, exist_ok=True)

    # create dummy files
    dummy_data = np.arange(32).reshape((4, 8))
    np.savetxt(basedir / "grid.0", dummy_data, delimiter=",")
    np.savetxt(basedir / "grid.1", dummy_data, delimiter=",")
    np.savetxt(basedir / "grid.2", dummy_data, delimiter=",")

    @register_backend(GridDataset)
    class Dummy_GridFile:
        name: str = "dummy_backend"

        def __init__(self, location: FileLocation) -> None:
            self.location = Path(location)
            self.name = "dummy_backend"
            self.data = np.loadtxt(location, delimiter=",")
            self.dataset_name = "dummy_grid"
            self.dataset_label = "dummy grid label"
            self.dataset_unit = "dummy unit"
            self.ndim = dummy_data.ndim
            self.shape = dummy_data.shape
            self.dtype = dummy_data.dtype
            self.axes_min = (0, 1)
            self.axes_max = (1, 2)
            self.axes_names = ("dummy_axis0", "dummy_axis1")
            self.axes_labels = ("dummy label axis0", "dummy label axis1")
            self.axes_units = ("dummy unit axis0", "dummy unit axis1")
            self.iteration = 0
            self.time_step = 1.0
            self.time_unit = "dummy time unit"

        @staticmethod
        def is_valid_backend(location: Union[str, Path]) -> bool:
            location = Path(location)
            if location.stem == "grid" and location.suffix in (".0", ".1", ".2"):
                return True
            else:
                return False

        def get_data(self, indexing: Optional[BasicIndexing] = None) -> ArrayLike:
            if indexing:
                return self.data[indexing]
            else:
                return self.data

    yield basedir
    GridDataset.remove_backend(Dummy_GridFile)


def test_GridDataset_from_array():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]])

    # general information
    assert grid.name == "unnamed"
    assert grid.label == "unlabeled"
    assert grid.unit == ""

    # array and grid information
    assert grid.ndim == 2
    assert grid.shape == (2, 3)
    assert grid.dtype == int

    # axis info
    assert grid.axes[0].name == "axis0"
    assert grid.axes[1].name == "axis1"


def test_GridDataset_from_array_naming():
    grid = GridDataset.from_array(
        [[1, 2, 3], [3, 4, 5]],
        name="my_new_name",
        label="My New Label",
        unit="some unit",
    )

    assert grid.name == "my_new_name"
    assert grid.label == "My New Label"
    assert grid.unit == "some unit"


def test_GridDataset_change_name():
    grid = GridDataset.from_array([[1, 2, 3], [3, 4, 5]], name="old_name")

    assert grid.name == "old_name"
    grid.name = "new_name"
    assert grid.name == "new_name"


def test_GridDataset_raise_invalid_name():
    with pytest.raises(ValueError, match="has to be an identifier"):
        GridDataset.from_array([1, 2], name="some invalid name")

    grid = GridDataset.from_array([1, 2])
    with pytest.raises(ValueError, match="has to be an identifier"):
        grid.name = "some invalid name"


def test_GridDataset_from_array_with_indexable_axes():
    grid = GridDataset.from_array(
        [[1, 2, 3], [3, 4, 5]],
        indexable_axes=[[0.0, 0.1, 0.2], [10, 20, 30]],
    )

    assert grid.ndim == 2
    assert grid.shape == (2, 3)

    assert grid.axes[0].name == "axis0"
    assert grid.axes[1].name == "axis1"


def test_GridDataset_from_array_with_non_indexable_axes():
    grid = GridDataset.from_array(
        [[1, 2, 3], [3, 4, 5]],
        hidden_axes=[0, 1, 2, 3, 4, 5],
    )

    assert grid.ndim == 2
    assert grid.shape == (2, 3)

    assert grid.axes[0].name == "axis0"
    assert grid.axes[1].name == "axis1"
    assert grid.axes.hidden[0].name == "axis2"


def test_GridDataset_append():
    grid = GridDataset.from_array([1, 2, 3])

    grid.append(GridDataset.from_array([2, 3, 4]))
    grid.append(GridDataset.from_array([3, 4, 5]))

    assert grid.shape == (3, 3)
    np.testing.assert_array_equal(grid, [[1, 2, 3], [2, 3, 4], [3, 4, 5]])


def test_GridDataset_from_path(grid_files):
    grid = GridDataset.from_path(grid_files / "grid.*")
    expected_data = [np.arange(32).reshape((4, 8)) for _ in range(3)]
    np.testing.assert_array_equal(grid, expected_data)
