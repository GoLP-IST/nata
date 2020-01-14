from pathlib import Path

import attr
import pytest
import numpy as np

from nata.containers.base import register_backend
from nata.containers.grid import _extract_grid_keys
from nata.containers.grid import GridDataset


@pytest.fixture(name="GridDataset_with_1d_Backend")
def _1d_backend():
    previous_backends = GridDataset._backends.copy()
    GridDataset._backends.clear()

    assert len(GridDataset._backends) == 0

    @attr.s
    @register_backend(GridDataset)
    class Grid_1d:
        location: Path = attr.ib()
        name: str = attr.ib(init=False)
        short_name: str = attr.ib(init=False)
        long_name: str = attr.ib(init=False)
        dim: int = attr.ib(init=False)
        dataset_unit: str = attr.ib(init=False)
        axis_min: np.ndarray = attr.ib(init=False)
        axis_max: np.ndarray = attr.ib(init=False)
        iteration: int = attr.ib(init=False)
        time_step: float = attr.ib(init=False)
        time_unit: str = attr.ib(init=False)
        axes_names: np.ndarray = attr.ib(init=False)
        axes_long_names: np.ndarray = attr.ib(init=False)
        axes_units: np.ndarray = attr.ib(init=False)
        _selection = None

        def __attrs_post_init__(self):
            self.name = "1d_grid_backend"
            self.short_name = "grid"
            self.long_name = "1d_grid"
            self.dim = 1
            self.dataset_unit = "dset unit"
            self.axis_min = np.array([-1.0])
            self.axis_max = np.array([1.0])
            self.time_unit = "time unit"
            self.axes_names = np.array(["axis1"])
            self.axes_long_names = np.array(["axis_1"])
            self.axes_units = np.array(["axis1 units"])

            location_suffix = self.location.suffix.strip(".")
            if location_suffix:
                self.iteration = int(location_suffix)
                self.time_step = int(location_suffix) * 0.1
            else:
                self.iteration = 0
                self.time_step = 0.0

        @property
        def dataset(self):
            return np.arange(256)

        @property
        def shape(self):
            return self.dataset.shape

        @staticmethod
        def is_valid_backend(s):
            if s.stem == "1d_grid":
                return True
            return False

        @property
        def selection(self):
            return self._selection

        @selection.setter
        def selection(self, new):
            assert not isinstance(new, (int, slice))
            self._selection = new

    assert Grid_1d in GridDataset._backends

    yield

    # teardown code
    GridDataset._backends = previous_backends
    assert Grid_1d not in GridDataset._backends


@pytest.fixture(name="patch_location_exist")
def _any_location_exist(monkeypatch):
    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)


@pytest.mark.parametrize(
    "key, spatial, expected",
    [
        # 2 is used just as an example
        # `np.index_exp` is used for easier indexing -> returns correct tuple
        [np.index_exp[2], False, (2,)],
        [np.index_exp[2, 2], False, (2,)],
        [np.index_exp[2:], False, (slice(2, None),)],
        [np.index_exp[2::], False, (slice(2, None),)],
        [np.index_exp[:2], False, (slice(None, 2),)],
        [np.index_exp[:2:], False, (slice(None, 2),)],
        [np.index_exp[::2], False, (slice(None, None, 2),)],
        [np.index_exp[2, 2], True, ((2,), (2,))],
        [np.index_exp[2, ::2], True, ((2,), (slice(None, None, 2),))],
    ],
)
def test_extract_grid_keys(key, spatial, expected):
    assert _extract_grid_keys(key, spatial=spatial) == expected


@pytest.mark.parametrize(
    "parameter, value, type_",
    [
        ("axes_min", [-1.0], np.ndarray),
        ("axes_max", [1.0], np.ndarray),
        ("num_entries", 1, int),
        ("iterations", [0], np.ndarray),
        ("time", [0.0], np.ndarray),
        ("time_unit", "time unit", str),
        ("data", np.arange(256), np.ndarray),
        ("backend_name", "1d_grid_backend", str),
    ],
)
def test_GridDataset_with_1d_single_ds(
    GridDataset_with_1d_Backend, patch_location_exist, parameter, value, type_
):
    grid = GridDataset("1d_grid")
    if type_ == np.ndarray:
        np.testing.assert_array_equal(getattr(grid, parameter), value)
    else:
        assert getattr(grid, parameter) == value
    assert isinstance(getattr(grid, parameter), type_)


@pytest.fixture(name="GridDataset_with_multiple_1d_grids")
def _generate_1d_multiple_backends(
    GridDataset_with_1d_Backend, patch_location_exist,
) -> GridDataset:
    grid = GridDataset("1d_grid.0")
    for i in range(1, 128):
        grid.append(GridDataset(f"1d_grid.{i}"))

    return grid


@pytest.mark.parametrize(
    "parameter, value, type_",
    [
        ("axes_min", np.array([-1.0] * 128).reshape((128, 1)), np.ndarray),
        ("axes_max", np.array([1.0] * 128).reshape((128, 1)), np.ndarray),
        ("num_entries", 128, int),
        ("iterations", np.arange(128), np.ndarray),
        ("time", np.arange(128) * 0.1, np.ndarray),
        ("time_unit", "time unit", str),
        ("data", np.tile(np.arange(256), (128, 1)), np.ndarray),
        ("backend_name", "1d_grid_backend", str),
    ],
)
def test_GridDataset_with_1d_multiple_ds(
    GridDataset_with_multiple_1d_grids, parameter, value, type_,
):
    example_ds = GridDataset_with_multiple_1d_grids
    if type_ == np.ndarray:
        np.testing.assert_array_equal(getattr(example_ds, parameter), value)
    else:
        assert getattr(example_ds, parameter) == value
    assert isinstance(getattr(example_ds, parameter), type_)


@pytest.mark.parametrize(
    "selection, expected_iterations",
    [
        (np.s_[5], np.arange(5, 6)),
        (np.s_[:5], np.arange(5)),
        (np.s_[5:], np.arange(5, 128)),
        (np.s_[5:98], np.arange(5, 98)),
        (np.s_[6:74:4], np.arange(6, 74, 4)),
        (np.s_[5, :], [5]),
        (np.s_[5:98, :], np.arange(5, 98)),
        (np.s_[6:74:4, :], np.arange(6, 74, 4)),
        (np.s_[8, 12], [8]),
        (np.s_[8, 14:87], [8]),
        (np.s_[8, 3:64:3], [8]),
        # TODO: requires changes on backend
        # (np.index_exp[:, 12], np.arange(128)),
        # (np.index_exp[:, 14:87], np.arange(128)),
        # (np.index_exp[:, 3:64:3], np.arange(128)),
    ],
)
def test_GridDataset_getitem(
    GridDataset_with_multiple_1d_grids, selection, expected_iterations
):
    example_ds = GridDataset_with_multiple_1d_grids
    new_ds = example_ds[selection]

    assert new_ds != example_ds
    assert new_ds.store != example_ds.store
    np.testing.assert_array_equal(new_ds.iterations, expected_iterations)
    # TODO: check for spatial selection
