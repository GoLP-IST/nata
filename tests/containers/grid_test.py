# -*- coding: utf-8 -*-
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Optional
from typing import Union

import dask.array as da
import numpy as np
import pytest
from numpy.typing import ArrayLike

from nata.containers import GridArray
from nata.containers.grid import GridBackendType
from nata.utils.container_tools import register_backend


@pytest.fixture(name="valid_grid_backend")
def _valid_grid_backend():
    class DummyBackend:
        name = "dummy_backend"
        location = Path()

        def __init__(self, location: Union[str, Path]) -> None:
            raise NotImplementedError

        @staticmethod
        def is_valid_backend(location: Union[str, Path]) -> bool:
            ...

        dataset_name = str()
        dataset_label = str()
        dataset_unit = str()

        axes_names = []
        axes_labels = []
        axes_units = []
        axes_min = np.empty(0)
        axes_max = np.empty(0)

        iteration = int()
        time_step = float()
        time_unit = str()

        shape = tuple()
        dtype = np.dtype("i")
        ndim = int()

    assert isinstance(DummyBackend, GridBackendType)
    yield DummyBackend

    if DummyBackend.name in GridArray.get_backends():
        GridArray.remove_backend(DummyBackend)


@pytest.fixture(name="grid_files")
def _grid_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # make checking for 'is_valid_backend' always succeed
    monkeypatch.setattr(GridArray, "is_valid_backend", lambda _: True)

    # make 'basedir'
    basedir = tmp_path / "grid_files"
    basedir.mkdir(parents=True, exist_ok=True)

    # create dummy files
    dummy_data = np.arange(32).reshape((4, 8))
    np.savetxt(basedir / "grid.0", dummy_data, delimiter=",")
    np.savetxt(basedir / "grid.1", dummy_data, delimiter=",")
    np.savetxt(basedir / "grid.2", dummy_data, delimiter=",")

    @register_backend(GridArray)
    class Dummy_GridFile:
        name: str = "dummy_backend"

        def __init__(self, location: Union[str, Path]) -> None:
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

        def get_data(self, indexing: Optional[Any] = None) -> ArrayLike:
            if indexing:
                return self.data[indexing]
            else:
                return self.data

    yield basedir
    GridArray.remove_backend(Dummy_GridFile)


def test_GridArray_from_array_default():
    grid_arr = GridArray.from_array(da.arange(12, dtype=int).reshape((4, 3)))

    assert grid_arr.shape == (4, 3)
    assert grid_arr.ndim == 2
    assert grid_arr.dtype == int

    assert grid_arr.axes[0].name == "axis0"
    assert grid_arr.axes[0].label == "unlabeled"
    assert grid_arr.axes[0].unit == ""
    assert grid_arr.axes[0].shape == (4,)

    assert grid_arr.axes[1].name == "axis1"
    assert grid_arr.axes[1].label == "unlabeled"
    assert grid_arr.axes[1].unit == ""
    assert grid_arr.axes[1].shape == (3,)

    assert grid_arr.time.name == "time"
    assert grid_arr.time.label == "time"
    assert grid_arr.time.unit == ""

    assert grid_arr.name == "unnamed"
    assert grid_arr.label == "unlabeled"
    assert grid_arr.unit == ""


def test_GridArray_from_array_check_name():
    grid_arr = GridArray.from_array([], name="custom_name")
    assert grid_arr.name == "custom_name"


def test_GridArray_from_array_check_label():
    grid_arr = GridArray.from_array([], label="custom label")
    assert grid_arr.label == "custom label"


def test_GridArray_from_array_check_unit():
    grid_arr = GridArray.from_array([], unit="custom unit")
    assert grid_arr.unit == "custom unit"


def test_GridArray_from_array_check_time():
    grid_arr = GridArray.from_array([], time=123)
    np.testing.assert_array_equal(grid_arr.time, 123)
    assert grid_arr.time.name == "time"
    assert grid_arr.time.label == "time"
    assert grid_arr.time.unit == ""


def test_GridArray_from_array_check_axes():
    grid_arr = GridArray.from_array([0, 1], axes=[[0, 1]])
    np.testing.assert_array_equal(grid_arr.axes[0], [0, 1])
    assert grid_arr.axes[0].name == "axis0"
    assert grid_arr.axes[0].label == "unlabeled"
    assert grid_arr.axes[0].unit == ""


def test_GridArray_from_array_raise_invalid_name():
    with pytest.raises(ValueError, match="'name' has to be a valid identifier"):
        GridArray.from_array([], name="invalid name")


def test_GridArray_from_array_raise_invalid_time():
    with pytest.raises(ValueError, match="time axis has to be 0 dimensional"):
        GridArray.from_array([], time=[0, 1, 2])


def test_GridArray_from_array_raise_invalid_axes():
    # invalid number of axes
    with pytest.raises(ValueError, match="mismatches with dimensionality of data"):
        GridArray.from_array([], axes=[0, 1])

    # axes which are not 1D dimensional
    with pytest.raises(ValueError, match="only 1D axis for GridArray are supported"):
        GridArray.from_array([0, 1], axes=[[[0, 1]]])

    # axis mismatch with shape of data
    with pytest.raises(ValueError, match="inconsistency between data and axis shape"):
        GridArray.from_array([0, 1], axes=[[0, 1, 2, 3]])


def test_GridArray_change_name_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.name == "unnamed"

    grid_arr.name = "new_name"
    assert grid_arr.name == "new_name"

    with pytest.raises(ValueError, match="name has to be an identifier"):
        grid_arr.name = "invalid name"


def test_GridArray_change_label_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.label == "unlabeled"

    grid_arr.label = "new label"
    assert grid_arr.label == "new label"


def test_GridArray_change_unit_by_prop():
    grid_arr = GridArray.from_array([])
    assert grid_arr.unit == ""

    grid_arr.unit = "new unit"
    assert grid_arr.unit == "new unit"


def test_GridArray_is_valid_backend(valid_grid_backend: GridBackendType):
    assert GridArray.is_valid_backend(valid_grid_backend)
    assert not GridArray.is_valid_backend(object)


def test_GridArray_add_remove_backend(valid_grid_backend: GridBackendType):
    assert valid_grid_backend.name not in GridArray.get_backends()
    GridArray.add_backend(valid_grid_backend)

    assert valid_grid_backend.name in GridArray.get_backends()
    GridArray.remove_backend(valid_grid_backend)

    assert valid_grid_backend.name not in GridArray.get_backends()


def test_GridArray_get_valid_backend(valid_grid_backend: GridBackendType):
    valid_grid_backend.is_valid_backend = staticmethod(lambda p: p == Path())

    # register backend
    GridArray.add_backend(valid_grid_backend)

    assert GridArray.get_valid_backend(Path()) is valid_grid_backend
    assert GridArray.get_valid_backend(Path("some/invalid/path")) is None


def test_GridArray_register_plugin():
    GridArray.register_plugin("my_custom_plugin", lambda _: None)

    assert GridArray.get_plugins()["my_custom_plugin"] == ""


def test_GridArray_calling_plugin_as_method():
    def plugin_function(obj: GridArray, should_return_obj: bool):
        """my custom docs"""
        return obj if should_return_obj else None

    GridArray.register_plugin("return_self_as_method", plugin_function, "method")
    grid = GridArray.from_array([])

    assert grid.return_self_as_method(True) is grid
    assert grid.return_self_as_method(False) is None

    # check plugin perserves docs
    assert grid.return_self_as_method.__doc__ == plugin_function.__doc__


def test_GridArray_calling_plugin_as_property():
    GridArray.register_plugin("return_self_as_property", lambda s: s, "property")
    grid = GridArray.from_array([])
    assert grid.return_self_as_property is grid


def test_GridArray_remove_plugin():
    assert "dummy_plugin" not in GridArray.get_plugins()
    GridArray.register_plugin("dummy_plugin", lambda s: s, "property")
    assert "dummy_plugin" in GridArray.get_plugins()
    GridArray.remove_plugin("dummy_plugin")
    assert "dummy_plugin" not in GridArray.get_plugins()


def test_GridArray_repr():
    grid = GridArray.from_array(np.arange(12, dtype=np.int32).reshape((4, 3)))
    expected_repr = (
        "GridArray<"
        "shape=(4, 3), "
        "dtype=int32, "
        "time=0.0, "
        "axes=(Axis(axis0), Axis(axis1))"
        ">"
    )
    assert repr(grid) == expected_repr


def test_GridArray_repr_html():
    grid = GridArray.from_array(np.arange(12, dtype=np.int32).reshape((4, 3)))
    expected_markdown = """
    | **GridArray** | |
    | ---: | :--- |
    | **name**  | unnamed |
    | **label** | unlabeled |
    | **unit**  | '' |
    | **shape** | (4, 3) |
    | **dtype** | int32 |
    | **time**  | 0.0 |
    | **axes**  | Axis(axis0), Axis(axis1) |

    """

    assert grid._repr_markdown_() == dedent(expected_markdown)


def test_GridArray_len():
    assert len(GridArray.from_array(np.zeros((3,)))) == 3
    assert len(GridArray.from_array(np.zeros((5, 3)))) == 5

    with pytest.raises(TypeError, match="unsized object"):
        len(GridArray.from_array(np.zeros(())))


def test_GridArray_array():
    grid = GridArray.from_array([1, 2])
    np.testing.assert_array_equal(grid, [1, 2])


def test_GridArray_implements_ufunc():
    assert np.add not in GridArray.get_handled_ufuncs()

    @GridArray.implements(np.add)
    def my_function(*args, **kwargs):
        return "my_function implementation"

    assert np.add in GridArray.get_handled_ufuncs()
    assert (GridArray.from_array([]) + 1) == "my_function implementation"
    GridArray.remove_handled_ufuncs(np.add)

    assert np.add not in GridArray.get_handled_ufuncs()


def test_GridArray_implements_array_function():
    assert np.fft.fft not in GridArray.get_handled_array_function()

    @GridArray.implements(np.fft.fft)
    def my_function(*args, **kwargs):
        return "my_function implementation"

    assert np.fft.fft in GridArray.get_handled_array_function()
    assert np.fft.fft(GridArray.from_array([])) == "my_function implementation"
    GridArray.remove_handled_array_function(np.fft.fft)

    assert np.fft.fft not in GridArray.get_handled_array_function()


def test_GridArray_ufunc_proxy():
    grid = GridArray.from_array([1, 2])

    # creation of new object
    np.testing.assert_array_equal(grid + 1, [2, 3])

    # in-place operation
    grid += 1
    np.testing.assert_array_equal(grid, [2, 3])


def test_GridArray_array_function_proxy():
    grid = GridArray.from_array([1, 2])
    np.testing.assert_array_equal(np.fft.fft(grid), np.fft.fft([1, 2]))


def test_GridArray_from_path(grid_files):
    grid = GridArray.from_path(grid_files / "grid.0")

    assert grid.name == "dummy_grid"
    assert grid.label == "dummy grid label"
    assert grid.unit == "dummy unit"

    assert grid.axes[0].name == "dummy_axis0"
    assert grid.axes[1].name == "dummy_axis1"
    assert grid.axes[0].label == "dummy label axis0"
    assert grid.axes[1].label == "dummy label axis1"
    assert grid.axes[0].unit == "dummy unit axis0"

    np.testing.assert_array_equal(grid, np.arange(32).reshape((4, 8)))


def test_GridArray_from_path_check_time_axis(grid_files):
    # using time axis implicit
    grid = GridArray.from_path(grid_files / "grid.0")
    assert grid.time.name == "time"

    # using time axis explicit
    grid = GridArray.from_path(grid_files / "grid.0", time_axis="time")
    assert grid.time.name == "time"

    # using time axis explicit
    grid = GridArray.from_path(grid_files / "grid.0", time_axis="iteration")
    assert grid.time.name == "iteration"
