# -*- coding: utf-8 -*-
from operator import eq
from operator import ne
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import dask.array as da
import numpy as np
import pytest
from numpy.typing import ArrayLike

from nata.containers import GridArray
from nata.containers import GridDataset
from nata.containers.axis import Axis
from nata.containers.grid import GridBackendType
from nata.utils.container_tools import register_backend

# TODO: restructure the test to represent individual classes rather then
#       GridDataset/GridArray


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


_testCases_getitem = {}
_testCases_getitem["(4,), [int]"] = {
    "arr": [1, 2, 3, 4],
    "indexing": np.s_[1],
}
_testCases_getitem["(4,), [:]"] = {
    "arr": [1, 2, 3, 4],
    "indexing": np.s_[:],
}
_testCases_getitem["(4,), [1:3]"] = {
    "arr": [1, 2, 3, 4],
    "indexing": np.s_[1:3],
    "expected_axes": (Axis([1, 2], name="axis0"),),
}
_testCases_getitem["(4,), [np.newaxis]"] = {
    "arr": [1, 2, 3, 4],
    "indexing": np.s_[np.newaxis],
    "expected_axes": (Axis([0]), Axis(np.arange(4), name="axis0")),
}
_testCases_getitem["(3, 4, 5), [1, 2, 3]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[1, 2, 3],
}
_testCases_getitem["(3, 4, 5, 6, 7), [:, ..., np.newaxis, ...]"] = {
    "arr": np.arange(3 * 4 * 5 * 6 * 7).reshape((3, 4, 5, 6, 7)),
    "indexing": np.s_[1:3, np.newaxis, ..., 4],
    "expected_axes": (
        Axis([1, 2], name="axis0"),
        Axis([0]),
        Axis(np.arange(4), name="axis1"),
        Axis(np.arange(5), name="axis2"),
        Axis(np.arange(6), name="axis3"),
    ),
}


@pytest.mark.parametrize(
    "case", _testCases_getitem.values(), ids=_testCases_getitem.keys()
)
def test_GridArray_getitem(case):
    arr = np.array(case["arr"])
    indexing = case["indexing"]
    expected_arr = case["expected_arr"] if "expected_arr" in case else arr[indexing]
    expected_axes = case["expected_axes"] if "expected_axes" in case else None

    grid = GridArray.from_array(arr)
    subgrid = grid[indexing]
    expected_grid = GridArray.from_array(expected_arr, axes=expected_axes)

    for ax, expected_ax in zip(subgrid.axes, expected_grid.axes):
        assert ax.name == expected_ax.name
        assert ax.label == expected_ax.label
        assert ax.unit == expected_ax.unit
        np.testing.assert_array_equal(ax, expected_ax)

    np.testing.assert_array_equal(subgrid, expected_grid)


def test_GridDataset_from_array_default():
    grid_ds = GridDataset.from_array(da.arange(12, dtype=int).reshape((4, 3)))

    assert grid_ds.shape == (4, 3)
    assert grid_ds.ndim == 2
    assert grid_ds.dtype == int

    assert grid_ds.axes[0].name == "time"
    assert grid_ds.axes[0].label == "time"
    assert grid_ds.axes[0].unit == ""
    assert grid_ds.axes[0].shape == (4,)

    assert grid_ds.axes[1].name == "axis0"
    assert grid_ds.axes[1].label == "unlabeled"
    assert grid_ds.axes[1].unit == ""
    assert grid_ds.axes[1].shape == (4, 3)
    np.testing.assert_array_equal(grid_ds.axes[1], np.tile(np.arange(3), (4, 1)))

    assert grid_ds.time is grid_ds.axes[0]

    assert grid_ds.name == "unnamed"
    assert grid_ds.label == "unlabeled"
    assert grid_ds.unit == ""


def test_GridDataset_from_array_check_name():
    grid_arr = GridDataset.from_array([], name="custom_name")
    assert grid_arr.name == "custom_name"


def test_GridDataset_from_array_check_label():
    grid_arr = GridDataset.from_array([], label="custom label")
    assert grid_arr.label == "custom label"


def test_GridDataset_from_array_check_unit():
    grid_arr = GridDataset.from_array([], unit="custom unit")
    assert grid_arr.unit == "custom unit"


def test_GridDataset_from_array_check_axes():
    grid_arr = GridDataset.from_array(
        [[0, 1], [1, 2]],
        axes=[[0, 1], [[10, 20], [30, 40]]],
    )

    np.testing.assert_array_equal(grid_arr.axes[0], [0, 1])
    assert grid_arr.axes[0].name == "time"
    assert grid_arr.axes[0].label == "time"
    assert grid_arr.axes[0].unit == ""
    assert grid_arr.axes[0].shape == (2,)

    np.testing.assert_array_equal(grid_arr.axes[1], [[10, 20], [30, 40]])
    assert grid_arr.axes[1].name == "axis0"
    assert grid_arr.axes[1].label == "unlabeled"
    assert grid_arr.axes[1].unit == ""
    assert grid_arr.axes[1].shape == (2, 2)


def test_GridDataset_from_array_raise_invalid_name():
    with pytest.raises(ValueError, match="'name' has to be a valid identifier"):
        GridDataset.from_array([], name="invalid name")


def test_GridDataset_from_array_raise_invalid_axes():
    # invalid number of axes
    with pytest.raises(ValueError, match="mismatches with dimensionality of data"):
        GridDataset.from_array([], axes=[0, 1])

    # axes which are not 1D dimensional
    with pytest.raises(ValueError, match="time axis has to be 1D"):
        GridDataset.from_array([0, 1], axes=[[[0, 1]]])

    # only 2D axes for GridDataset are supported
    with pytest.raises(ValueError, match="axis for GridDataset are supported"):
        GridDataset.from_array([[0, 1]], axes=[[0], [[[0, 1]]]])

    # axis mismatch with shape of data
    with pytest.raises(ValueError, match="inconsistency between data and axis shape"):
        GridDataset.from_array([[0, 1]], axes=[[0], [[0, 1, 2, 3]]])


def test_GridDataset_change_name_by_prop():
    grid_ds = GridDataset.from_array([])
    assert grid_ds.name == "unnamed"

    grid_ds.name = "new_name"
    assert grid_ds.name == "new_name"

    with pytest.raises(ValueError, match="name has to be an identifier"):
        grid_ds.name = "invalid name"


def test_GridDataset_change_label_by_prop():
    grid_ds = GridDataset.from_array([])
    assert grid_ds.label == "unlabeled"

    grid_ds.label = "new label"
    assert grid_ds.label == "new label"


def test_GridDataset_change_unit_by_prop():
    grid_ds = GridDataset.from_array([])
    assert grid_ds.unit == ""

    grid_ds.unit = "new unit"
    assert grid_ds.unit == "new unit"


@pytest.mark.parametrize(
    "left, right, operation",
    [
        # general
        (GridArray.from_array([]), GridArray.from_array([]), eq),
        # content
        (GridArray.from_array([1]), GridArray.from_array([2]), eq),
        # shape
        (GridArray.from_array([]), GridArray.from_array([2]), ne),
        # name
        (GridArray.from_array([]), GridArray.from_array([], name="some"), ne),
        # label
        (GridArray.from_array([]), GridArray.from_array([], label="some"), ne),
        # unit
        (GridArray.from_array([]), GridArray.from_array([], unit="some"), ne),
        # time value
        (GridArray.from_array([], time=0), GridArray.from_array([], time=1), eq),
        # time name
        (
            GridArray.from_array([], time=Axis(0)),
            GridArray.from_array([], time=Axis(0, name="some")),
            ne,
        ),
        # time label
        (
            GridArray.from_array([], time=Axis(0)),
            GridArray.from_array([], time=Axis(0, label="some")),
            ne,
        ),
        # time unit
        (
            GridArray.from_array([], time=Axis(0)),
            GridArray.from_array([], time=Axis(0, unit="some")),
            ne,
        ),
        # axes[..] value
        (
            GridArray.from_array([1], axes=[Axis([1])]),
            GridArray.from_array([1], axes=[Axis([2])]),
            eq,
        ),
        # axes[..].name value
        (
            GridArray.from_array([1], axes=[Axis([1])]),
            GridArray.from_array([1], axes=[Axis([1], name="some")]),
            ne,
        ),
        # axes[..].label value
        (
            GridArray.from_array([1], axes=[Axis([1])]),
            GridArray.from_array([1], axes=[Axis([1], label="some")]),
            ne,
        ),
        # axes[..].unit value
        (
            GridArray.from_array([1], axes=[Axis([1])]),
            GridArray.from_array([1], axes=[Axis([1], unit="some")]),
            ne,
        ),
    ],
    ids=(
        "general",
        "content",
        "shape",
        "name",
        "label",
        "unit",
        "time",
        "time.name",
        "time.label",
        "time.unit",
        "axes",
        "axes.name",
        "axes.label",
        "axes.unit",
    ),
)
def test_GridArray_hash(left: GridArray, right: GridArray, operation: Callable):
    assert operation(hash(left), hash(right))


# def test_stack_GridArrays():
#     grid_arr_1 = GridArray.from_array([1, 2, 3])
#     grid_arr_2 = GridArray.from_array([4, 5, 6])

#     grid_ds = stack([grid_arr_1, grid_arr_2])

#     assert isinstance(grid_ds, GridDataset)


def test_GridDataset_from_path(grid_files: Path):
    grid = GridDataset.from_path(grid_files / "*")

    assert grid.name == "dummy_grid"
    assert grid.label == "dummy grid label"
    assert grid.unit == "dummy unit"

    assert grid.axes[0].name == "time"
    assert grid.axes[0].label == "time"
    assert grid.axes[0].unit == "dummy time unit"
    np.testing.assert_array_equal(grid.axes[0], [1.0, 1.0, 1.0])

    assert grid.axes[1].name == "dummy_axis0"
    assert grid.axes[2].name == "dummy_axis1"

    assert grid.axes[1].label == "dummy label axis0"
    assert grid.axes[2].label == "dummy label axis1"

    assert grid.axes[1].unit == "dummy unit axis0"
    assert grid.axes[2].unit == "dummy unit axis1"

    np.testing.assert_array_equal(
        grid, np.tile(np.arange(32).reshape((4, 8)), (3, 1, 1))
    )


_testCases_GridDataset_getitem = {}
_testCases_GridDataset_getitem["(3, 4, 5), [:, :, :]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[:, :, :],
    "instance_after_indexing": GridDataset,
}
_testCases_GridDataset_getitem["(3, 4, 5), [1, :, :]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[1, :, :],
    "instance_after_indexing": GridArray,
}
_testCases_GridDataset_getitem["(3, 4, 5), [:, :, 0]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[:, :, 0],
    "instance_after_indexing": GridDataset,
}
_testCases_GridDataset_getitem["(3, 4, 5), [1]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[1],
    "instance_after_indexing": GridArray,
}
_testCases_GridDataset_getitem["(3, 4, 5), [..., np.newaxis]"] = {
    "arr": np.arange(3 * 4 * 5).reshape((3, 4, 5)),
    "indexing": np.s_[..., np.newaxis],
    "instance_after_indexing": GridDataset,
    "expected_axes": (
        Axis(np.arange(3), name="time", label="time"),
        Axis(np.tile(np.arange(4), (3, 1)), name="axis0"),
        Axis(np.tile(np.arange(5), (3, 1)), name="axis1"),
        Axis(np.zeros((3, 1))),
    ),
}


@pytest.mark.parametrize(
    "case",
    _testCases_GridDataset_getitem.values(),
    ids=_testCases_GridDataset_getitem.keys(),
)
def test_GridDataset_getitem(case):
    arr = np.array(case["arr"])
    indexing = case["indexing"]
    instance = case["instance_after_indexing"]
    expected_arr = case["expected_arr"] if "expected_arr" in case else arr[indexing]
    expected_axes = case["expected_axes"] if "expected_axes" in case else None

    grid = GridDataset.from_array(arr)
    subgrid = grid[indexing]
    expected_grid = instance.from_array(expected_arr, axes=expected_axes)

    assert isinstance(subgrid, instance)
    assert hash(subgrid) == hash(expected_grid)
    np.testing.assert_array_equal(subgrid, expected_grid)


def test_GridDataset_raise_invalid_new_axis():
    grid = GridDataset.from_array(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
    with pytest.raises(IndexError):
        grid[np.newaxis]
