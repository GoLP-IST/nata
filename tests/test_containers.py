# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import basic_indices
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.extra.numpy import integer_dtypes
from hypothesis.strategies import composite
from hypothesis.strategies import one_of

from nata.containers import GridDataset
from nata.types import AxisType
from nata.types import GridBackendType
from nata.types import GridDatasetType


@composite
def random_array(
    draw,
    *,
    min_dims: int = 0,
    max_dims: int = 3,
    return_with_indexing: bool = False,
):
    arr = draw(
        arrays(
            dtype=one_of(integer_dtypes(), floating_dtypes()),
            shape=array_shapes(min_dims=min_dims, max_dims=max_dims),
        )
    )

    assume(not np.any(np.isnan(arr)))
    assume(np.all(np.isfinite(arr)))

    if return_with_indexing:
        ind = draw(basic_indices(arr.shape))
        return arr, ind
    else:
        return arr


class ReadingData(Exception):
    pass


def test_GridDataset_type_check():
    assert isinstance(GridDataset, GridDatasetType)


@pytest.fixture(name="SampleGridBackend")
def _dummy_GridBackend():
    class Backend:
        name = "dummy_backend"
        location = None

        dataset_name = "dummy_grid"
        dataset_label = "dummy_label"
        dataset_unit = "dummy_unit"

        iteration = 0
        time_step = 0.0
        time_unit = "time_unit"

        # ignore definition - only used for type-checking
        shape: Tuple[int] = tuple()
        dtype: np.dtype = np.dtype(int)
        ndim: int = 1

        axes_names: Sequence[str] = []
        axes_labels: Sequence[str] = []
        axes_units: Sequence[str] = []
        axes_min: np.ndarray = np.array(None)
        axes_max: np.ndarray = np.array(None)

        def __init__(
            self, data: Optional[Any] = None, raise_on_read: bool = True
        ):
            if data is not None:
                self._data = data
            else:
                self._data = np.random.random_sample((10,))

            self.shape = self._data.shape
            self.ndim = self._data.ndim
            self.dtype = self._data.dtype

            self.axes_names = []
            self.axes_labels = []
            self.axes_units = []
            axes_min = []
            axes_max = []

            for i in range(len(self.shape)):
                self.axes_names.append(f"axis{i}")
                self.axes_labels.append(f"axis_{i}")
                self.axes_units.append(f"axis{i}_units")

                axes_min.append(-1.0)
                axes_max.append(+1.0)

            self.axes_min = np.array(axes_min)
            self.axes_max = np.array(axes_max)

            self.raise_on_read = raise_on_read

        @staticmethod
        def is_valid_backend(path) -> bool:
            return True

        def get_data(self, indexing=None) -> np.ndarray:
            if self.raise_on_read:
                raise ReadingData

            return self._data

    # ensure dummy backend is of valid type
    assert isinstance(Backend, GridBackendType)

    GridDataset.add_backend(Backend)
    yield Backend
    # teardown
    GridDataset.remove_backend(Backend)


def test_GridDataset_registration(SampleGridBackend):
    assert SampleGridBackend.name in GridDataset.get_backends()


@given(backend_arr=random_array())
def test_GridDataset_default_init_backend(
    backend_arr, SampleGridBackend: GridBackendType
):
    backend = SampleGridBackend(data=backend_arr, raise_on_read=False)
    grid = GridDataset(backend)

    # > general information
    assert grid.backend == backend.name
    assert grid.name == backend.dataset_name
    assert grid.label == backend.dataset_label
    assert grid.unit == backend.dataset_unit

    # > iteration axis
    assert "iteration" in grid.axes
    assert isinstance(grid.axes["iteration"], AxisType)
    assert grid.axes["iteration"].name == "iteration"
    assert grid.axes["iteration"].label == "iteration"
    assert grid.axes["iteration"].unit == ""
    np.testing.assert_array_equal(grid.axes["iteration"], backend.iteration)

    # > time axis
    assert "time" in grid.axes
    assert isinstance(grid.axes["time"], AxisType)
    assert grid.axes["time"].name == "time"
    assert grid.axes["time"].label == "time"
    assert grid.axes["time"].unit == backend.time_unit
    np.testing.assert_array_equal(grid.axes["time"], backend.time_step)

    # > grid axes
    assert "grid_axes" in grid.axes
    assert isinstance(grid.axes["grid_axes"], list)
    assert len(grid.axes["grid_axes"]) == len(backend.axes_names)
    # name
    for grid_axis, expected_name in zip(
        grid.axes["grid_axes"], backend.axes_names
    ):
        assert grid_axis.name == expected_name
    # label
    for grid_axis, expected_label in zip(
        grid.axes["grid_axes"], backend.axes_labels
    ):
        assert grid_axis.label == expected_label
    # unit
    for grid_axis, expected_unit in zip(
        grid.axes["grid_axes"], backend.axes_units
    ):
        assert grid_axis.unit == expected_unit

    # > array props
    assert grid.shape == backend.shape
    assert grid.grid_shape == backend.shape
    assert grid.ndim == backend.ndim
    assert grid.dtype == backend.dtype

    np.testing.assert_array_equal(grid, backend.get_data())
    np.testing.assert_array_equal(grid.data, backend.get_data())


def test_GridDataset_default_init_array():
    arr = np.random.random_sample((10,))
    grid = GridDataset(arr)

    # > general information
    assert grid.backend is None
    assert grid.name == "unnamed"
    assert grid.label == "unnamed"
    assert grid.unit == ""

    # > iteration axis
    assert "iteration" in grid.axes
    assert grid.axes["iteration"] is None

    # > time axis
    assert "time" in grid.axes
    assert grid.axes["time"] is None

    # > grid axes
    assert "grid_axes" in grid.axes
    assert isinstance(grid.axes["grid_axes"], list)
    assert len(grid.axes["grid_axes"]) == 0

    # > array props
    assert grid.shape == arr.shape
    assert grid.grid_shape == arr.shape[1:]
    assert grid.ndim == arr.ndim
    assert grid.dtype == arr.dtype

    np.testing.assert_array_equal(grid, arr)
    np.testing.assert_array_equal(grid.data, arr)


def test_GridDataset_default_init_array_2d_array():
    arr = np.random.random_sample((10, 5))
    grid = GridDataset(arr)

    # > grid axes
    assert "grid_axes" in grid.axes
    assert isinstance(grid.axes["grid_axes"], list)
    assert len(grid.axes["grid_axes"]) == 1

    # > array props
    assert grid.shape == arr.shape
    assert grid.grid_shape == arr.shape[1:]
    assert grid.ndim == arr.ndim
    assert grid.dtype == arr.dtype

    np.testing.assert_array_equal(grid, arr)
    np.testing.assert_array_equal(grid.data, arr)


def test_GridDataset_change_data():
    grid = GridDataset(np.random.random_sample((10,)))

    new = np.random.random_sample(grid.shape)
    grid.data = new

    np.testing.assert_array_equal(grid, new)


def test_GridDataset_change_data_invalid_shape():
    with pytest.raises(
        ValueError, match=r"Shapes inconsistent \(10,\) -> \(1,\)"
    ):
        grid = GridDataset(np.random.random_sample((10,)))
        grid.data = np.array((1,))


def test_GridDataset_change_data_change_dtype():
    grid = GridDataset(np.arange(10, dtype=int))
    assert np.issubdtype(grid, np.integer)
    grid.data = np.arange(10, dtype=float)
    assert np.issubdtype(grid, np.floating)


def test_GridDataset_equivalent():
    arr = np.array(1, dtype=int)
    base = GridDataset(arr)

    assert base.equivalent(GridDataset(arr)) is True
    assert base.equivalent(object) is False

    # changed props
    assert base.equivalent(GridDataset(arr, name="other")) is False
    assert base.equivalent(GridDataset(arr, label="other")) is False
    assert base.equivalent(GridDataset(arr, unit="other")) is False
    assert base.equivalent(GridDataset(arr.astype(float))) is True
    assert base.equivalent(GridDataset(arr.reshape((1, 1)))) is True


def test_GridDataset_append():
    grid = GridDataset(np.array(1))
    grid.append(GridDataset(np.array(2)))
    np.testing.assert_array_equal(grid, np.array([1, 2]))


def test_GridDataset_append_based_on_backend(
    SampleGridBackend: GridBackendType,
):
    backend = SampleGridBackend(raise_on_read=False)
    grid = GridDataset(backend)
    grid.append(GridDataset(backend))

    assert len(grid) == 2
    assert grid.shape == (2,) + backend.shape

    np.testing.assert_array_equal(
        grid.axes["iteration"], [backend.iteration] * 2
    )
    np.testing.assert_array_equal(grid.axes["time"], [backend.time_step] * 2)

    for axis in grid.axes["grid_axes"]:
        min_ = backend.axes_min[0]
        max_ = backend.axes_max[0]
        N = backend.shape[0]

        np.testing.assert_array_equal(
            axis, [np.linspace(min_, max_, N)] * 2,
        )

    np.testing.assert_array_equal(grid, [backend.get_data()] * 2)


def test_GridDataset_append_wrong_type():
    with pytest.raises(
        TypeError, match=r"Can not append 'int' to 'GridDataset'"
    ):
        grid = GridDataset(np.array(1))
        grid.append(2)


def test_GridDataset_append_wrong_GridDatset():
    # should be triggered by equivalence - checking only name
    with pytest.raises(ValueError, match="GridDatasets are not equivalent"):
        grid = GridDataset(np.array(1))
        grid.append(GridDataset(np.array(2), name="test"))


def test_GridDataset_iterator_single_item(SampleGridBackend):
    backend = SampleGridBackend()
    grid = GridDataset(backend)

    for g in grid:
        assert g is grid


def test_GridDataset_iterator_multiple_items(SampleGridBackend):
    backend = SampleGridBackend(raise_on_read=False)
    grid = GridDataset(backend)
    grid.append(GridDataset(backend))
    grid.append(GridDataset(backend))

    for g in grid:
        assert g is not grid
        assert isinstance(g, GridDataset)
        assert g.name == grid.name
        assert g.label == grid.label
        assert g.unit == grid.unit
        for axis_name in grid.axes.keys():
            assert axis_name in g.axes
            origin_axis = grid.axes[axis_name]

            # if     grid_axis -> List[AxisType]
            # if not grid_axis -> Axistype
            if axis_name != "grid_axes":
                assert g.axes[axis_name].equivalent(origin_axis)
            else:
                for grid_axis, origin_grid_axis in zip(
                    g.axes["grid_axes"], grid.axes["grid_axes"]
                ):
                    assert origin_grid_axis.equivalent(grid_axis)

        np.testing.assert_array_equal(g, backend.get_data())


@pytest.mark.skip
@given(arr_and_ind=random_array(return_with_indexing=True))
def test_GridDataset_getitem_single(arr_and_ind, SampleGridBackend):
    arr, key = arr_and_ind
    grid = GridDataset(SampleGridBackend(data=arr, raise_on_read=False))
    subgrid = grid[key]

    assert isinstance(subgrid, GridDataset)
    assert grid.equivalent(subgrid)
    np.testing.assert_array_equal(subgrid, arr[key])


@pytest.mark.skip
@pytest.mark.parametrize(
    "arr, ind, newarr, num_grid_axes, num_iterations",
    [
        (np.arange(10).reshape((2, 5)), np.s_[0], None, 1, 1),
        (np.arange(10).reshape((2, 5)), np.s_[:, :], None, 1, 2),
    ],
)
def test_GridDataset_getitem_multi(
    arr, ind, newarr, num_grid_axes, num_iterations, SampleGridBackend
):
    grid = GridDataset(SampleGridBackend(data=arr[0], raise_on_read=False))

    for i, subarr in enumerate(arr):
        if i == 0:
            continue
        grid.append(
            GridDataset(SampleGridBackend(data=subarr, raise_on_read=False))
        )

    subgrid = grid[ind]
    if newarr is not None:
        subarr = newarr
    else:
        subarr = arr[ind]

    assert subgrid is not grid
    np.testing.assert_array_equal(subgrid, subarr)

    assert len(subgrid.axes["grid_axes"]) == num_grid_axes
    assert len(subgrid.axes["iteration"]) == num_iterations
    assert len(subgrid.axes["time"]) == num_iterations


def test_GridDataset_change_name():
    grid = GridDataset(0, name="old")
    assert grid.name == "old"
    grid.name = "new"
    assert grid.name == "new"


def test_GridDataset_change_label():
    grid = GridDataset(0, label="old")
    assert grid.label == "old"
    grid.label = "new"
    assert grid.label == "new"


def test_GridDataset_change_unit():
    grid = GridDataset(0, unit="old")
    assert grid.unit == "old"
    grid.unit = "new"
    assert grid.unit == "new"


def test_GridDataset_repr_default():
    grid = GridDataset(np.array(0))
    assert repr(grid) == (
        "GridDataset("
        + "name='unnamed', "
        + "label='unnamed', "
        + "unit='', "
        + "shape=(), "
        + "iteration=None, "
        + "time=None, "
        + "grid_axes=[]"
        + ")"
    )


def test_GridDataset_repr_backend(SampleGridBackend: GridBackendType):
    backend = SampleGridBackend(data=np.arange(10).reshape((2, 5)))
    grid = GridDataset(backend)
    assert repr(grid) == (
        "GridDataset("
        + f"name='{backend.dataset_name}', "
        + f"label='{backend.dataset_label}', "
        + f"unit='{backend.dataset_unit}', "
        + f"shape={backend.shape}, "
        + f"iteration={backend.iteration}, "
        + f"time={backend.time_step}, "
        + f"grid_axes=["
        + f"Axis('{backend.axes_names[0]}', len=1, shape=(2,)), "
        + f"Axis('{backend.axes_names[1]}', len=1, shape=(5,))"
        + "]"
        + ")"
    )
