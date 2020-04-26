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
from numpy.lib.recfunctions import unstructured_to_structured

from nata.axes import Axis
from nata.axes import GridAxis
from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.containers import ParticleQuantity
from nata.containers import _separation_newaxis
from nata.containers import _transform_particle_data_array
from nata.types import AxisType
from nata.types import GridBackendType
from nata.types import GridDatasetType
from nata.types import ParticleBackendType
from nata.types import ParticleDatasetType


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


@pytest.mark.parametrize(
    "data, expected_array",
    (([1, 2, 3], np.array([1, 2, 3])), ([[1, 2, 3]], np.array([1, 2, 3]))),
)
def test_GridDataset_default_Sequence(data, expected_array):
    grid = GridDataset(data)

    assert grid.backend is None
    assert grid.name == "unnamed"
    assert grid.label == "unnamed"
    assert grid.unit == ""
    assert grid.shape == expected_array.shape
    assert grid.ndim == expected_array.ndim

    assert grid.axes["iteration"] is None
    assert grid.axes["time"] is None
    assert tuple(grid.axes["grid_axes"]) == tuple([None] * len(grid.grid_shape))

    np.testing.assert_array_equal(grid, expected_array)


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


@pytest.mark.parametrize(
    [
        "key",
        "expected_deconstruction",
        "expected_temporal_key",
        "expected_spatial_key",
        "two_types",
    ],
    [
        (np.index_exp[:], np.index_exp[:], (), (), True),
        (np.index_exp[:, :, :], np.index_exp[:, :, :], (), (), True),
        (np.index_exp[4, 2, 0], np.index_exp[4, 2, 0], (), (), True),
        (np.index_exp[None, :], np.index_exp[:], (None,), (), True),
        (
            np.index_exp[None, None, None, :],
            np.index_exp[:],
            (None, None, None),
            (),
            True,
        ),
        (
            np.index_exp[:, None, :, None, :, :, None],
            np.index_exp[:, :, :, :],
            (),
            (1, 3, 6),
            True,
        ),
        (
            np.index_exp[None, None, :, None, :, None, :, :, None],
            np.index_exp[:, :, :, :],
            (None, None),
            (1, 3, 6),
            True,
        ),
        (
            np.index_exp[None, None, :, None, :, None, :, :, None],
            np.index_exp[:, :, :, :],
            (),
            (0, 1, 3, 5, 8),
            False,
        ),
    ],
    ids=[
        ":",
        ":, :, :",
        "int, int, int",
        "newaxis, :",
        "newaxis, newaxis, newaxis, :",
        ":, newaxis, :, newaxis, :, :, newaxis",
        "newaxis, newaxis, :, newaxis, :, newaxis, :, :, newaxis",
        "newaxis, newaxis, :, newaxis, :, newaxis, :, :, newaxis | single_type",
    ],
)
def test_separation_newaxis(
    key,
    expected_deconstruction,
    expected_temporal_key,
    expected_spatial_key,
    two_types,
):
    deconstructed_key, temporal_key, spatial_key = _separation_newaxis(
        key, two_types=two_types
    )

    assert deconstructed_key == expected_deconstruction
    assert temporal_key == expected_temporal_key
    assert spatial_key == expected_spatial_key


@pytest.mark.parametrize(
    [
        "indexing",
        "data",
        "iteration",
        "time",
        "grid_axes",
        "expected_data",
        "expected_iteration",
        "expected_time",
        "expected_grid_axes",
    ],
    [
        ########################################################################
        # > 1d, single time step, [int]
        (
            # indexing
            np.s_[2],
            # data
            np.arange(10).reshape((1, 10)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [GridAxis(np.arange(10).reshape((1, 10)))],
            # expected data
            np.array(2),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [],
        ),
        ########################################################################
        # > 1d, single time step, [:]
        (
            # indexing
            np.s_[:],
            # data
            np.arange(10).reshape((1, 10)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [GridAxis(np.arange(10).reshape((1, 10)))],
            # expected data
            np.arange(10).reshape((1, 10)),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [GridAxis(np.arange(10).reshape((1, 10)))],
        ),
        ########################################################################
        # > 1d, single time step, [range]
        (
            # indexing
            np.s_[1:7],
            # data
            np.arange(10).reshape((1, 10)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [GridAxis(np.arange(10).reshape((1, 10)))],
            # expected data
            np.arange(10).reshape((1, 10))[:, 1:7],
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [GridAxis(np.arange(10).reshape((1, 10))[:, 1:7])],
        ),
        ########################################################################
        # > 1d, multiple time steps, [int]
        (
            # indexing
            np.s_[1],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.arange(21).reshape((3, 7))[1],
            # expected iteration
            Axis(1),
            # expected time
            Axis(1),
            # expected grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))[1]],
        ),
        ########################################################################
        # > 1d, multiple time steps, [:]
        (
            # indexing
            np.s_[:],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.arange(21).reshape((3, 7)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
        ),
        ########################################################################
        # > 1d, multiple time steps, [range]
        (
            # indexing
            np.s_[0:1],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.arange(7).reshape((1, 7)),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [GridAxis(np.arange(7).reshape((1, 7)))],
        ),
        ########################################################################
        # > 1d, multiple time steps, [int, int]
        (
            # indexing
            np.s_[0, 5],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.array(5),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [],
        ),
        ########################################################################
        # > 1d, multiple time steps, [:, int]
        (
            # indexing
            np.s_[:, 5],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.array([5, 12, 19]),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [],
        ),
        ########################################################################
        # > 1d, multiple time steps, [int, :]
        (
            # indexing
            np.s_[1, :],
            # data
            np.arange(21).reshape((3, 7)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
            # expected data
            np.arange(21).reshape((3, 7))[1, :],
            # expected iteration
            Axis(1),
            # expected time
            Axis(1),
            # expected grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))[1]],
        ),
        ########################################################################
        # > 2d, multiple time steps, [:, :, :]
        (
            # indexing
            np.s_[:, :, :],
            # data
            np.arange(105).reshape((3, 7, 5)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
            ],
            # expected data
            np.arange(105).reshape((3, 7, 5)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
            ],
        ),
        ########################################################################
        # > 2d, multiple time steps, [int, :, :]
        (
            # indexing
            np.s_[1, :, :],
            # data
            np.arange(105).reshape((3, 7, 5)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
            ],
            # expected data
            np.arange(105).reshape((3, 7, 5))[1, :, :],
            # expected iteration
            Axis(1),
            # expected time
            Axis(1),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7)))[1],
                GridAxis(np.arange(15).reshape((3, 5)))[1],
            ],
        ),
        ########################################################################
        # > 2d, multiple time steps, [:, int, :]
        (
            # indexing
            np.s_[:, 4, :],
            # data
            np.arange(105).reshape((3, 7, 5)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
            ],
            # expected data
            np.arange(105).reshape((3, 7, 5))[:, 4, :],
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [GridAxis(np.arange(15).reshape((3, 5)))],
        ),
        ########################################################################
        # > 2d, multiple time steps, [:, :, int]
        (
            # indexing
            np.s_[:, :, 2],
            # data
            np.arange(105).reshape((3, 7, 5)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
            ],
            # expected data
            np.arange(105).reshape((3, 7, 5))[:, :, 2],
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [GridAxis(np.arange(21).reshape((3, 7)))],
        ),
        ########################################################################
        # > 4d, multiple time steps, [:, :, :, :, :]
        (
            # indexing
            np.s_[:, :, :, :, :],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [:, ...]
        (
            # indexing
            np.s_[:, ..., :],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [:, ..., int]
        (
            # indexing
            np.s_[:, ..., 4],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 5, 2, 6))[:, ..., 4],
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [...]
        (
            # indexing
            np.s_[...],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [newaxis, ...]
        #   -> new axis for time/iteration
        (
            # indexing
            np.s_[np.newaxis, ...],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((1, 3, 7, 5, 2, 6)),
            # expected iteration
            Axis([[0, 1, 2]]),
            # expected time
            Axis([[0, 1, 2]]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [:, newaxis, ...]
        #   -> new axis with None on 1st position
        (
            # indexing
            np.s_[:, np.newaxis, ...],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 1, 7, 5, 2, 6)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                None,
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [:, :, newaxis,  :, :, :]
        #   -> new axis with None on 2nd position
        (
            # indexing
            np.s_[:, :, np.newaxis, ...],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 1, 5, 2, 6)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                None,
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
        ),
        ########################################################################
        # > 4d, multiple time steps, [..., newaxis]
        #   -> new axis with None on last position
        (
            # indexing
            np.s_[..., np.newaxis],
            # data
            np.arange(1260).reshape((3, 7, 5, 2, 6)),
            # iteration
            Axis([0, 1, 2]),
            # time
            Axis([0, 1, 2]),
            # grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
            ],
            # expected data
            np.arange(1260).reshape((3, 7, 5, 2, 6, 1)),
            # expected iteration
            Axis([0, 1, 2]),
            # expected time
            Axis([0, 1, 2]),
            # expected grid_axes
            [
                GridAxis(np.arange(21).reshape((3, 7))),
                GridAxis(np.arange(15).reshape((3, 5))),
                GridAxis(np.arange(6).reshape((3, 2))),
                GridAxis(np.arange(18).reshape((3, 6))),
                None,
            ],
        ),
        ########################################################################
        # > 3d, single time step [newaxis, ...]
        (
            # indexing
            # np.s_[np.newaxis, ...],
            np.s_[np.newaxis, ...],
            # data
            np.arange(60).reshape((1, 3, 4, 5)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [
                GridAxis(np.arange(3).reshape((1, 3))),
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
            ],
            # expected data
            np.arange(60).reshape((1, 1, 3, 4, 5)),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [
                None,
                GridAxis(np.arange(3).reshape((1, 3))),
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
            ],
        ),
        ########################################################################
        # > 3d, single time step [:, newaxis, ...]
        (
            # indexing
            np.s_[:, np.newaxis, ...],
            # data
            np.arange(60).reshape((1, 3, 4, 5)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [
                GridAxis(np.arange(3).reshape((1, 3))),
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
            ],
            # expected data
            np.arange(60).reshape((1, 3, 1, 4, 5)),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [
                GridAxis(np.arange(3).reshape((1, 3))),
                None,
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
            ],
        ),
        ########################################################################
        # > 3d, single time step [:, :, :, newaxis]
        (
            # indexing
            np.s_[..., np.newaxis],
            # data
            np.arange(60).reshape((1, 3, 4, 5)),
            # iteration
            Axis(0),
            # time
            Axis(0),
            # grid_axes
            [
                GridAxis(np.arange(3).reshape((1, 3))),
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
            ],
            # expected data
            np.arange(60).reshape((1, 3, 4, 5, 1)),
            # expected iteration
            Axis(0),
            # expected time
            Axis(0),
            # expected grid_axes
            [
                GridAxis(np.arange(3).reshape((1, 3))),
                GridAxis(np.arange(4).reshape((1, 4))),
                GridAxis(np.arange(5).reshape((1, 5))),
                None,
            ],
        ),
    ],
    ids=[
        "1d, single time step, [int]",
        "1d, single time step, [:]",
        "1d, single time step, [range]",
        "1d, multiple time steps, [int]",
        "1d, multiple time steps, [:]",
        "1d, multiple time steps, [range]",
        "1d, multiple time steps, [int, int]",
        "1d, multiple time steps, [:, int]",
        "1d, multiple time steps, [int, :]",
        "2d, multiple time steps, [:, :, :]",
        "2d, multiple time steps, [int, :, :]",
        "2d, multiple time steps, [:, int, :]",
        "2d, multiple time steps, [:, :, int]",
        "4d, multiple time steps, [:, :, :, :, :]",
        "4d, multiple time steps, [:, ...]",
        "4d, multiple time steps, [:, ..., 4]",
        "4d, multiple time steps, [...]",
        "4d, multiple time steps, [newaxis, ...]",
        "4d, multiple time steps, [:, newaxis, ...]",
        "4d, multiple time steps, [:, :, newaxis, ...]",
        "4d, multiple time steps, [..., newaxis]",
        "3d, single time step, [newaxis, ...]",
        "3d, single time step, [:, newaxis, ...]",
        "3d, single time step, [..., newaxis]",
    ],
)
def test_GridDataset_getitem(
    indexing,
    data,
    iteration,
    time,
    grid_axes,
    expected_data,
    expected_iteration,
    expected_time,
    expected_grid_axes,
):
    # some valid names GridDatset
    name = "some_name"
    label = "some_label"
    unit = "some_unit"

    grid = GridDataset(
        data,
        iteration=iteration,
        time=time,
        grid_axes=grid_axes,
        name=name,
        label=label,
        unit=unit,
    )

    expected_subgrid = GridDataset(
        expected_data,
        iteration=expected_iteration,
        time=expected_time,
        grid_axes=expected_grid_axes,
        name=name,
        label=label,
        unit=unit,
    )

    subgrid = grid[indexing]

    assert isinstance(subgrid, GridDataset)

    # will check equivalence of axis as well
    # do it two different ways as equivalent is not guaranteed to be commutative
    assert subgrid.equivalent(expected_subgrid)
    assert expected_subgrid.equivalent(subgrid)

    # data
    np.testing.assert_array_equal(subgrid, expected_subgrid)

    # iteration
    np.testing.assert_array_equal(
        subgrid.axes["iteration"], expected_subgrid.axes["iteration"]
    )
    # time
    np.testing.assert_array_equal(
        subgrid.axes["time"], expected_subgrid.axes["time"]
    )

    # grid_axes
    assert len(subgrid.axes["grid_axes"]) == len(
        expected_subgrid.axes["grid_axes"]
    )
    for grid_axis, expected_grid_axis in zip(
        subgrid.axes["grid_axes"], expected_subgrid.axes["grid_axes"]
    ):
        np.testing.assert_array_equal(grid_axis, expected_grid_axis)


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

    grid = GridDataset([1, 2, 3])
    assert repr(grid) == (
        "GridDataset("
        + "name='unnamed', "
        + "label='unnamed', "
        + "unit='', "
        + "shape=(3,), "
        + "iteration=None, "
        + "time=None, "
        + "grid_axes=[]"
        + ")"
    )

    grid = GridDataset([[1, 2, 3]])
    assert repr(grid) == (
        "GridDataset("
        + "name='unnamed', "
        + "label='unnamed', "
        + "unit='', "
        + "shape=(3,), "
        + "iteration=None, "
        + "time=None, "
        + "grid_axes=[None]"
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


def test_ParticleDatset_type_check():
    assert isinstance(ParticleDataset, ParticleDatasetType)


@pytest.fixture(name="SampleParticleBackend")
def _dummy_ParticleBackend():
    class Backend:
        name = "dummy_backend"
        location = None

        dataset_name = "dummy_particles"

        # definition here circumvents the type check an props definition
        num_particles = 10
        quantity_names = ("x1", "p1", "p2")
        quantity_labels = ("x_1", "p_1", "p_2")
        quantity_units = ("", "", "")

        iteration = 0
        time_step = 0.0
        time_unit = "time_unit"

        dtype = np.dtype(int)

        def __init__(
            self, data: Optional[np.ndarray] = None, raise_on_read: bool = True
        ) -> None:
            self.raise_on_read = raise_on_read
            if data is not None:
                fields = [f"q{i}" for i in range(data.shape[-1])]
                self._data = unstructured_to_structured(data, names=fields)

                self.quantity_names = fields
                self.quantity_labels = [f"q_{i}" for i in range(data.shape[-1])]
                self.quantity_units = [""] * data.shape[-1]
                self.num_particles = data.shape[0]
                self.dtype = self._data.dtype
            else:
                self._data = unstructured_to_structured(
                    np.random.random_sample((10, 3)), names=self.quantity_names
                )
                self.dtype = self._data.dtype

        @staticmethod
        def is_valid_backend(path) -> bool:
            return True

        def get_data(self, indexing=None, fields=None) -> np.ndarray:
            if self.raise_on_read:
                raise ValueError

            if fields in self._data.dtype.fields:
                return self._data[fields]
            else:
                raise KeyError

    assert isinstance(Backend, ParticleBackendType)
    return Backend


@pytest.fixture(name="RegisteredSampleParticleBackend")
def _register_dummy_ParticleBackend(SampleParticleBackend):
    ParticleDataset.add_backend(SampleParticleBackend)
    yield SampleParticleBackend
    # teardown
    ParticleDataset.remove_backend(SampleParticleBackend)


def test_ParticleDataset_is_valid_backend(SampleParticleBackend):
    class InvalidBackend:
        pass

    assert ParticleDataset.is_valid_backend(InvalidBackend) is False
    assert ParticleDataset.is_valid_backend(SampleParticleBackend) is True


def test_ParticleDataset_backend_registration(SampleParticleBackend):
    # registration
    ParticleDataset.add_backend(SampleParticleBackend)
    assert SampleParticleBackend in ParticleDataset._backends

    # removing
    ParticleDataset.remove_backend(SampleParticleBackend)
    assert SampleParticleBackend not in ParticleDataset._backends


def test_ParticleDataset_get_backends(RegisteredSampleParticleBackend):
    assert (
        RegisteredSampleParticleBackend.name in ParticleDataset.get_backends()
    )


def test_ParticleDataset_name(RegisteredSampleParticleBackend):
    assert (
        ParticleDataset(np.array([1, 2, 3]), name="some_name").name
        == "some_name"
    )
    assert (
        ParticleDataset(RegisteredSampleParticleBackend()).name
        == "dummy_particles"
    )


def test_ParticleDataset_init_array():
    # without fields
    prt = ParticleDataset(0)
    assert "quant0" in prt.quantities
    assert isinstance(prt.quantities["quant0"], ParticleQuantity)
    np.testing.assert_array_equal(prt.quantities["quant0"], 0)

    assert isinstance(prt.num_particles, AxisType)
    np.testing.assert_array_equal(prt.num_particles, 1)


def test_ParticleDataset_init_backend(RegisteredSampleParticleBackend):
    arr = np.random.random_sample((10, 2))
    backend = RegisteredSampleParticleBackend(data=arr)

    particles = ParticleDataset(backend)
    assert particles.name == "dummy_particles"

    assert isinstance(particles.num_particles, AxisType)
    np.testing.assert_array_equal(particles.num_particles, 10)

    # q0 and q1 are quantity names provided by backend -> avoid using default
    assert "q0" in particles.quantities
    assert isinstance(particles.quantities["q0"], ParticleQuantity)
    assert "q1" in particles.quantities
    assert isinstance(particles.quantities["q1"], ParticleQuantity)

    backend.raise_on_read = False
    np.testing.assert_array_equal(particles.quantities["q0"], arr[:, 0])
    np.testing.assert_array_equal(particles.quantities["q1"], arr[:, 1])


def test_ParticleDataset_axes(RegisteredSampleParticleBackend):
    prt = ParticleDataset(RegisteredSampleParticleBackend())

    assert "iteration" in prt.axes
    assert isinstance(prt.axes["iteration"], AxisType)
    assert "time" in prt.axes
    assert isinstance(prt.axes["time"], AxisType)

    prt = ParticleDataset(0)
    assert "iteration" in prt.axes
    assert prt.axes["iteration"] is None
    assert "time" in prt.axes
    assert prt.axes["time"] is None


def test_ParticleDataset_equivalent():
    particles = ParticleDataset(np.arange(10))

    assert particles.equivalent(ParticleDataset(np.arange(10))) is True
    assert particles.equivalent(object()) is False
    assert particles.equivalent(ParticleDataset(0, name="some_name")) is False


def test_ParticleDataset_append():
    particles = ParticleDataset(np.arange(10))
    particles.append(ParticleDataset(np.arange(10)))
    np.testing.assert_almost_equal(
        particles.quantities["quant0"], [np.arange(10)] * 2
    )
    np.testing.assert_array_equal(particles.num_particles, [10, 10])


def test_ParticleDataset_getitem():
    particles = ParticleDataset(np.arange(10))
    assert particles.quantities["quant0"] is particles["quant0"]


@pytest.mark.parametrize(
    "arr, expected_data",
    [
        (
            np.array(0, dtype=int),
            np.array(
                [[0]], dtype=np.dtype({"names": ["quant0"], "formats": [int]})
            ),
        ),
        (
            np.array([0, 1, 2], dtype=int),
            np.array(
                [[0, 1, 2]],
                dtype=np.dtype({"names": ["quant0"], "formats": [int]}),
            ),
        ),
        (
            np.array([[0, 1, 2], [2, 3, 4]], dtype=int),
            np.array(
                [[0, 1, 2], [2, 3, 4]],
                dtype=np.dtype({"names": ["quant0"], "formats": [int]}),
            ),
        ),
    ],
)
def test_transform_particle_data_array(arr, expected_data):
    transformed_arr = _transform_particle_data_array(arr)

    assert transformed_arr.ndim == 2
    assert transformed_arr.ndim == expected_data.ndim
    assert transformed_arr.shape == expected_data.shape

    np.testing.assert_array_equal(transformed_arr, expected_data)


def test_ParticleQuantity_props_without_backend():
    quant = ParticleQuantity(123)

    assert len(quant) == 1
    assert quant.ndim == 1
    assert quant.shape == (1,)
    assert quant.name == "unnamed"
    assert quant.label == ""
    assert quant.unit == ""
    assert np.issubdtype(quant.dtype, np.integer)
    assert quant.num_particles.shape == ()
    assert quant.num_particles == 1
    assert np.issubdtype(quant.dtype, np.integer)
    np.testing.assert_array_equal(quant, [123])


@pytest.fixture(name="SimpleParticleBackend")
def _particle_quantity_backend():
    class Backend:
        def __init__(
            self,
            data: Optional[np.ndarray] = None,
            raise_on_read: bool = True,
            field: str = "quant0",
        ):
            self._data = data if data is not None else np.arange(10)
            self.num_particles = len(self._data)
            self.raise_on_read = raise_on_read
            self._field = field

        def get_data(self, fields=None):
            if self.raise_on_read:
                raise ValueError("reading block active")

            if fields != self._field:
                raise ValueError("Wrong field")

            return self._data

    return Backend


def test_ParticleQuantity_props_with_backend(SimpleParticleBackend):
    data = np.random.random_sample((10,))
    backend = SimpleParticleBackend(data=data, field="unnamed")
    quant = ParticleQuantity(
        backend, particles=backend.num_particles, dtype=data.dtype
    )

    backend.raise_on_read = True

    # props which don't depend on reading
    assert len(quant) == 1
    assert quant.ndim == 1
    assert quant.shape == (10,)
    assert quant.name == "unnamed"
    assert quant.label == ""
    assert quant.unit == ""
    assert quant.num_particles == 10

    with pytest.raises(ValueError, match="reading block active"):
        quant.data
        np.testing.assert_array_equal(quant, data)

    backend.raise_on_read = False
    np.testing.assert_array_equal(quant, data)


def test_ParticleQuantity_repr():
    quant = ParticleQuantity(1)

    assert (
        repr(quant)
        == "ParticleQuantity(name='unnamed', label='', unit='', len=1)"
    )


def test_ParticleQuantity_append(SimpleParticleBackend):
    # w/o backend
    quant = ParticleQuantity(1)
    quant.append(ParticleQuantity(2))
    assert len(quant) == 2
    assert quant.ndim == 2
    assert quant.shape == (2, 1)
    np.testing.assert_array_equal(quant, [[1], [2]])
    assert quant.num_particles.shape == (2,)
    np.testing.assert_array_equal(quant.num_particles, [1, 1])

    # w/ backend
    arr = np.random.random_sample((10,))
    backend = SimpleParticleBackend(data=arr, field="unnamed")
    quant = ParticleQuantity(
        backend, particles=backend.num_particles, dtype=arr.dtype
    )
    quant.append(
        ParticleQuantity(
            backend, particles=backend.num_particles, dtype=arr.dtype
        )
    )

    assert len(quant) == 2

    assert quant.num_particles.shape == (2,)
    np.testing.assert_array_equal(quant.num_particles, [10, 10])

    assert quant.ndim == 2
    assert quant.shape == (2, 10)
    backend.raise_on_read = False  # allows reading from backend
    np.testing.assert_array_equal(quant, [arr, arr])
