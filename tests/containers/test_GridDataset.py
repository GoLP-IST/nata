# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pytest

from nata import GridDataset
from nata.axes import Axis
from nata.axes import GridAxis
from nata.types import AxisType
from nata.types import BasicIndexing
from nata.types import FileLocation
from nata.types import GridBackendType
from nata.types import GridDatasetType


@pytest.fixture(name="TestGridBackend")
def _TestGridBackend():
    """Fixture for returning GridBackend"""

    class TestGridBackend:
        """Test backend for GridDatsets

        * Backend provides 3D random data.
        """

        name = "TetsGridBackend"
        location = None

        dataset_name = "test_dataset_name"
        dataset_label = "test dataset label"
        dataset_unit = "test dataset unit"

        axes_names = ("axes0", "axes1", "axes2")
        axes_labels = ("axes 0", "axes 1", "axes 3")
        axes_units = ("axes unit 0", "axes unit 1", "axes unit 2")
        axes_min = np.array([-1.0, -2.0, -3.0])
        axes_max = np.array([0.5, 1.5, 2.5])

        iteration = 42

        time_step = 12.3
        time_unit = "time unit"

        shape = (4, 5, 6)
        dtype = np.float

        ndim = 3

        def __init__(
            self,
            location: Optional[FileLocation] = None,
            raise_on_read: bool = True,
        ):
            self.location = location
            self._raise_on_read = raise_on_read

        @staticmethod
        def is_valid_backend(location: Union[Path, str]) -> bool:
            return Path(location) == Path("TestGridBackend_location")

        def get_data(
            self, indexing: Optional[BasicIndexing] = None
        ) -> np.ndarray:
            if self._raise_on_read:
                raise IOError("Should not read any file")

            data = np.arange(4 * 5 * 6).reshape(self.shape)
            return data[indexing] if indexing else data

    # ensure dummy backend is of valid type
    assert isinstance(TestGridBackend, GridBackendType)

    GridDataset.add_backend(TestGridBackend)
    yield TestGridBackend
    # teardown
    GridDataset.remove_backend(TestGridBackend)


@pytest.fixture(name="MultipleBackends")
def _TestMultiBackend():
    """Fixture for returning multiple backends"""

    class MultiBackend:
        """Test backend for MultiBackend

        * simple backend to provide data based on the path
        * `MultiBackend.*` triggers this backend but not `MultiBackend.*.*`
        * `iteration` is based on suffix `*.42` means iteration is `42`
        * `time_step` is based on iteration and is `float(iteration * 10)`
        """

        name = "MultiTestBackend"
        location = None

        dataset_name = "test_dataset_name"
        dataset_label = "test dataset label"
        dataset_unit = "test dataset unit"

        axes_names = ("axes0", "axes1", "axes2")
        axes_labels = ("axes 0", "axes 1", "axes 3")
        axes_units = ("axes unit 0", "axes unit 1", "axes unit 2")
        axes_min = np.array([-1.0, -2.0, -3.0])
        axes_max = np.array([0.5, 1.5, 2.5])

        iteration = None
        time_step = None
        time_unit = "time unit"

        shape = (4, 5, 6)
        dtype = np.float

        ndim = 3

        def __init__(
            self,
            location: Optional[FileLocation] = None,
            raise_on_read: bool = True,
        ):
            self.location = location
            self._raise_on_read = raise_on_read
            self.iteration = int(Path(location).suffix.strip("."))
            self.time_step = float(self.iteration * 10)

        @staticmethod
        def is_valid_backend(location: Union[Path, str]) -> bool:
            return Path(location).stem == "MultiGridBackend"

        def get_data(
            self, indexing: Optional[BasicIndexing] = None
        ) -> np.ndarray:
            data = np.arange(4 * 5 * 6).reshape(self.shape)
            return data[indexing] if indexing else data

    # ensure dummy backend is of valid type
    assert isinstance(MultiBackend, GridBackendType)

    GridDataset.add_backend(MultiBackend)
    yield MultiBackend
    # teardown
    GridDataset.remove_backend(MultiBackend)


def test_GridDataset_isinstance_GridDatasetType():
    """Ensures that a GridDataset fulfills `GridDatasetType` protocol"""
    assert isinstance(GridDataset, GridDatasetType)


def test_GridDataset_registration(TestGridBackend):
    """Check if fixture registers backend properly"""
    assert TestGridBackend.name in GridDataset.get_backends()


@pytest.mark.parametrize(
    "attr, value",
    [
        ("backend", "TetsGridBackend"),
        ("name", "test_dataset_name"),
        ("label", "test dataset label"),
        ("unit", "test dataset unit"),
    ],
    ids=["backend", "name", "label", "unit"],
)
def test_GridDataset_attr_propagation_from_Backend(
    TestGridBackend, attr, value
):
    """Parameterize check for different props of GridDataset"""
    ds = GridDataset("TestGridBackend_location")
    assert getattr(ds, attr) == value


def test_GridDataset_axes_from_Backend(TestGridBackend):
    """Tests extraction correct type for .axes and its proper extraction"""
    ds = GridDataset("TestGridBackend_location")

    assert isinstance(ds.axes, Mapping)

    for axes_name, type_ in [
        ("iteration", AxisType),
        ("time", AxisType),
        ("grid_axes", Sequence),
    ]:
        assert axes_name in ds.axes
        assert isinstance(ds.axes.get(axes_name), type_)


@pytest.mark.parametrize(
    "attr, value",
    [("name", "iteration"), ("label", "iteration"), ("unit", "")],
    ids=["name", "label", "unit"],
)
def test_GridDataset_iteration_axis_from_Backend(TestGridBackend, attr, value):
    """Extraction is correct for iteration axis. Check attributes for axis"""
    ds = GridDataset("TestGridBackend_location")
    assert getattr(ds.axes["iteration"], attr) == value


@pytest.mark.parametrize(
    "attr, value",
    [("name", "time"), ("label", "time"), ("unit", "time unit")],
    ids=["name", "label", "unit"],
)
def test_GridDataset_time_axis_from_Backend(TestGridBackend, attr, value):
    """Extraction is correct for iteration axis. Check attributes for axis"""
    ds = GridDataset("TestGridBackend_location")
    assert getattr(ds.axes["time"], attr) == value


def test_GridDataset_grid_axes_from_Backend(TestGridBackend):
    """Tests grid_axes have been properly extracted."""
    ds = GridDataset("TestGridBackend_location")

    # obtain the expected ndim, names, labels and units from fixture
    assert len(ds.axes["grid_axes"]) == TestGridBackend.ndim

    expected_names = TestGridBackend.axes_names
    expected_labels = TestGridBackend.axes_labels
    expected_units = TestGridBackend.axes_units

    for axis, expected_name in zip(ds.axes["grid_axes"], expected_names):
        assert axis.name == expected_name

    for axis, expected_label in zip(ds.axes["grid_axes"], expected_labels):
        assert axis.label == expected_label

    for axis, expected_unit in zip(ds.axes["grid_axes"], expected_units):
        assert axis.unit == expected_unit


def test_GridDataset_grid_shape_from_Backend(TestGridBackend):
    """Tests propagation of grid_shape from backend"""
    ds = GridDataset("TestGridBackend_location")
    assert ds.grid_shape == TestGridBackend.shape


@pytest.mark.skip
def test_GridDataset_grid_ndim_from_Backend(TestGridBackend):
    """Tests propagation of ndim from backend to dataset"""
    ds = GridDataset("TestGridBackend_location")
    assert ds.grid_nim == TestGridBackend.ndim


def test_GridDataset_array_props_from_Backend(TestGridBackend):
    """Tests propagation of array properties from backend"""
    ds = GridDataset("TestGridBackend_location")
    assert ds.shape == TestGridBackend.shape
    assert ds.ndim == TestGridBackend.ndim
    assert ds.dtype == TestGridBackend.dtype


def test_GridDataset_array_representation_from_Backend(TestGridBackend):
    """Tests if GridDataset represent same array as backend"""
    backend = TestGridBackend(raise_on_read=False)
    ds = GridDataset(backend)
    np.testing.assert_array_equal(ds, np.arange(4 * 5 * 6).reshape((4, 5, 6)))
    np.testing.assert_array_equal(
        ds.data, np.arange(4 * 5 * 6).reshape((4, 5, 6))
    )


def test_GridDataset_array_representation_from_MultiBackend(MultipleBackends):
    """Tests if GridDataset represent same array in multi case"""
    ds = GridDataset("MultiGridBackend.0")
    # using append here -> only way to construct multi dataset from backend
    ds.append(GridDataset("MultiGridBackend.10"))
    ds.append(GridDataset("MultiGridBackend.20"))
    ds.append(GridDataset("MultiGridBackend.30"))

    assert len(ds) == 4
    np.testing.assert_array_equal(
        ds, np.array([np.arange(4 * 5 * 6).reshape((4, 5, 6))] * 4)
    )
    np.testing.assert_array_equal(
        ds.data, np.array([np.arange(4 * 5 * 6).reshape((4, 5, 6))] * 4)
    )


@pytest.mark.parametrize(
    "attr, value",
    [
        ("backend", None),
        ("name", "unnamed"),
        ("label", "unnamed"),
        ("unit", ""),
    ],
    ids=["backend", "name", "label", "unit"],
)
def test_GridDataset_attr_from_array_init(TestGridBackend, attr, value):
    """Parameterize check for different props from random array"""
    ds = GridDataset(np.random.random_sample((4, 5, 6)))
    assert getattr(ds, attr) == value


_GridDataset_getitem_tests = {}
_GridDataset_getitem_tests["1d, single time step, [int]"] = (
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
)
_GridDataset_getitem_tests["1d, single time step, [:]"] = (
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
)
_GridDataset_getitem_tests["1d, single time step, [range]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [int]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [:]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [range]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [int, int]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [:, int]"] = (
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
)
_GridDataset_getitem_tests["1d, multiple time steps, [int, :]"] = (
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
)
_GridDataset_getitem_tests["2d, multiple time steps, [:, :, :]"] = (
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
)
_GridDataset_getitem_tests["2d, multiple time steps, [int, :, :]"] = (
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
)
_GridDataset_getitem_tests["2d, multiple time steps, [:, int, :]"] = (
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
)
_GridDataset_getitem_tests["2d, multiple time steps, [:, :, int]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [:, :, :, :, :]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [:, ...]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [:, ..., int]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [...]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [newaxis, ...]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [:, newaxis, ...]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [:, :, newaxis, ...]"] = (
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
)
_GridDataset_getitem_tests["4d, multiple time steps, [..., newaxis]"] = (
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
)
_GridDataset_getitem_tests["3d, single time step, [newaxis, ...]"] = (
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
)
_GridDataset_getitem_tests["3d, single time step, [:, newaxis, ...]"] = (
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
)
_GridDataset_getitem_tests["3d, single time step, [..., newaxis]"] = (
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
)


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
    [v for v in _GridDataset_getitem_tests.values()],
    ids=[k for k in _GridDataset_getitem_tests.keys()],
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
    """Check for changing the name of GridDataset"""
    grid = GridDataset(0, name="old")
    assert grid.name == "old"
    grid.name = "new"
    assert grid.name == "new"


def test_GridDataset_change_label():
    """Check for changing the label of GridDataset"""
    grid = GridDataset(0, label="old")
    assert grid.label == "old"
    grid.label = "new"
    assert grid.label == "new"


def test_GridDataset_change_unit():
    """Check for changing the unit of GridDataset"""
    grid = GridDataset(0, unit="old")
    assert grid.unit == "old"
    grid.unit = "new"
    assert grid.unit == "new"


@pytest.mark.skip
def test_GridDataset_repr():
    """Check repr is correct"""
    pass  # TODO


@pytest.mark.skip
def test_GridDataset_from_path():
    """Init GridDataset from path"""
    pass  # TODO


@pytest.mark.skip
def test_GridDataset_from_array():
    """Init GridDataset from array"""
    pass  # TODO


def test_GridDataset_basic_numerical_operations_scalar():
    """Check basic numerical operation for GridDataset"""
    arr = np.random.random_sample((4, 3, 5))
    value = float(np.random.random_sample())
    grid = GridDataset(arr[np.newaxis])

    # ensure a new object is returned
    assert (grid + value) is not grid
    assert isinstance((grid + value), GridDataset)

    np.testing.assert_array_equal(grid + value, arr + value)
    np.testing.assert_array_equal(grid - value, arr - value)
    np.testing.assert_array_equal(grid * value, arr * value)
    np.testing.assert_array_equal(grid / value, arr / value)
    np.testing.assert_array_equal(grid ** value, arr ** value)


def test_GridDataset_basic_numerical_operations_matrix():
    """Check basic numerical operation using other GridDataset"""
    arr1 = np.random.random_sample((4, 3, 5))
    grid1 = GridDataset(arr1[np.newaxis])
    arr2 = np.random.random_sample((4, 3, 5))
    grid2 = GridDataset(arr2[np.newaxis])

    # ensure a new object is returned
    assert (grid1 + grid2) is not grid1
    assert isinstance((grid1 + grid2), GridDataset)

    # operation keep shape
    assert (grid1 + grid2).shape == grid1.shape

    np.testing.assert_array_equal(grid1 + grid2, arr1 + arr2)
    np.testing.assert_array_equal(grid1 - grid2, arr1 - arr2)
    np.testing.assert_array_equal(grid1 * grid2, arr1 * arr2)
    np.testing.assert_array_equal(grid1 / grid2, arr1 / arr2)
    np.testing.assert_array_equal(grid1 ** grid2, arr1 ** arr2)


@pytest.mark.skip
def test_GridDataset_basic_numerical_operation_in_place():
    """Check if basic numerical operation can be applied in place"""
    pass  # TODO
