# -*- coding: utf-8 -*-
# from pathlib import Path
# from typing import Optional
# from typing import Union

# import numpy as np
# import pytest

# import dask.array as da
# from nata import GridDataset
# from nata.axes import Axis
# from nata.containers.GridDataset import GridDatasetAxes
# from nata.types import BasicIndexing
# from nata.types import FileLocation
# from nata.types import GridBackendType
# from nata.types import GridDatasetType
# from nata.utils.container_tools import register_backend


# def test_GridDatasetAxes_init():
#     assert GridDatasetAxes([]).mapping == {}

#     axes = [Axis([0], name="a0"), Axis([0], name="a1")]
#     assert GridDatasetAxes(axes).mapping == {i: a for i, a in enumerate(axes)}

#     axis = Axis([0], name="some_name")
#     assert isinstance(GridDatasetAxes([axis]).some_name, Axis)
#     assert GridDatasetAxes([axis]).some_name is axis


# _testCases_GridDatasetAxes_eq = {}
# _testCases_GridDatasetAxes_eq["default"] = (
#     GridDatasetAxes([]),
#     GridDatasetAxes([]),
#     True,
# )
# _testCases_GridDatasetAxes_eq["axes length"] = (
#     GridDatasetAxes([]),
#     GridDatasetAxes([Axis([0])]),
#     False,
# )
# _testCases_GridDatasetAxes_eq["axes equivalence"] = (
#     GridDatasetAxes([Axis([0], name="some")]),
#     GridDatasetAxes([Axis([1], name="some")]),
#     True,
# )
# _testCases_GridDatasetAxes_eq["time axis"] = (
#     GridDatasetAxes([], time=Axis(0)),
#     GridDatasetAxes([], time=None),
#     False,
# )

# _testCases_GridDatasetAxes_eq["time axis equivalence"] = (
#     GridDatasetAxes([], time=Axis(0)),
#     GridDatasetAxes([], time=Axis(0, name="some")),
#     False,
# )

# _testCases_GridDatasetAxes_eq["iteration axis"] = (
#     GridDatasetAxes([], iteration=Axis(0)),
#     GridDatasetAxes([], iteration=None),
#     False,
# )

# _testCases_GridDatasetAxes_eq["iteration axis equivalence"] = (
#     GridDatasetAxes([], iteration=Axis(0)),
#     GridDatasetAxes([], iteration=Axis(0, name="some")),
#     False,
# )


# @pytest.mark.parametrize(
#     "left, right, equal",
#     _testCases_GridDatasetAxes_eq.values(),
#     ids=_testCases_GridDatasetAxes_eq.keys(),
# )
# def test_GridDatasetAxes_eq(left, right, equal):
#     if equal:
#         assert left == right
#         assert not (left != right)
#     else:
#         assert left != right
#         assert not (left == right)


# def test_GridDatasetAxes_len():
#     assert len(GridDatasetAxes([])) == 0
#     assert len(GridDatasetAxes([Axis([0])])) == 1
#     assert len(GridDatasetAxes([], time=Axis(0))) == 0


# def test_GridDatasetAxes_contains():
#     assert "time" in GridDatasetAxes(())
#     assert "some_name" in GridDatasetAxes((Axis([0], name="some_name"),))
#     assert "time" in GridDatasetAxes((), time=Axis(0, name="time"))
#     assert "time" in GridDatasetAxes([Axis([0, 1], axis_dim=0, name="time")])
#     assert "iteration" in GridDatasetAxes((), iteration=Axis(0, name="iteration"))


# def test_GridDatasetAxes_raises_warning():
#     with pytest.warns(UserWarning):
#         GridDatasetAxes([Axis([0]), Axis([0])])


# def test_GridDatasetAxes_index():
#     example = GridDatasetAxes((Axis([0], name="a0"), Axis([0], name="a1")))
#     assert example.index("a0") == 0
#     assert example.index("a1") == 1
#     assert example.index("some") is None
#     assert GridDatasetAxes((), time=Axis(0, name="time")).index("time") is None


# def test_GridDatasetAxes_raises_TypeError_invalid_axis_object():
#     with pytest.raises(TypeError):
#         GridDatasetAxes((object(),))


# def test_GridDatasetAxes_raises_TypeError_invalid_time():
#     with pytest.raises(TypeError):
#         GridDatasetAxes([], time=1.0)


# def test_GridDatasetAxes_raises_TypeError_invalid_iteration():
#     with pytest.raises(TypeError):
#         GridDatasetAxes([], iteration=1)


# def test_GridDatasetAxes_raises_TypeError_invalid_0d_axis():
#     with pytest.raises(ValueError):
#         GridDatasetAxes([Axis(0)])


# def test_GridDatasetAxes_raises_ValueError():
#     with pytest.raises(ValueError):
#         GridDatasetAxes(
#             [
#                 Axis(np.arange(10).reshape(5, 2), axis_dim=1),
#                 Axis(np.arange(5), axis_dim=1),
#             ]
#         )


# def test_GridDatasetAxes_iter():
#     axes = (Axis([0], name="a0"), Axis([0], name="a1"))
#     grid_axes = GridDatasetAxes(axes)

#     for ax, expected in zip(grid_axes, axes):
#         assert ax is expected


# def test_GridDatasetAxes_getitem():
#     axes = (Axis([0], name="a"), Axis([0], name="b"))
#     grid_axes = GridDatasetAxes(axes)

#     assert grid_axes[0] is axes[0]
#     assert grid_axes[1] is axes[1]
#     assert grid_axes["a"] is axes[0]
#     assert grid_axes["b"] is axes[1]


# def test_GridDataset_span():
#     assert GridDatasetAxes(
#         [
#             Axis(range(10), name="a0", axis_dim=1),
#             Axis(range(20), name="a1", axis_dim=1),
#             Axis(range(30), name="a2", axis_dim=1),
#         ]
#     ).span == (10, 20, 30)


# @pytest.fixture(name="TestGridBackend")
# def _TestGridBackend():
#     """Fixture for returning GridBackend"""

#     class TestGridBackend:
#         """Test backend for GridDatsets

#         * Backend provides 3D random data.
#         """

#         name = "TetsGridBackend"
#         location = None

#         dataset_name = "test_dataset_name"
#         dataset_label = "test dataset label"
#         dataset_unit = "test dataset unit"

#         axes_names = ("axes0", "axes1", "axes2")
#         axes_labels = ("axes 0", "axes 1", "axes 3")
#         axes_units = ("axes unit 0", "axes unit 1", "axes unit 2")
#         axes_min = np.array([-1.0, -2.0, -3.0])
#         axes_max = np.array([0.5, 1.5, 2.5])

#         iteration = 42

#         time_step = 12.3
#         time_unit = "time unit"

#         shape = (4, 5, 6)
#         dtype = np.float

#         ndim = 3

#         def __init__(
#             self, location: Optional[FileLocation] = None, raise_on_read: bool = True,
#         ):
#             self.location = location
#             self._raise_on_read = raise_on_read

#         @staticmethod
#         def is_valid_backend(location: Union[Path, str]) -> bool:
#             return Path(location) == Path("TestGridBackend_location")

#         def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
#             if self._raise_on_read:
#                 raise IOError("Should not read any file")

#             data = np.arange(4 * 5 * 6).reshape(self.shape)
#             return data[indexing] if indexing else data

#     # ensure dummy backend is of valid type
#     assert isinstance(TestGridBackend, GridBackendType)

#     GridDataset.add_backend(TestGridBackend)
#     yield TestGridBackend
#     # teardown
#     GridDataset.remove_backend(TestGridBackend)


# def test_GridDataset_isinstance_GridDatasetType():
#     """Ensures that a GridDataset fulfills `GridDatasetType` protocol"""
#     assert isinstance(GridDataset, GridDatasetType)


# def test_GridDataset_registration(TestGridBackend):
#     """Check if fixture registers backend properly"""
#     assert TestGridBackend.name in GridDataset.get_backends()


# def test_GridDataset_default_init():
#     """Checks default init of GridDataset. Ensures some properties are proper setup"""
#     axis = Axis([0, 1, 2])
#     ds = GridDataset(np.arange(3), GridDatasetAxes([axis]))

#     assert ds.name == "unnamed"
#     assert ds.label == "unnamed"
#     assert ds.unit == ""
#     assert "time" in ds.axes
#     assert "iteration" in ds.axes
#     assert ds.axes.time is None
#     assert ds.axes.iteration is None
#     assert ds.axes[0] is axis


# def test_GridDataset_init_raises_ValueError_for_mismatched_dimensionality():
#     with pytest.raises(ValueError):
#         GridDataset(np.arange(3), GridDatasetAxes([]))


# def test_GridDataset_init_raises_ValueError_for_mismatched_shapes():
#     with pytest.raises(ValueError):
#         GridDataset(np.arange(3), GridDatasetAxes([Axis(0)]))


# def test_GridDatset_init_with_name():
#     """Checks if name property changes if name was defined"""
#     args = (np.arange(3), GridDatasetAxes([Axis(np.arange(3))]))
#     ds = GridDataset(*args, name="some_name")
#     assert ds.name == "some_name"


# def test_GridDatset_init_with_label():
#     """Checks if label property changes if label was defined"""
#     args = (np.arange(3), GridDatasetAxes([Axis(np.arange(3))]))
#     ds = GridDataset(*args, label="some label")
#     assert ds.label == "some label"


# def test_GridDatset_init_with_unit():
#     """Checks if unit property changes if unit was defined"""
#     args = (np.arange(3), GridDatasetAxes([Axis(np.arange(3))]))
#     ds = GridDataset(*args, unit="some unit")
#     assert ds.unit == "some unit"


# def test_GridDatset_init_with_backend():
#     """Checks if backend property changes if backend was defined"""
#     args = (np.arange(3), GridDatasetAxes([Axis(np.arange(3))]))
#     ds = GridDataset(*args, backend="some_backend")
#     assert ds.backend == "some_backend"


# def test_GridDatset_init_with_locations():
#     """Checks if location property changes if location was defined"""
#     args = (np.arange(3), GridDatasetAxes([Axis(np.arange(3))]))
#     ds = GridDataset(*args, locations=(Path("some/location"),))
#     assert isinstance(ds.locations, list)
#     assert len(ds.locations) == 1
#     assert ds.locations[0] == Path("some/location")


# def test_GridDatset_init_with_grid_axes():
#     """Checks if grid axes are properly set if passed to init"""
#     # additional dimension is used for temporal (t_dim, *spatial_dim)
#     ds = GridDataset(
#         np.arange(3), GridDatasetAxes([Axis.from_limits(0, 1, 3, name="a1")]),
#     )
#     assert "a1" in ds.axes
#     assert ds.axes.grid_axes is not None
#     assert isinstance(ds.axes.grid_axes, list)
#     assert len(ds.axes.grid_axes) == 1
#     assert isinstance(ds.axes.grid_axes[0], Axis)
#     assert ds.axes.grid_axes[0].shape == (3,)


# def test_GridDatset_init_with_time():
#     time = Axis([0, 1], name="time", axis_dim=0)
#     other_axis = Axis(np.arange(6).reshape((2, 3)), axis_dim=1)

#     ds = GridDataset(
#         np.random.random_sample((2, 3)), GridDatasetAxes([time, other_axis]),
#     )
#     assert "time" in ds.axes
#     assert ds.axes.time is time


# def test_GridDatset_init_with_iteration():
#     iteration = Axis([0, 1], name="iteration", axis_dim=0)
#     other_axis = Axis(np.arange(6).reshape((2, 3)), axis_dim=1)

#     ds = GridDataset(
#         np.random.random_sample((2, 3)), GridDatasetAxes([iteration, other_axis]),
#     )
#     assert "iteration" in ds.axes
#     assert ds.axes.iteration is iteration


# def test_GridDataset_from_array_default():
#     """Checks default behavior for init by class method `.from_array`"""
#     array_content = np.random.random_sample((4, 3, 5))
#     ds = GridDataset.from_array(array_content)

#     assert ds.name == "unnamed"
#     assert ds.label == "unlabeled"
#     assert ds.unit == ""

#     assert ds.backend is None
#     assert ds.locations is None

#     assert "time" in ds.axes
#     assert ds.axes.time is None
#     assert "iteration" in ds.axes
#     assert ds.axes.iteration is None
#     assert len(ds.axes) == 3
#     for i, s in enumerate(array_content.shape):
#         assert f"axis{i}" in ds.axes
#         assert f"axis{i}" == ds.axes[i].name
#         assert f"axis{i}" == ds.axes[i].label
#         np.testing.assert_equal(ds.axes[i], np.arange(s))

#     np.testing.assert_array_equal(ds, array_content)


# def test_GridDataset_from_array_with_name():
#     """Test if `.from_array` passes name through"""
#     ds = GridDataset.from_array([], name="some_name")
#     assert ds.name == "some_name"


# def test_GridDataset_from_array_with_label():
#     """Test if `.from_array` passes label through"""
#     ds = GridDataset.from_array([], label="some label")
#     assert ds.label == "some label"


# def test_GridDataset_from_array_with_unit():
#     """Test if `.from_array` passes unit through"""
#     ds = GridDataset.from_array([], unit="some unit")
#     assert ds.unit == "some unit"


# _testCases_GridDataset_from_array_with_time = {}
# _testCases_GridDataset_from_array_with_time["numpy"] = (
#     np.arange(10),
#     Axis(np.arange(10), name="time", label="time", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_time["dask"] = (
#     da.arange(10),
#     Axis(np.arange(10), name="time", label="time", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_time["iterator"] = (
#     range(10),
#     Axis(np.arange(10), name="time", label="time", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_time["Axis"] = (
#     Axis(np.arange(10)),
#     Axis(np.arange(10), name="time", label="time", unit="", axis_dim=0),
# )


# @pytest.mark.parametrize(
#     "case",
#     _testCases_GridDataset_from_array_with_time.values(),
#     ids=_testCases_GridDataset_from_array_with_time.keys(),
# )
# def test_GridDataset_from_array_with_time(case):
#     """Test if `.from_array` initializes time correctly"""
#     input_, expected = case
#     ds = GridDataset.from_array(np.arange(10), time=input_)
#     assert "time" in ds.axes
#     assert isinstance(ds.axes.time, Axis)
#     assert ds.axes.time is ds.axes[0]
#     assert ds.axes[0].is_equiv_to(expected)
#     np.testing.assert_array_equal(ds.axes.time, expected)


# _testCases_GridDataset_from_array_with_iteration = {}
# _testCases_GridDataset_from_array_with_iteration["numpy"] = (
#     np.arange(10),
#     Axis(np.arange(10), name="iteration", label="iteration", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_iteration["dask"] = (
#     da.arange(10),
#     Axis(np.arange(10), name="iteration", label="iteration", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_iteration["iterator"] = (
#     range(10),
#     Axis(np.arange(10), name="iteration", label="iteration", unit="", axis_dim=0),
# )
# _testCases_GridDataset_from_array_with_iteration["Axis"] = (
#     Axis(np.arange(10)),
#     Axis(np.arange(10), name="iteration", label="iteration", unit="", axis_dim=0),
# )


# @pytest.mark.parametrize(
#     "case",
#     _testCases_GridDataset_from_array_with_iteration.values(),
#     ids=_testCases_GridDataset_from_array_with_iteration.keys(),
# )
# def test_GridDataset_from_array_with_iteration(case):
#     """Test if `.from_array` initializes iteration correctly"""
#     input_, expected = case
#     ds = GridDataset.from_array(np.arange(10), iteration=input_)
#     assert "iteration" in ds.axes
#     assert isinstance(ds.axes.iteration, Axis)
#     assert ds.axes.iteration is ds.axes[0]
#     assert ds.axes[0].is_equiv_to(expected)
#     np.testing.assert_array_equal(ds.axes.iteration, expected)


# _testCases_GridDataset_from_array_with_grid_axes = {}
# _testCases_GridDataset_from_array_with_grid_axes["numpy"] = (
#     [np.arange(3), np.arange(4)],
#     [
#         Axis(np.arange(3), name="axis0", label="axis 0", axis_dim=1),
#         Axis(np.arange(4), name="axis1", label="axis 1", axis_dim=1),
#     ],
# )
# _testCases_GridDataset_from_array_with_grid_axes["dask"] = (
#     [da.arange(3), da.arange(4)],
#     [
#         Axis(np.arange(3), name="axis0", label="axis 0", axis_dim=1),
#         Axis(np.arange(4), name="axis1", label="axis 1", axis_dim=1),
#     ],
# )
# _testCases_GridDataset_from_array_with_grid_axes["iterator"] = (
#     [range(3), range(4)],
#     [
#         Axis(np.arange(3), name="axis0", label="axis 0", axis_dim=1),
#         Axis(np.arange(4), name="axis1", label="axis 1", axis_dim=1),
#     ],
# )
# _testCases_GridDataset_from_array_with_grid_axes["Axis"] = (
#     [
#         Axis(
#             np.arange(3),
#             name="a0",
#             label="some axis",
#             unit="some_unit",
#             axis_dim=1
#         ),
#         Axis(
#             np.arange(4),
#             name="a1",
#             label="some other axis",
#             unit="some other unit",
#             axis_dim=1,
#         ),
#     ],
#     [
#         Axis(
#             np.arange(3),
#             name="a0",
#             label="some axis",
#             unit="some_unit",
#             axis_dim=1),
#         Axis(
#             np.arange(4),
#             name="a1",
#             label="some other axis",
#             unit="some other unit",
#             axis_dim=1,
#         ),
#     ],
# )


# @pytest.mark.parametrize(
#     "case",
#     _testCases_GridDataset_from_array_with_grid_axes.values(),
#     ids=_testCases_GridDataset_from_array_with_grid_axes.keys(),
# )
# def test_GridDataset_from_array_with_grid_axes(case):
#     """Test if `.from_array` initializes iteration correctly"""
#     input_, expected = case
#     ds = GridDataset.from_array(np.arange(3 * 4).reshape((3, 4)), grid_axes=input_)

#     for actual_axis, expected_axis in zip(ds.axes.grid_axes, expected):
#         assert isinstance(actual_axis, Axis)
#         assert expected_axis.is_equiv_to(actual_axis)
#         np.testing.assert_array_equal(actual_axis, expected_axis)


# @pytest.fixture(name="backend_on_disk")
# def _create_backend_file(tmp_path):
#     @register_backend(GridDataset)
#     class DummyBackend:
#         valid_paths = set()
#         name = "DummyBackend"
#         location: Optional[str] = None
#         dataset_name = "dummy_name"
#         dataset_label = "dummy label"
#         dataset_unit = "dummy unit"
#         iteration = 123
#         time_step = -12314.0
#         time_unit = "time unit"
#         shape = (2, 5)
#         ndim = 2
#         axes_min = (-123.0, 321.0)
#         axes_max = (346.0, 898.0)
#         axes_names = ("dummy_axis_0", "dummy_axis_1")
#         axes_labels = ("dummy axis 0", "dummy axis 1")
#         axes_units = ("axis0 unit", "axis1 unit")
#         dtype = np.int

#         def __init__(self, location: str) -> None:
#             self.location = location

#         def get_data(self, indexing):
#             raise NotImplementedError

#         @classmethod
#         def is_valid_backend(cls, path: Path) -> bool:
#             return True if path in cls.valid_paths else False

#     def create_file(filename: str):
#         f = tmp_path / filename
#         f.touch()
#         DummyBackend.valid_paths |= {f}
#         return f

#     create_file.root_path = tmp_path
#     create_file.backend_info = DummyBackend

#     return create_file


# def test_GridDataset_get_valid_backend(backend_on_disk):
#     test_file = backend_on_disk("test")
#     assert GridDataset.get_valid_backend(test_file)


# def test_GridDataset_from_file_with_single_file_path(backend_on_disk):
#     # create backend file
#     backend_on_disk("test")

#     ds = GridDataset.from_path(backend_on_disk.root_path / "test")

#     # temporal info
#     assert ds.temporal_steps == 1

#     # check for grid properties
#     assert len(ds) == 1
#     assert ds.shape == backend_on_disk.backend_info.shape
#     assert ds.grid_shape == backend_on_disk.backend_info.shape
#     assert ds.ndim == backend_on_disk.backend_info.ndim
#     assert ds.grid_ndim == backend_on_disk.backend_info.ndim
#     assert ds.dtype == backend_on_disk.backend_info.dtype

#     # check propagation of namings
#     assert ds.name == backend_on_disk.backend_info.dataset_name
#     assert ds.label == backend_on_disk.backend_info.dataset_label
#     assert ds.unit == backend_on_disk.backend_info.dataset_unit

#     # check axes init
#     assert isinstance(ds.axes.iteration, Axis)
#     assert isinstance(ds.axes.time, Axis)
#     assert all(isinstance(ax, Axis) for ax in ds.axes.grid_axes)


# def test_GridDataset_from_file_with_multiple_files(backend_on_disk):
#     # create backend file
#     for i in range(10):
#         backend_on_disk(f"test{i}")

#     ds = GridDataset.from_path(backend_on_disk.root_path / "*")

#     # temporal info
#     assert ds.temporal_steps == 10

#     # check for grid properties
#     assert len(ds) == 10
#     assert ds.shape == (10,) + backend_on_disk.backend_info.shape
#     assert ds.grid_shape == backend_on_disk.backend_info.shape
#     assert ds.ndim == backend_on_disk.backend_info.ndim + 1
#     assert ds.grid_ndim == backend_on_disk.backend_info.ndim
#     assert ds.dtype == backend_on_disk.backend_info.dtype

#     # check propagation of namings
#     assert ds.name == backend_on_disk.backend_info.dataset_name
#     assert ds.label == backend_on_disk.backend_info.dataset_label
#     assert ds.unit == backend_on_disk.backend_info.dataset_unit

#     # check axes init
#     assert isinstance(ds.axes.iteration, Axis)
#     assert isinstance(ds.axes.time, Axis)
#     assert all(isinstance(ax, Axis) for ax in ds.axes.grid_axes)


# def test_GridDataset___array__():
#     ds = GridDataset.from_array(np.arange(10).reshape((1, 10)))
#     assert ds.shape == (1, 10)
#     assert ds.grid_shape == (1, 10)
#     np.testing.assert_array_equal(ds, np.arange(10).reshape((1, 10)))

#     ds = GridDataset.from_array(np.arange(10))
#     assert ds.shape == (10,)
#     assert ds.grid_shape == (10,)
#     np.testing.assert_array_equal(ds, np.arange(10))


# _ufunc_test_cases = [
#     (np.add, (10,), {}),
#     (np.subtract, (10,), {}),
#     (np.multiply, (10,), {}),
#     # (np.matmul, ),
#     (np.divide, (10,), {}),
#     (np.logaddexp, (10,), {}),
#     (np.logaddexp, (10,), {}),
#     (np.true_divide, (10,), {}),
#     (np.floor_divide, (10,), {}),
#     (np.negative, (), {}),
#     (np.positive, (), {}),
#     (np.power, (10,), {}),
#     (np.float_power, (10,), {}),
#     (np.remainder, (10,), {}),
#     (np.mod, (10,), {}),
#     (np.fmod, (10,), {}),
#     # (np.divmod, (10,), {}),
#     (np.absolute, (), {}),
#     (np.fabs, (), {}),
#     (np.rint, (), {}),
#     (np.sign, (), {}),
#     (np.heaviside, (10,), {}),
#     (np.conj, (), {}),
#     (np.conjugate, (), {}),
#     (np.exp, (), {}),
#     (np.exp2, (), {}),
#     (np.log, (), {}),
#     (np.log2, (), {}),
#     (np.log10, (), {}),
#     (np.expm1, (), {}),
#     (np.log1p, (), {}),
#     (np.sqrt, (), {}),
#     (np.square, (), {}),
#     (np.cbrt, (), {}),
#     (np.reciprocal, (), {}),
#     # (np.gcd, (10,), {}),
#     # (np.lcm, (10,), {}),
#     (np.sin, (), {}),
#     (np.cos, (), {}),
#     (np.tan, (), {}),
#     (np.arcsin, (), {}),
#     (np.arccos, (), {}),
#     (np.arctan, (), {}),
#     (np.arctan2, (10,), {}),
#     (np.hypot, (10,), {}),
#     (np.sinh, (), {}),
#     (np.cosh, (), {}),
#     (np.tanh, (), {}),
#     (np.arcsinh, (), {}),
#     (np.arccosh, (), {}),
#     (np.arctanh, (), {}),
#     (np.degrees, (), {}),
#     (np.radians, (), {}),
#     (np.deg2rad, (), {}),
#     (np.rad2deg, (), {}),
#     # (np.bitwise_and, (10,), {}),
#     # (np.bitwise_or, (), {}),
#     # (np.bitwise_xor, (10,), {}),
#     # (np.invert, (), {}),
#     # (np.left_shift, (10,), {}),
#     # (np.right_shift, (10,), {}),
#     (np.greater, (10,), {}),
#     (np.greater_equal, (10,), {}),
#     (np.less, (10,), {}),
#     (np.less_equal, (10,), {}),
#     (np.not_equal, (10,), {}),
#     (np.equal, (10,), {}),
#     (np.logical_and, (10,), {}),
#     (np.logical_or, (10,), {}),
#     (np.logical_xor, (10,), {}),
#     (np.logical_not, (), {}),
#     (np.maximum, (10,), {}),
#     (np.minimum, (10,), {}),
#     (np.fmax, (10,), {}),
#     (np.fmin, (10,), {}),
#     (np.isfinite, (), {}),
#     (np.isinf, (), {}),
#     (np.isnan, (), {}),
#     # (np.isnat, (), {}),
#     (np.signbit, (), {}),
#     (np.copysign, (10,), {}),
#     (np.nextafter, (10,), {}),
#     (np.spacing, (), {}),
#     # (np.modf, (), {}),
#     (np.ldexp, (10,), {}),
#     # (np.frexp, (), {}),
#     (np.floor, (), {}),
#     (np.ceil, (), {}),
#     (np.trunc, (), {}),
# ]


# @pytest.mark.parametrize(
#     "func, args, kwargs",
#     _ufunc_test_cases,
#     ids=[str(case[0].__name__) for case in _ufunc_test_cases],
# )
# @pytest.mark.filterwarnings("ignore:invalid value")
# def test_GridDAtaset___array_ufunc__(func, args, kwargs):
#     ds = GridDataset.from_array(np.linspace(0.1, 0.9))
#     np.testing.assert_array_equal(
#         func(ds, *args, **kwargs), func(np.linspace(0.1, 0.9), *args, **kwargs)
#     )


# _array_function_test_cases = [
#     (np.fft.fft, (), {}),
# ]


# @pytest.mark.parametrize(
#     "func, args, kwargs",
#     _array_function_test_cases,
#     ids=[str(case[0].__name__) for case in _array_function_test_cases],
# )
# @pytest.mark.filterwarnings("ignore:invalid value")
# def test_GridDAtaset___array_function__(func, args, kwargs):
#     ds = GridDataset.from_array(np.linspace(0.1, 0.9))
#     np.testing.assert_array_equal(
#         func(ds, *args, **kwargs), func(np.linspace(0.1, 0.9), *args, **kwargs)
#     )


# _testCases_GridDataset_getitem = {}
# _testCases_GridDataset_getitem["(axis,) [int]"] = {
#     "arr": np.arange(123),
#     "indexing": np.s_[2],
#     "expected_arr": np.array(2),
# }
# _testCases_GridDataset_getitem["(axis,) [:]"] = {
#     "arr": np.arange(123),
#     "indexing": np.s_[:],
#     "expected_arr": np.arange(123),
# }
# _testCases_GridDataset_getitem["(axis,) [range]"] = {
#     "arr": np.arange(123),
#     "indexing": np.s_[8:-12],
#     "expected_arr": np.arange(123)[8:-12],
# }
# _testCases_GridDataset_getitem["(axis,) [newaxis]"] = {
#     "arr": np.arange(123),
#     "indexing": np.s_[np.newaxis],
#     "expected_arr": np.arange(123).reshape((1, 123)),
#     "expected_kwargs": {
#         "grid_axes": [Axis([0], name="newaxis0"), Axis(np.arange(123), name="axis0")]
#     },
# }
# _testCases_GridDataset_getitem["(axis,) [..., newaxis]"] = {
#     "arr": np.arange(123),
#     "indexing": np.s_[..., np.newaxis],
#     "expected_arr": np.arange(123).reshape((123, 1)),
#     "expected_kwargs": {
#         "grid_axes": [Axis(np.arange(123), name="axis0"), Axis([0], name="newaxis0")]
#     },
# }
# _testCases_GridDataset_getitem["(time, axis) [int]"] = {
#     "arr": np.arange(4 * 5).reshape((4, 5)),
#     "kwargs": {
#         "grid_axes": [
#             Axis(np.arange(4), name="time", label="time", axis_dim=0),
#             Axis(
#                 np.arange(5).reshape((1, 5)).repeat(4, axis=0),
#                 name="axis0",
#                 label="axis0",
#                 axis_dim=1,
#             ),
#         ]
#     },
#     "indexing": np.s_[2],
#     "expected_arr": np.arange(4 * 5).reshape((4, 5))[2],
#     "expected_kwargs": {
#         "time": Axis(2, name="time", label="time", axis_dim=0),
#         "grid_axes": [Axis(np.arange(5), name="axis0", label="axis0", axis_dim=1)],
#     },
# }
# _testCases_GridDataset_getitem["(time, axis) [newaxis]"] = {
#     "arr": np.arange(4 * 5).reshape((4, 5)),
#     "kwargs": {
#         "grid_axes": [
#             Axis(np.arange(4), name="time", label="time", axis_dim=0),
#             Axis(
#                 np.arange(5).reshape((1, 5)).repeat(4, axis=0),
#                 name="axis0",
#                 label="axis0",
#                 axis_dim=1,
#             ),
#         ]
#     },
#     "indexing": np.s_[np.newaxis],
#     "expected_arr": np.arange(4 * 5).reshape((1, 4, 5)),
#     "expected_kwargs": {
#         "grid_axes": [
#             Axis(
#                 np.arange(1).reshape((1, 1)).repeat(4, axis=0),
#                 name="newaxis0",
#                 axis_dim=1,
#             ),
#             Axis(np.arange(4), name="time", label="time", axis_dim=0),
#             Axis(
#                 np.arange(5).reshape((1, 5)).repeat(4, axis=0),
#                 name="axis0",
#                 label="axis0",
#                 axis_dim=1,
#             ),
#         ],
#     },
# }
# _testCases_GridDataset_getitem["(iteration, axis) [int]"] = {
#     "arr": np.arange(4 * 5).reshape((4, 5)),
#     "kwargs": {
#         "grid_axes": [
#             Axis(np.arange(4), name="iteration", label="iteration", axis_dim=0),
#             Axis(
#                 np.arange(4 * 5).reshape((4, 5)),
#                 name="axis0",
#                 label="axis0",
#                 axis_dim=1,
#             ),
#         ]
#     },
#     "indexing": np.s_[2],
#     "expected_arr": np.arange(4 * 5).reshape((4, 5))[2],
#     "expected_kwargs": {
#         "iteration": Axis(2, name="iteration", label="iteration", axis_dim=0),
#         "grid_axes": [
#             Axis(
#                 np.arange(4 * 5).reshape((4, 5))[2],
#                 name="axis0",
#                 label="axis0",
#                 axis_dim=1,
#             ),
#         ],
#     },
# }


# @pytest.mark.parametrize(
#     "case",
#     _testCases_GridDataset_getitem.values(),
#     ids=_testCases_GridDataset_getitem.keys(),
# )
# def test_GridDataset_getitem(case):
#     arr = case["arr"]
#     kwargs = case["kwargs"] if "kwargs" in case else {}

#     indexing = case["indexing"]

#     expected_arr = case["expected_arr"]
#     expected_args = case["expected_args"] if "expected_args" in case else ()
#     expected_kwargs = case["expected_kwargs"] if "expected_kwargs" in case else {}

#     grid = GridDataset.from_array(arr, **kwargs)
#     subgrid = grid[indexing]
#     expected_subgrid = GridDataset.from_array(
#         expected_arr, *expected_args, **expected_kwargs
#     )

#     assert isinstance(subgrid, GridDataset)
#     assert subgrid.is_equiv_to(expected_subgrid)
#     np.testing.assert_array_equal(subgrid, expected_subgrid)

#     if expected_subgrid.axes.time:
#         assert subgrid.axes.time.is_equiv_to(expected_subgrid.axes.time, verbose=True)
#         np.testing.assert_array_equal(subgrid.axes.time, expected_subgrid.axes.time)
#     else:
#         assert subgrid.axes.time is None

#     if expected_subgrid.axes.iteration:
#         assert subgrid.axes.iteration.is_equiv_to(
#             expected_subgrid.axes.iteration, verbose=True
#         )
#         np.testing.assert_array_equal(
#             subgrid.axes.iteration, expected_subgrid.axes.iteration
#         )
#     else:
#         assert subgrid.axes.iteration is None
