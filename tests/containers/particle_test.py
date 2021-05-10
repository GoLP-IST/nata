# -*- coding: utf-8 -*-
from operator import eq
from operator import ne
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import dask.array as da
import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured
from numpy.typing import ArrayLike

from nata.containers.axis import Axis
from nata.containers.particle import Particle
from nata.containers.particle import ParticleArray
from nata.containers.particle import ParticleDataReader
from nata.containers.particle import ParticleDataset
from nata.containers.particle import Quantity
from nata.containers.particle import QuantityArray
from nata.containers.particle import expand_and_stack
from nata.containers.particle import expand_arr
from nata.containers.utils import register_backend


@pytest.fixture(name="path_to_particle_files")
def _path_to_particle_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # make checking for 'is_valid_backend' always succeed.
    # ensures that 'any' backend can be added, even if properties defined in '__init__'
    monkeypatch.setattr(ParticleArray, "is_valid_backend", lambda _: True)

    # make 'basedir'
    basedir = tmp_path / "particle_files"
    basedir.mkdir(parents=True, exist_ok=True)

    # create dummy files
    dummy_data = unstructured_to_structured(np.arange(32, dtype=float).reshape((8, 4)))
    np.savetxt(basedir / "prt.0", dummy_data, delimiter=",")

    dummy_data = unstructured_to_structured(np.arange(64, dtype=float).reshape((16, 4)))
    np.savetxt(basedir / "prt.1", dummy_data, delimiter=",")

    dummy_data = unstructured_to_structured(np.empty((0, 4), dtype=float))
    np.savetxt(basedir / "prt.2", dummy_data, delimiter=",")

    @register_backend(ParticleArray)
    class Dummy_ParticleFile:
        name = "dummy_backend"

        def __init__(self, location: Union[str, Path]) -> None:
            self.location = Path(location)
            # capture UserWarnings when opening empty file
            with pytest.warns(None):
                self.data = np.loadtxt(location, dtype=dummy_data.dtype, delimiter=",")
            self.dataset_name = "dummy_prt"
            self.dataset_label = "dummy prt label"

            self.num_particles = self.data.shape[-1]

            self.quantity_names = tuple(f"f{i}" for i in range(4))
            self.quantity_labels = tuple(f"f{i} label" for i in range(4))
            self.quantity_units = tuple(f"f{i} unit" for i in range(4))

            self.ndim = self.data.ndim
            self.shape = self.data.shape
            self.dtype = self.data.dtype

            self.iteration = int(self.location.suffix[1:])
            self.time_step = float(self.location.suffix[1:])
            self.time_unit = "dummy time unit"

        @staticmethod
        def is_valid_backend(location: Union[str, Path]) -> bool:
            location = Path(location)
            if location.stem == "prt" and location.suffix in (".0", ".1", ".2"):
                return True
            else:
                return False

        def get_data(
            self,
            indexing: Optional[Any] = None,
            fields: Optional[Union[str, Sequence[str]]] = None,
        ) -> ArrayLike:
            return self.data[indexing]

    assert isinstance(Dummy_ParticleFile(basedir / "prt.0"), ParticleDataReader)
    assert isinstance(Dummy_ParticleFile(basedir / "prt.1"), ParticleDataReader)
    assert isinstance(Dummy_ParticleFile(basedir / "prt.2"), ParticleDataReader)

    yield basedir
    # remove backend from ParticleArray -> avoids collision with other backends
    ParticleArray.remove_backend(Dummy_ParticleFile)


def test_Quantity():
    quant = Quantity.from_array(123)

    assert quant.name == "unnamed"
    assert quant.label == "unlabeled"
    assert quant.unit == ""

    assert quant.count == 1

    assert quant.time.name == "time"
    assert quant.time.label == "time"
    assert quant.time.unit == ""
    assert quant.time.to_numpy() == 0.0

    assert quant.shape == ()
    assert quant.ndim == 0
    assert quant.dtype == int

    np.testing.assert_array_equal(quant, 123)


def test_QuantityArray():
    quantArr = QuantityArray.from_array([1, 2, 3])

    assert quantArr.name == "unnamed"
    assert quantArr.label == "unlabeled"
    assert quantArr.unit == ""

    assert quantArr.count == 3

    assert quantArr.time.name == "time"
    assert quantArr.time.label == "time"
    assert quantArr.time.unit == ""
    assert quantArr.time.to_numpy() == 0.0

    assert quantArr.shape == (3,)
    assert quantArr.ndim == 1
    assert quantArr.dtype == int

    np.testing.assert_array_equal(quantArr, [1, 2, 3])


def test_Quantity_raise_non_0d():
    with pytest.raises(ValueError, match="only 0d data is supported"):
        Quantity.from_array([123])


def test_Quantity_raise_when_fields_represent():
    with pytest.raises(ValueError, match="only unstructured data types supported"):
        Quantity.from_array(np.array(0, dtype=[("f1", int)]))


@pytest.mark.skip
def test_Quantity_getitem():
    pass


@pytest.mark.parametrize(
    "left, right, operation",
    [
        # general
        (Quantity.from_array(1), Quantity.from_array(1), eq),
        # content
        (Quantity.from_array(1), Quantity.from_array(2), eq),
        # name
        (Quantity.from_array(1), Quantity.from_array(1, name="some"), ne),
        # label
        (Quantity.from_array(1), Quantity.from_array(1, label="some"), ne),
        # unit
        (Quantity.from_array(1), Quantity.from_array(1, unit="some"), ne),
        # time value
        (Quantity.from_array(1, time=0), Quantity.from_array(1, time=1), eq),
        # time name
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, name="some")),
            ne,
        ),
        # time label
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, label="some")),
            ne,
        ),
        # time unit
        (
            Quantity.from_array(1, time=Axis.from_array(0)),
            Quantity.from_array(1, time=Axis.from_array(0, unit="some")),
            ne,
        ),
    ],
    ids=(
        "general",
        "content",
        "name",
        "label",
        "unit",
        "time",
        "time.name",
        "time.label",
        "time.unit",
    ),
)
def test_Quantity_hash(left: Quantity, right: Quantity, operation: Callable):
    assert operation(hash(left), hash(right))


def test_Quantity_repr():
    quant = Quantity.from_array(np.array(0, dtype=np.int64))
    assert repr(quant) == "Quantity<name=unnamed, dtype=int64, time=0.0>"


def test_Quantity_repr_html():
    grid = Quantity.from_array(np.array(0, dtype=np.int64))
    expected_markdown = """
    | **Quantity** | |
    | ---: | :--- |
    | **name**  | unnamed |
    | **label** | unlabeled |
    | **unit**  | '' |
    | **count** | 1 |
    | **dtype** | int64 |
    | **time**  | 0.0 |

    """

    assert grid._repr_markdown_() == dedent(expected_markdown)


def test_Particle():
    prt = Particle.from_array(0.0)
    expected_dtype = np.dtype([("quant1", float)])

    assert prt.name == "unnamed"
    assert prt.label == "unlabeled"

    assert prt.time.name == "time"
    assert prt.time.label == "time"
    assert prt.time.unit == ""
    assert prt.time.to_numpy() == 0.0

    assert prt.count == 1
    assert prt.quantities == (("quant1", "quant1 label", ""),)
    assert prt.quantity_names == ("quant1",)
    assert prt.quantity_labels == ("quant1 label",)
    assert prt.quantity_units == ("",)

    assert prt.ndim == 0
    assert prt.shape == ()
    assert prt.dtype == expected_dtype

    np.testing.assert_array_equal(prt, np.array(0.0, expected_dtype))


@pytest.mark.parametrize(
    "arr, kwargs, quantities, expected_arr",
    [
        (
            1.0,
            {},
            (("quant1", "quant1 label", ""),),
            np.array(1.0, dtype=np.dtype([("quant1", float)])),
        ),
        (
            np.array(1.0, dtype=np.dtype([("x", float)])),
            {},
            (("x", "x", ""),),
            np.array(1.0, dtype=np.dtype([("x", float)])),
        ),
        (
            np.array([1, 2, 3, 4]),
            {
                "quantities": tuple(
                    (f"q{i}", f"q_{i} label", f"q{i} unit") for i in range(4)
                ),
            },
            tuple((f"q{i}", f"q_{i} label", f"q{i} unit") for i in range(4)),
            np.array((1, 2, 3, 4), dtype=np.dtype([(f"q{i}", int) for i in range(4)])),
        ),
    ],
    ids=["number -> 1 quantity", "structured array", "array -> n quantities"],
)
def test_Particle_from_array_various_inputs(arr, kwargs, quantities, expected_arr):
    prt = Particle.from_array(arr, **kwargs)

    assert prt.count == 1
    assert prt.ndim == 0
    assert prt.shape == ()

    assert prt.quantities == quantities

    np.testing.assert_array_equal(prt, expected_arr)


def test_ParticleArray():
    prt = ParticleArray.from_array([0.0, 1.0, 2.0])
    expected_dtype = np.dtype([("quant1", float)])

    assert prt.name == "unnamed"
    assert prt.label == "unlabeled"

    assert prt.time.name == "time"
    assert prt.time.label == "time"
    assert prt.time.unit == ""
    assert prt.time.to_numpy() == 0.0

    assert prt.count == 3
    assert prt.quantities == (("quant1", "quant1 label", ""),)
    assert prt.quantity_names == ("quant1",)
    assert prt.quantity_labels == ("quant1 label",)
    assert prt.quantity_units == ("",)

    assert prt.ndim == 1
    assert prt.shape == (3,)
    assert prt.dtype == expected_dtype

    np.testing.assert_array_equal(prt, np.array([0.0, 1.0, 2.0], expected_dtype))


def test_ParticleArray_from_path(path_to_particle_files: Path):
    particleArray_file = path_to_particle_files / "prt.0"

    prt_arr = ParticleArray.from_path(particleArray_file)
    expected_arr = unstructured_to_structured(
        np.arange(32, dtype=float).reshape((8, 4))
    )

    # information stored in fixture for dummy particle backend
    assert prt_arr.name == "dummy_prt"
    assert prt_arr.label == "dummy prt label"

    assert prt_arr.time.name == "time"
    assert prt_arr.time.label == "time"
    assert prt_arr.time.unit == "dummy time unit"
    assert prt_arr.time.to_numpy() == 0.0

    assert prt_arr.count == 8
    assert prt_arr.quantities == tuple(
        (f"f{i}", f"f{i} label", f"f{i} unit") for i in range(4)
    )
    assert prt_arr.quantity_names == tuple(f"f{i}" for i in range(4))
    assert prt_arr.quantity_labels == tuple(f"f{i} label" for i in range(4))
    assert prt_arr.quantity_units == tuple(f"f{i} unit" for i in range(4))

    assert prt_arr.ndim == 1
    assert prt_arr.shape == (8,)
    assert prt_arr.dtype == expected_arr.dtype

    np.testing.assert_array_equal(prt_arr, expected_arr)


def test_ParticleDataset():
    # passed array has to fulfill following shape requirements
    # array.shape == (t_steps, n_particles, n_quantities)
    #
    #           `t_step`: number of time steps, REQUIRED
    #      `n_particles`: number of particles, REQUIRED
    #     `n_quantities`: number of quantities, OPTIONAL

    prt_ds = ParticleDataset.from_array([[1, 2, 3], [3, 4, 5]])
    expected_dtype = np.dtype([("quant1", int)])
    expected_arr = unstructured_to_structured(
        np.array([[1, 2, 3], [3, 4, 5]])[..., np.newaxis], dtype=expected_dtype
    )

    assert prt_ds.name == "unnamed"
    assert prt_ds.label == "unlabeled"

    assert prt_ds.time.name == "time"
    assert prt_ds.time.label == "time"
    assert prt_ds.time.unit == ""
    np.testing.assert_array_equal(prt_ds.time, [0, 1])

    assert prt_ds.count == 3
    assert prt_ds.quantities == (("quant1", "quant1 label", ""),)
    assert prt_ds.quantity_names == ("quant1",)
    assert prt_ds.quantity_labels == ("quant1 label",)
    assert prt_ds.quantity_units == ("",)

    assert prt_ds.ndim == 2
    assert prt_ds.shape == (2, 3)
    assert prt_ds.dtype == expected_dtype

    np.testing.assert_array_equal(prt_ds, expected_arr)


@pytest.mark.parametrize(
    "left, right, operation",
    [
        # general
        (ParticleDataset.from_array([[1]]), ParticleDataset.from_array([[1]]), eq),
        # content
        (ParticleDataset.from_array([[1]]), ParticleDataset.from_array([[2]]), eq),
        # name
        (
            ParticleDataset.from_array([[1]]),
            ParticleDataset.from_array([[1]], name="some"),
            ne,
        ),
        # label
        (
            ParticleDataset.from_array([[1]]),
            ParticleDataset.from_array([[1]], label="some"),
            ne,
        ),
        # time value
        (
            ParticleDataset.from_array([[1]], time=[0]),
            ParticleDataset.from_array([[1]], time=[1]),
            eq,
        ),
        # time name
        (
            ParticleDataset.from_array([[1]], time=Axis.from_array([0])),
            ParticleDataset.from_array([[1]], time=Axis.from_array([0], name="some")),
            ne,
        ),
        # time label
        (
            ParticleDataset.from_array([[1]], time=Axis.from_array([0])),
            ParticleDataset.from_array([[1]], time=Axis.from_array([0], label="some")),
            ne,
        ),
        # time unit
        (
            ParticleDataset.from_array([[1]], time=Axis.from_array([0])),
            ParticleDataset.from_array([[1]], time=Axis.from_array([0], unit="some")),
            ne,
        ),
    ],
    ids=(
        "general",
        "content",
        "name",
        "label",
        "time",
        "time.name",
        "time.label",
        "time.unit",
    ),
)
def test_ParticleDataset_hash(
    left: ParticleDataset,
    right: ParticleDataset,
    operation: Callable,
):
    assert operation(hash(left), hash(right))


def test_ParticleDataset_repr():
    prt_ds = ParticleDataset.from_array(np.arange(32, dtype=np.int64).reshape((4, 8)))
    assert repr(prt_ds) == (
        "ParticleDataset<"
        "name=unnamed, "
        "dtype=[('quant1', '<i8')], "
        "quantities=('quant1',), "
        "time=Axis<name='time', label='time', unit=''>"
        ">"
    )


def test_ParticleDataset_repr_html():
    grid = ParticleDataset.from_array(np.arange(32, dtype=np.int64).reshape((4, 8)))
    expected_markdown = """
    | **ParticleDataset** | |
    | ---: | :--- |
    | **name**       | unnamed |
    | **label**      | unlabeled |
    | **count**      | 8 |
    | **shape**      | (4, 8) |
    | **dtype**      | [('quant1', '<i8')] |
    | **quantities** | ('quant1',) |
    | **time**       | Axis<name='time', label='time', unit=''> |

    """

    assert grid._repr_markdown_() == dedent(expected_markdown)


def test_GridDataset_from_path(path_to_particle_files: Path):
    prt_ds = ParticleDataset.from_path(path_to_particle_files / "*")

    assert prt_ds.name == "dummy_prt"
    assert prt_ds.label == "dummy prt label"

    assert prt_ds.count == 16

    assert prt_ds.quantity_names == tuple(f"f{i}" for i in range(4))
    assert prt_ds.quantity_labels == tuple(f"f{i} label" for i in range(4))
    assert prt_ds.quantity_units == tuple(f"f{i} unit" for i in range(4))

    assert prt_ds.ndim == 2
    assert prt_ds.shape == (3, 16)


def test_GridDataset_variable_prt_number():
    prt_ds = ParticleDataset.from_array([[1, 2, 3], [3, 5]])
    assert prt_ds.shape == (2, 3)


def test_expand_to_array():
    arr = expand_and_stack([[1, 2, 3, 4], [5, 6], []])
    assert isinstance(arr, da.Array)
    assert arr.shape == (3, 4)
    assert np.sum(np.arange(1, 7)) == np.sum(arr).compute()


@pytest.mark.parametrize(
    "arr, required_shape, has_mask",
    [
        (da.arange(12), (32,), True),
        (da.arange(12), (12,), False),
    ],
    ids=[
        "(12,) -> (32,)",
        "(12,) -> (12,)",
    ],
)
def test_expand_arr(arr: da.Array, required_shape: Tuple[int, ...], has_mask: bool):
    output = expand_arr(arr, required_shape)

    assert isinstance(output, da.Array)
    assert output.shape == required_shape
    assert isinstance(output.compute(), np.ma.MaskedArray)
    assert output.compute().mask.any() == has_mask
    # masked arrays have to keep same values -> testing by using aggregation
    assert np.sum(output).compute() == np.sum(arr).compute()


@pytest.mark.parametrize(
    "init_arr, index, expected_type",
    [
        (
            unstructured_to_structured(np.arange(8 * 5 * 4).reshape((8, 5, 4))),
            np.s_[1:5],
            ParticleDataset,
        ),
        (
            unstructured_to_structured(np.arange(8 * 5 * 4).reshape((8, 5, 4))),
            np.s_[1],
            ParticleArray,
        ),
        (
            unstructured_to_structured(np.arange(8 * 5 * 4).reshape((8, 5, 4))),
            np.s_[2, 1:3],
            ParticleArray,
        ),
        (
            unstructured_to_structured(np.arange(8 * 5 * 4).reshape((8, 5, 4))),
            np.s_[2, 3],
            Particle,
        ),
    ],
    ids=[
        "t1:t2",
        "t1",
        "t1, i1:i2",
        "t1, i1",
    ],
)
def test_ParticleDataset_getitem_perserve_quants(
    init_arr: np.ndarray, index: Any, expected_type: type
):
    prt_ds = ParticleDataset.from_array(
        init_arr,
        name="some_name",
        label="some label",
    )
    selection = prt_ds[index]

    assert selection.name == prt_ds.name
    assert selection.label == prt_ds.label

    assert selection.quantities == prt_ds.quantities
    assert isinstance(selection, expected_type)

    # get data from masked array
    np.testing.assert_array_equal(selection.to_numpy().data, init_arr[index])
