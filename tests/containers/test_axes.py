import pytest
from pytest import raises
import numpy as np

from nata.containers.axes import Axis
from nata.containers.axes import IterationAxis
from nata.containers.axes import TimeAxis
from nata.containers.axes import GridAxis
from nata.containers.axes import DataStock

from ..conftest import does_not_raise


@pytest.fixture(name="Parent")
def _dummy_parent():
    class Parent:
        def __init__(self, iterations=[]):
            self._iterations = iterations

        def iterations_to_keep(self, to_keep):
            self._iterations = [k for k in self._iterations if k in to_keep]

    yield Parent


@pytest.mark.parametrize(
    "key, value, label, unit, exception",
    (
        (1, 100, "some label", "some unit", does_not_raise()),
        (None, 100, "some label", "some unit", does_not_raise()),
    ),
)
def test_Axis_init(Parent, key, value, label, unit, exception):
    parent = Parent()
    with exception:
        axis = Axis(parent=parent, key=key, value=value, label=label, unit=unit)
        assert axis._parent is parent
        assert axis._mapping == {key: value}
        assert axis.label == label
        assert axis.unit == unit
        assert axis._dtype == int


def test_Axis_update(Parent):
    parent = Parent()
    axis = Axis(parent=parent, key=1, value=10, label="label", unit="unit")
    other_axis = Axis(
        parent=Parent(), key=2, value=10, label="label", unit="unit"
    )

    assert axis == other_axis
    axis.update(other_axis)
    assert axis._parent is parent
    assert axis._mapping == {1: 10, 2: 10}
    assert axis.label == "label"
    assert axis.unit == "unit"

    with pytest.raises(TypeError, match=f"type `{Axis}`"):
        axis.update(42)

    with pytest.raises(ValueError, match="equal axis"):
        axis.update(Axis(parent=parent, key=2, value=10, label="", unit="unit"))

    with pytest.raises(ValueError, match="equal axis"):
        axis.update(
            Axis(parent=parent, key=2, value=10, label="label", unit="")
        )


def test_Axis_len(Parent):
    axis = Axis(parent=Parent(), key=0, value=10, label="label", unit="unit")
    for i in range(1, 10):
        axis.update(
            Axis(parent=Parent(), key=i, value=10, label="label", unit="unit")
        )

    assert len(axis) == 10


def test_Axis_update_mapping(Parent):
    axis = Axis(parent=Parent(), key=0, value=10, label="label", unit="unit")
    for i in range(1, 10):
        axis.update(
            Axis(parent=Parent(), key=i, value=10, label="label", unit="unit")
        )

    axis.update_mapping([2, 3, 4])

    assert len(axis) == 3
    assert axis._mapping == {2: 10, 3: 10, 4: 10}


def test_Axis_update_parent(Parent):
    original_parent = Parent([k for k in range(10)])
    axis = Axis(
        parent=original_parent, key=1, value=10, label="label", unit="unit"
    )

    new_parent = axis._update_parent([3, 4])

    assert axis._parent is original_parent
    assert new_parent is not original_parent
    assert len(new_parent._iterations) == 2
    assert all(k == v for k, v in zip(new_parent._iterations, [3, 4]))

    with pytest.raises(TypeError, match="iterator"):
        axis._update_parent(1)


def test_Axis_asarray(Parent):
    axis = Axis(parent=Parent(), key=0, value=0, label="label", unit="unit")
    for i in range(1, 10):
        axis.update(
            Axis(parent=Parent(), key=i, value=i, label="label", unit="unit")
        )

    dtype = np.dtype({"names": ["key", "value"], "formats": [int, int]})
    arr = np.fromiter(axis._mapping.items(), dtype=dtype)

    np.testing.assert_array_equal(axis.asarray(), arr["value"])
    np.testing.assert_array_equal(
        axis.asarray(with_keys=True), (arr["key"], arr["value"])
    )


def test_Axis_iter(Parent):
    axis = Axis(parent=Parent(), key=0, value=10, label="label", unit="unit")
    for i in range(1, 10):
        axis.update(
            Axis(
                parent=Parent(), key=i, value=i + 10, label="label", unit="unit"
            )
        )

    np.testing.assert_array_equal([k for k in axis], np.arange(10) + 10)


def test_IterationAxis_init(Parent):
    with pytest.raises(TypeError, match="`int`"):
        IterationAxis(Parent(), key=1, value=10.0)

    iteration = IterationAxis(Parent(), key=1, value=10)

    assert iteration.label == "iteration"
    assert iteration.unit == ""
    assert iteration._mapping == {1: 10}
    assert iteration._dtype == int

    iteration = IterationAxis(
        Parent(), key=1, value=10, label="label", unit="unit"
    )
    assert iteration.label == "label"
    assert iteration.unit == "unit"


def test_IterationAxis_getitem(Parent):
    iteration = IterationAxis(
        Parent(iterations=[k * 10 for k in range(10)]), key=1, value=10
    )
    iteration._mapping = {
        k: v for k, v in enumerate(iteration._parent._iterations)
    }

    new_parent = iteration[10]
    np.testing.assert_array_equal(new_parent._iterations, [10])

    iteration = IterationAxis(
        Parent(iterations=[k * 10 for k in range(10)]), key=1, value=10
    )
    iteration._mapping = {
        k: v for k, v in enumerate(iteration._parent._iterations)
    }

    new_parent = iteration[9]
    np.testing.assert_array_equal(new_parent._iterations, [10])

    iteration = IterationAxis(
        Parent(iterations=[k * 10 for k in range(10)]), key=1, value=10
    )
    iteration._mapping = {
        k: v for k, v in enumerate(iteration._parent._iterations)
    }
    new_parent = iteration[9:25]
    np.testing.assert_array_equal(new_parent._iterations, [10, 20])

    with pytest.raises(TypeError, match="`int`, `float` and `slice`"):
        iteration = IterationAxis(Parent(), key=1, value=10)
        iteration["42"]


def test_TimeAxis_init(Parent):
    with pytest.raises(TypeError, match="`float`"):
        TimeAxis(Parent(), key=1, value=10)

    time = TimeAxis(Parent(), key=1, value=10.0)

    assert time.label == "time"
    assert time.unit == ""
    assert time._mapping == {1: 10.0}
    assert time._dtype == float

    time = TimeAxis(Parent(), key=1, value=10.0, label="label", unit="unit")
    assert time.label == "label"
    assert time.unit == "unit"


def test_TimeAxis_getitem(Parent):
    time = TimeAxis(
        Parent(iterations=[k * 10.0 for k in range(10)]), key=1, value=10.0
    )
    time._mapping = {k: v for k, v in enumerate(time._parent._iterations)}

    new_parent = time[10]
    np.testing.assert_array_equal(new_parent._iterations, [10.0])

    time = TimeAxis(
        Parent(iterations=[k * 10.0 for k in range(10)]), key=1, value=10.0
    )
    time._mapping = {k: v for k, v in enumerate(time._parent._iterations)}

    new_parent = time[9.0]
    np.testing.assert_array_equal(new_parent._iterations, [10.0])

    time = TimeAxis(
        Parent(iterations=[k * 10.0 for k in range(10)]), key=1, value=10.0
    )
    time._mapping = {k: v for k, v in enumerate(time._parent._iterations)}
    new_parent = time[9.0:25.0]
    np.testing.assert_array_equal(new_parent._iterations, [10.0, 20.0])

    with pytest.raises(TypeError, match="`int`, `float` and `slice`"):
        time = TimeAxis(Parent(), key=1, value=10.0)
        time["42"]


def test_GridAxis_init(Parent):
    gridaxis = GridAxis(Parent(), key=0, value=(-5.0, 5.0), name="x", length=10)

    with pytest.raises(TypeError, match="array-like"):
        GridAxis(Parent(), key=0, value=1, name="x", length=10)

    with pytest.raises(ValueError, match="two entries"):
        GridAxis(Parent(), key=0, value=(1,), name="x", length=10)


def test_GridAxis_asarray():
    gridaxis = GridAxis(object, key=0, value=(-5.0, 5.0), name="x", length=10)
    np.testing.assert_array_equal(gridaxis.asarray(), [[-5.0, 5.0]])
    np.testing.assert_array_equal(gridaxis.asarray(with_keys=True)[0], [0])
    np.testing.assert_array_equal(
        gridaxis.asarray(with_keys=True)[1], [[-5.0, 5.0]]
    )


def test_GridAxis_axis_values():
    gridaxis = GridAxis(object, key=0, value=(-5.0, 5.0), name="x", length=10)
    np.testing.assert_almost_equal(
        gridaxis.axis_values, np.linspace(-5.0, 5.0, 10).reshape((1, 10))
    )

    gridaxis = GridAxis(
        object,
        key=0,
        value=(1.0, 1000.0),
        name="x",
        length=4,
        axis_type="logarithmic",
    )
    np.testing.assert_almost_equal(
        gridaxis.axis_values, np.logspace(0.0, 3.0, 4).reshape((1, 4))
    )


def test_GridAxis_min():
    gridaxis = GridAxis(object, key=0, value=(-5.0, 5.0), name="x", length=10)
    np.testing.assert_array_equal(gridaxis.min, [-5.0])
    gridaxis.min = [-10.0]
    np.testing.assert_array_equal(gridaxis.min, [-10.0])
    with pytest.raises(ValueError, match="wrong"):
        gridaxis.min = []
    with pytest.raises(ValueError, match="wrong"):
        gridaxis.min = [10.0, 123.0]


def test_GridAxis_max():
    gridaxis = GridAxis(object, key=0, value=(-5.0, 5.0), name="x", length=10)
    np.testing.assert_array_equal(gridaxis.max, [5.0])
    gridaxis.max = [10.0]
    np.testing.assert_array_equal(gridaxis.max, [10.0])

    with pytest.raises(ValueError, match="wrong"):
        gridaxis.max = []
    with pytest.raises(ValueError, match="wrong"):
        gridaxis.max = [10.0, 123.0]


def test_DataStock_init():
    data = DataStock(
        key=1, value=np.arange(10, dtype=float), shape=(10,), dtype=float
    )
    assert data.dim == 1
    for given, expected in zip(
        data._mapping.items(),
        {1: np.arange(10, dtype=float).reshape((1, 10))}.items(),
    ):
        assert given[0] == expected[0]
        np.testing.assert_array_equal(given[1], expected[1])
    assert data.shape == (10,)
    assert data.dtype == np.float

    with pytest.raises(TypeError, match="int"):
        DataStock(
            key="1", value=np.arange(10, dtype=float), shape=(10,), dtype=float
        )

    with pytest.raises(TypeError, match="BaseGrid|np.ndarray"):
        DataStock(key=1, value=1, shape=(10,), dtype=float)

    with pytest.raises(TypeError, match="type"):
        DataStock(key=1, value=np.arange(10, dtype=float), shape=(10,), dtype=1)


def test_DataStock_getitem():
    data = DataStock(
        key=1, value=np.arange(10, dtype=float), shape=(10,), dtype=np.float
    )

    np.testing.assert_equal(data[:], np.arange(10).reshape((1, 10)))
    np.testing.assert_equal(data[1], np.arange(10).reshape((1, 10)))


def test_DataStock_setitem():
    data = DataStock(
        key=1, value=np.arange(10, dtype=float), shape=(10,), dtype=np.float
    )

    data[:] = np.arange(10).reshape((1, 10)) + 2
    np.testing.assert_equal(data[:], np.arange(10).reshape((1, 10)) + 2)
    data[1] = np.arange(10).reshape((1, 10))
    np.testing.assert_equal(data[:], np.arange(10).reshape((1, 10)))


def test_DataStock_update():
    data = DataStock(key=1, value=np.arange(10), shape=(10,), dtype=np.float)
    other = DataStock(key=2, value=np.arange(10), shape=(10,), dtype=np.float)

    data.update(other)

    for given, expected in zip(
        data._mapping.items(),
        {
            1: np.arange(10).reshape((1, 10)),
            2: np.arange(10).reshape((1, 10)),
        }.items(),
    ):
        assert given[0] == expected[0]
        np.testing.assert_array_equal(given[1], expected[1])
