# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nata.axes import Axis
from nata.utils.exceptions import DimensionError

_testCases_Axis_init_array_prop = {}
_testCases_Axis_init_array_prop["int, axis_dim=0"] = {
    "args": (0,),
    "kwargs": {"axis_dim": 0},
    "shape": (),
    "ndim": 0,
    "axis_dim": 0,
    "len(axis)": 1,
    "expected_arr": 0,
}
_testCases_Axis_init_array_prop["[int], axis_dim=0"] = {
    "args": ([0],),
    "kwargs": {"axis_dim": 0},
    "shape": (1,),
    "ndim": 1,
    "axis_dim": 0,
    "len(axis)": 1,
    "expected_arr": [0],
}
_testCases_Axis_init_array_prop["[int, int, int], axis_dim=0"] = {
    "args": ([0, 1, 3],),
    "kwargs": {"axis_dim": 0},
    "shape": (3,),
    "ndim": 1,
    "axis_dim": 0,
    "len(axis)": 3,
    "expected_arr": [0, 1, 3],
}
_testCases_Axis_init_array_prop["[int, int, int], axis_dim=1"] = {
    "args": ([0, 1, 3],),
    "kwargs": {"axis_dim": 1},
    "shape": (3,),
    "ndim": 1,
    "axis_dim": 1,
    "len(axis)": 1,
    "expected_arr": [0, 1, 3],
}
_testCases_Axis_init_array_prop["[int, int, int]"] = {
    "args": ([0, 1, 3],),
    "kwargs": {},
    "shape": (3,),
    "ndim": 1,
    "axis_dim": 1,
    "len(axis)": 1,
    "expected_arr": [0, 1, 3],
}


@pytest.mark.parametrize(
    "case",
    [v for v in _testCases_Axis_init_array_prop.values()],
    ids=[k for k in _testCases_Axis_init_array_prop.keys()],
)
def test_Axis_init_array_props(case):
    # single time step
    axis = Axis(*case["args"], **case["kwargs"])
    assert axis.shape == case["shape"]
    assert axis.ndim == case["ndim"]
    assert axis.axis_dim == case["axis_dim"]
    assert len(axis) == case["len(axis)"]
    np.testing.assert_array_equal(axis, case["expected_arr"])


def test_Axis_init_name():
    assert Axis(0).name == "unnamed"
    assert Axis(0, name="some_name").name == "some_name"


def test_Axis_name_identifiable():
    assert Axis(0, name="?/some * ^name").name == "some__name"


def test_Axis_name_setter():
    axis = Axis(0, name="something")
    assert axis.name == "something"

    axis.name = "something_else"
    assert axis.name == "something_else"

    with pytest.raises(ValueError, match="Invalid name provided!"):
        axis.name = ""


def test_Axis_init_label():
    assert Axis(0).label == "unlabeled"
    assert Axis(0, label="some label").label == "some label"


def test_Axis_label_setter():
    axis = Axis(0, label="something")
    assert axis.label == "something"

    axis.label = "something_else"
    assert axis.label == "something_else"


def test_Axis_init_unit():
    assert Axis(0).unit == ""
    assert Axis(0, unit="some unit").unit == "some unit"


def test_Axis_unit_setter():
    axis = Axis(0, unit="something")
    assert axis.unit == "something"

    axis.unit = "something_else"
    assert axis.unit == "something_else"


def test_Axis_iteration():
    """Iteration should only occur for time"""
    axis = Axis(0)
    for ax in axis:
        assert ax is axis

    values = [0, 1, 2]
    axis = Axis(values, axis_dim=0)
    for ax, v in zip(axis, values):
        np.testing.assert_array_equal(ax, v)


def test_Axis_raises_DimensionalityMismatch():
    with pytest.raises(DimensionError):
        Axis(10, axis_dim=546546)

    with pytest.raises(DimensionError):
        Axis(10, axis_dim=-123)


_testCases_data_change = {}
_testCases_data_change["(), int"] = (0, 10, 10)
_testCases_data_change["(), float"] = (0, 10.0, 10)
_testCases_data_change["(10,), float"] = (np.random.rand(10), 10.0, np.ones(10) * 10)
_testCases_data_change["(10,), (10,)"] = (
    np.random.rand(10),
    np.arange(10),
    np.arange(10),
)


@pytest.mark.parametrize(
    "input_, new, expected",
    _testCases_data_change.values(),
    ids=_testCases_data_change.keys(),
)
def test_Axis_data_change(input_, new, expected):
    axis = Axis(input_)
    np.testing.assert_array_equal(axis, input_)

    axis.data = new
    assert axis.dtype == np.array(new).dtype
    np.testing.assert_array_equal(axis, expected)


def test_Axis_data_change_not_broadcastable():
    axis = Axis(0)
    with pytest.raises(ValueError):
        axis.data = [1, 2, 3]


_testCases_dtype = {}
_testCases_dtype["float"] = float
_testCases_dtype["int"] = int
_testCases_dtype["bool"] = bool
_testCases_dtype["complex"] = complex
# some random test cases
_testCases_dtype["i4"] = "i4"
_testCases_dtype["f8"] = "f8"
_testCases_dtype["c16"] = "c16"
_testCases_dtype["a25"] = "a25"
_testCases_dtype["U25"] = "U25"


@pytest.mark.parametrize(
    "dtype", _testCases_dtype.values(), ids=_testCases_dtype.keys(),
)
def test_Axis_dtype(dtype):
    arr = np.array(0, dtype=dtype)
    axis = Axis(arr)
    assert axis.dtype == arr.dtype


def test_Axis_is_equiv_to():
    axis = Axis(1)

    assert axis.is_equiv_to(Axis(1)) is True
    assert axis.is_equiv_to(object()) is False
    assert axis.is_equiv_to(Axis(np.array(1), name="other")) is False
    assert axis.is_equiv_to(Axis(np.array(1), label="other")) is False
    assert axis.is_equiv_to(Axis(np.array(1), unit="other")) is False
    assert axis.is_equiv_to(Axis([1], axis_dim=0)) is True
    assert axis.is_equiv_to(Axis([1])) is False


def test_Axis_is_equiv_to_verbosity():
    axis = Axis(
        [1, 2, 3], axis_dim=0, name="some_name", label="some label", unit="some unit"
    )

    with pytest.warns(UserWarning, match="Types mismatch"):
        axis.is_equiv_to(object(), verbose=True)

    with pytest.warns(UserWarning, match="Dimension mismatch"):
        axis.is_equiv_to(Axis([1, 2, 3], axis_dim=1), verbose=True)

    with pytest.warns(UserWarning, match="Names mismatch"):
        axis.is_equiv_to(
            Axis(
                [1, 2, 3],
                axis_dim=0,
                name="some_other",
                label="some label",
                unit="some unit",
            ),
            verbose=True,
        )

    with pytest.warns(UserWarning, match="Labels mismatch"):
        axis.is_equiv_to(
            Axis(
                [1, 2, 3],
                axis_dim=0,
                name="some_name",
                label="some other",
                unit="some unit",
            ),
            verbose=True,
        )
    with pytest.warns(UserWarning, match="Units mismatch"):
        axis.is_equiv_to(
            Axis(
                [1, 2, 3],
                axis_dim=0,
                name="some_name",
                label="some label",
                unit="some other unit",
            ),
            verbose=True,
        )


def test_Axis_append_0d_case():
    axis = Axis(1)
    assert axis.shape == ()

    axis.append(Axis(2))
    assert axis.shape == (2,)
    np.testing.assert_array_equal(axis, [1, 2])

    axis.append(Axis(3))
    assert axis.shape == (3,)
    np.testing.assert_array_equal(axis, [1, 2, 3])


def test_Axis_append_1d_case():
    axis = Axis(np.arange(10), axis_dim=1)
    assert axis.shape == (10,)
    np.testing.assert_array_equal(axis, np.arange(10))

    axis.append(Axis(np.arange(10) + 1, axis_dim=1))
    assert axis.shape == (2, 10)
    np.testing.assert_array_equal(axis, [np.arange(10), np.arange(10) + 1])

    axis.append(Axis(np.arange(10) - 1, axis_dim=1))
    assert axis.shape == (3, 10)
    np.testing.assert_array_equal(
        axis, [np.arange(10), np.arange(10) + 1, np.arange(10) - 1]
    )


def test_Axis_append_wrong_type():
    with pytest.raises(TypeError, match="Can not append"):
        Axis(np.array(1)).append(1)


def test_Axis_append_different_axis():
    with pytest.raises(ValueError, match="Mismatch in attributes"):
        Axis(np.array(1), name="some_name").append(
            Axis(np.array(1), name="different_name")
        )


def test_Axis_len():
    axis = Axis(1)
    assert len(axis) == 1

    axis.append(Axis(2))
    assert len(axis) == 2

    axis.append(Axis([3, 4], axis_dim=0))
    assert len(axis) == 4

    axis = Axis([1])
    assert len(axis) == 1

    axis = Axis([1, 2])
    assert len(axis) == 1

    axis = Axis([[1, 2]])
    assert len(axis) == 1


_testCases_getitem = {}
_testCases_getitem["0d, multi time steps - [int]"] = {
    "args": ([0, 1, 2],),
    "kwargs": {"axis_dim": 0},
    "indexing": np.s_[2],
    "expected_shape": (),
    "expected_axis_dim": 0,
    "expected_len": 1,
    "expected_values": 2,
}
_testCases_getitem["0d, multi time steps - [range]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 0},
    "indexing": np.s_[3:7],
    "expected_shape": (4,),
    "expected_axis_dim": 0,
    "expected_len": 4,
    "expected_values": [3, 4, 5, 6],
}
_testCases_getitem["0d, multi time steps - [:]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 0},
    "indexing": np.s_[:],
    "expected_shape": (10,),
    "expected_axis_dim": 0,
    "expected_len": 10,
    "expected_values": np.arange(10),
}
_testCases_getitem["0d, multi time steps - [newaxis]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 0},
    "indexing": np.s_[np.newaxis],
    "expected_shape": (1, 10),
    "expected_axis_dim": 0,
    "expected_len": 1,
    "expected_values": np.arange(10).reshape((1, 10)),
}
_testCases_getitem["1d, single time step - [int]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[2],
    "expected_shape": (),
    "expected_axis_dim": 0,
    "expected_len": 1,
    "expected_values": 2,
}
_testCases_getitem["1d, single time step - [range]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[4:8],
    "expected_shape": (4,),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": [4, 5, 6, 7],
}
_testCases_getitem["1d, single time step - [:]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[:],
    "expected_shape": (10,),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": np.arange(10),
}
_testCases_getitem["1d, single time step - [newaxis]"] = {
    "args": (np.arange(10),),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[np.newaxis],
    "expected_shape": (1, 10),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": np.arange(10).reshape((1, 10)),
}
_testCases_getitem["1d, multi time step - [int]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[2],
    "expected_shape": (10,),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": np.arange(10) + 1,
}
_testCases_getitem["1d, multi time step - [range]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[1:],
    "expected_shape": (3, 10),
    "expected_axis_dim": 1,
    "expected_len": 3,
    "expected_values": [np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
}

_testCases_getitem["1d, multi time step - [:]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[:],
    "expected_shape": (4, 10),
    "expected_axis_dim": 1,
    "expected_len": 4,
    "expected_values": [
        np.arange(10),
        np.arange(10) - 1,
        np.arange(10) + 1,
        np.arange(10)[::-1],
    ],
}
_testCases_getitem["1d, multi time step - [newaxis]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[np.newaxis],
    "expected_shape": (1, 4, 10),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": [
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]]
    ],
}
_testCases_getitem["1d, multi time step - [int, int]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[2, 1],
    "expected_shape": (1,),
    "expected_axis_dim": 1,
    "expected_len": 1,
    "expected_values": 2,
}
_testCases_getitem["1d, multi time step - [int, int]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[2, 1],
    "expected_shape": (),
    "expected_axis_dim": 0,
    "expected_len": 1,
    "expected_values": 2,
}
_testCases_getitem["1d, multi time step - [:, int]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[:, 1],
    "expected_shape": (4,),
    "expected_axis_dim": 0,
    "expected_len": 4,
    "expected_values": [1, 0, 2, 8],
}
_testCases_getitem["1d, multi time step - [range, range]"] = {
    "args": (
        [np.arange(10), np.arange(10) - 1, np.arange(10) + 1, np.arange(10)[::-1]],
    ),
    "kwargs": {"axis_dim": 1},
    "indexing": np.s_[1:, 3:6],
    "expected_shape": (3, 3),
    "expected_axis_dim": 1,
    "expected_len": 3,
    "expected_values": [[2, 3, 4], [4, 5, 6], [6, 5, 4]],
}


@pytest.mark.parametrize(
    "case", _testCases_getitem.values(), ids=_testCases_getitem.keys()
)
def test_Axis_getitem(case):
    axis = Axis(*case["args"], **case["kwargs"])
    subaxis = axis[case["indexing"]]

    assert subaxis is not axis
    assert subaxis.name == axis.name
    assert subaxis.label == axis.label
    assert subaxis.unit == axis.unit
    assert subaxis.shape == case["expected_shape"]
    assert subaxis.axis_dim == case["expected_axis_dim"]
    assert len(subaxis) == case["expected_len"]
    np.testing.assert_array_equal(subaxis, case["expected_values"])


@pytest.mark.parametrize("indexing", [object(), "something"], ids=["object", "str"])
def test_Axis_getitem_raise_when_not_basic_indexing(indexing):
    axis = Axis([0, 1, 2])

    with pytest.raises(IndexError):
        axis[indexing]


_testCases_from_limits = {}
_testCases_from_limits["linear"] = {
    "args": (-12.3, 42.3, 110),
    "kwargs": {"axis_type": "linear"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.linspace(-12.3, 42.3, 110),
}
_testCases_from_limits["lin"] = {
    "args": (-12.3, 42.3, 123),
    "kwargs": {"axis_type": "lin"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.linspace(-12.3, 42.3, 123),
}
_testCases_from_limits["logarithmic"] = {
    "args": (0.1, 1000.0, 42),
    "kwargs": {"axis_type": "logarithmic"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.logspace(-1, 3, 42),
}
_testCases_from_limits["log"] = {
    "args": (0.1, 100.0, 13),
    "kwargs": {"axis_type": "log"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.logspace(-1, 2, 13),
}


@pytest.mark.parametrize(
    "case", _testCases_from_limits.values(), ids=_testCases_from_limits.keys()
)
def test_Axis_from_limits(case):
    axis = Axis.from_limits(*case["args"], **case["kwargs"])

    assert axis.name == case["expected_name"]
    assert axis.label == case["expected_label"]
    assert axis.unit == case["expected_unit"]
    assert axis.axis_dim == 1
    np.testing.assert_array_equal(axis, case["expected_array"])


def test_Axis_from_limits_invlid_axis_type_raise():
    with pytest.raises(ValueError, match="Invalid axis type provided"):
        Axis.from_limits(0.0, 1.0, 10, axis_type="some_invalid_type")


def test_Axis_add():
    axis = Axis(123)
    new_axis = axis + 12
    assert new_axis is not axis
    assert type(new_axis) is Axis
    np.testing.assert_array_equal(new_axis, 123 + 12)


def test_Axis_iadd():
    axis = Axis(123)
    axis += 12
    assert type(axis) is Axis
    np.testing.assert_array_equal(axis, 123 + 12)


def test_Axis_array_function():
    arr = np.random.random_sample(100)
    axis = Axis(arr)
    new_axis = np.fft.fft(axis)
    assert new_axis is not axis
    assert type(new_axis) is Axis
    np.testing.assert_array_equal(new_axis, np.fft.fft(arr))


def test_Axis_from_another_Axis():
    arr = np.random.random_sample(10)
    axis = Axis(arr, axis_dim=1, name="some_axis", label="some_label", unit="some_unit")

    axis_of_axis = Axis(axis, axis_dim=0)
    assert axis_of_axis.axis_dim == 0
    assert axis_of_axis.name == "unnamed"
    assert axis_of_axis.label == "unlabeled"
    assert axis_of_axis.unit == ""
