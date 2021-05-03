# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
import pytest

from nata.containers.axis import Axis


def test_Axis_init_default():
    """Create from an array-like object with pre-defined names, labels, and units"""
    axis = Axis.from_array([1, 2, 3])
    assert axis.name == "unnamed"
    assert axis.label == "unlabeled"
    assert axis.unit == ""


def test_Axis_repr():
    """Ensures correct repr formatting"""
    axis = Axis.from_array(())
    expected = f"Axis<name='{axis.name}', label='{axis.label}', unit='{axis.unit}'>"
    assert repr(axis) == expected


def test_Axis_change_name():
    """Makes sure name property of Axis can be changed"""
    axis = Axis.from_array((), name="some_name")
    assert axis.name == "some_name"

    axis.name = "some_new_name"
    assert axis.name == "some_new_name"


def test_Axis_raises_not_identifier():
    with pytest.raises(ValueError, match="has to be a valid identifier"):
        Axis.from_array((), name="invalid name with space")


def test_Axis_change_label():
    """Makes sure label property of Axis can be changed"""
    axis = Axis.from_array((), label="some label")
    assert axis.label == "some label"

    axis.label = "some new label"
    assert axis.label == "some new label"


def test_Axis_change_unit():
    """Makes sure unit property of Axis can be changed"""
    axis = Axis.from_array((), unit="some unit")
    assert axis.unit == "some unit"

    axis.unit = "some new unit"
    assert axis.unit == "some new unit"


def test_Axis_to_dask():
    """Check that '.to_dask' returns a dask array"""
    axis = Axis.from_array(())
    assert isinstance(axis.to_dask(), da.Array)


def test_Axis_to_numpy():
    """Check that '.to_numpy' returns a numpy array"""
    axis = Axis.from_array(())
    assert isinstance(axis.to_numpy(), np.ndarray)


def test_Axis_len():
    """Check length represents appendable dimension."""
    axis = Axis.from_array([1, 2, 3])
    assert len(axis) == 3

    axis = Axis.from_array([[1, 2, 3]])
    assert len(axis) == 1


def test_Axis_array_props():
    axis = Axis.from_array([1, 2, 3])

    assert axis.shape == (3,)
    assert axis.ndim == 1
    assert axis.dtype == int


def test_Axis_array_method():
    axis = Axis.from_array([0, 1, 2])
    np.testing.assert_almost_equal(np.array(axis), [0, 1, 2])


def test_Axis_getitem():
    axis = Axis.from_array(np.arange(12).reshape((4, 3)))
    sub_axis = axis[3]
    np.testing.assert_array_equal(sub_axis, [9, 10, 11])

    axis = Axis.from_array(np.arange(12).reshape((4, 3)))
    sub_axis = axis[1:3]
    np.testing.assert_array_equal(sub_axis, [[3, 4, 5], [6, 7, 8]])

    axis = Axis.from_array(np.arange(12).reshape((4, 3)))
    sub_axis = axis[1:3]
    np.testing.assert_array_equal(sub_axis, [[3, 4, 5], [6, 7, 8]])


_testCases_from_limits = {}
_testCases_from_limits["linear"] = {
    "args": (-12.3, 42.3, 110),
    "kwargs": {"spacing": "linear"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.linspace(-12.3, 42.3, 110),
}
_testCases_from_limits["lin"] = {
    "args": (-12.3, 42.3, 123),
    "kwargs": {"spacing": "lin"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.linspace(-12.3, 42.3, 123),
}
_testCases_from_limits["logarithmic"] = {
    "args": (0.1, 1000.0, 42),
    "kwargs": {"spacing": "logarithmic"},
    "expected_name": "unnamed",
    "expected_label": "unlabeled",
    "expected_unit": "",
    "expected_array": np.logspace(-1, 3, 42),
}
_testCases_from_limits["log"] = {
    "args": (0.1, 100.0, 13),
    "kwargs": {"spacing": "log"},
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
    np.testing.assert_array_equal(axis, case["expected_array"])


def test_Axis_ufunc():
    axis = Axis.from_array([0, 1, 2])
    axis += 1
    np.testing.assert_array_equal(axis, [1, 2, 3])

    axis = Axis.from_array([0, 1, 2])
    np.testing.assert_array_equal(axis + 1, [1, 2, 3])


def test_Axis_array_function():
    # passing as a tuple of args
    concanated_axes = np.concatenate((Axis.from_array([0, 1]), Axis.from_array([2, 3])))
    np.testing.assert_array_equal(concanated_axes, [0, 1, 2, 3])

    # passing as a list of args
    concanated_axes = np.concatenate([Axis.from_array([0, 1]), Axis.from_array([2, 3])])
    np.testing.assert_array_equal(concanated_axes, [0, 1, 2, 3])
