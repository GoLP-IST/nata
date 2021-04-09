# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
import pytest

from nata.containers.axis import Axis


def test_Axis_init_default():
    """Create from an array-like object with pre-defined names, labels, and units"""
    axis = Axis([1, 2, 3])
    assert axis.name == "unnamed"
    assert axis.label == "unlabeled"
    assert axis.unit == ""


def test_Axis_repr():
    """Ensures correct repr formatting"""
    axis = Axis(())
    expected = f"Axis(name='{axis.name}', label='{axis.label}', unit='{axis.unit}')"
    assert repr(axis) == expected


def test_Axis_repr_html():
    """Ensures correct repr_html formatting"""
    axis = Axis(())
    expected = (
        "<span>Axis</span>"
        "<span style='color: var(--jp-info-color0);'>"
        "("
        f"name='{axis.name}', "
        f"label='{axis.label}', "
        f"unit='{axis.unit}'"
        ")"
        "</span>"
    )

    assert axis._repr_html_() == expected


def test_Axis_change_name():
    """Makes sure name property of Axis can be changed"""
    axis = Axis((), name="some_name")
    assert axis.name == "some_name"

    axis.name = "some_new_name"
    assert axis.name == "some_new_name"


def test_Axis_raises_not_identifier():
    with pytest.raises(ValueError, match="has to be a valid identifier"):
        Axis((), name="invalid name with space")

    with pytest.raises(ValueError, match="has to be a valid identifier"):
        axis = Axis(())
        axis.name = "invalid name with space"


def test_Axis_change_label():
    """Makes sure label property of Axis can be changed"""
    axis = Axis((), label="some label")
    assert axis.label == "some label"

    axis.label = "some new label"
    assert axis.label == "some new label"


def test_Axis_change_unit():
    """Makes sure unit property of Axis can be changed"""
    axis = Axis((), unit="some unit")
    assert axis.unit == "some unit"

    axis.unit = "some new unit"
    assert axis.unit == "some new unit"


def test_Axis_as_dask():
    """Check that '.as_dask' returns a dask array"""
    axis = Axis(())
    assert isinstance(axis.as_dask(), da.Array)


def test_Axis_as_numpy():
    """Check that '.as_numpy' returns a numpy array"""
    axis = Axis(())
    assert isinstance(axis.as_numpy(), np.ndarray)


def test_Axis_len():
    """Check length represents appendable dimension."""
    axis = Axis([1, 2, 3])
    assert len(axis) == 1

    axis = Axis([1, 2, 3], has_appendable_dim=True)
    assert len(axis) == 3


def test_Axis_array_props():
    axis = Axis([1, 2, 3])

    assert axis.shape == (3,)
    assert axis.ndim == 1
    assert axis.dtype == int


def test_Axis_array_method():
    axis = Axis([0, 1, 2])
    np.testing.assert_almost_equal(np.array(axis), [0, 1, 2])


def test_Axis_append():
    axis = Axis(0)

    axis.append(1)
    np.testing.assert_array_equal(axis, [0, 1])

    axis.append(2)
    np.testing.assert_array_equal(axis, [0, 1, 2])

    axis.append([3, 4])
    np.testing.assert_array_equal(axis, [0, 1, 2, 3, 4])


def test_Axis_append_other_Axis():
    some_axis = Axis(0, name="some_axis")
    some_other_axis = Axis(1, name="some_other_axis")

    some_axis.append(some_other_axis)

    assert some_axis.name == "some_axis"
    np.testing.assert_array_equal(some_axis, [0, 1])


def test_Axis_getitem():
    axis = Axis(np.arange(12).reshape((4, 3)))
    sub_axis = axis[3]
    np.testing.assert_array_equal(sub_axis, [9, 10, 11])
    assert len(sub_axis) == 1

    axis = Axis(np.arange(12).reshape((4, 3)))
    sub_axis = axis[1:3]
    np.testing.assert_array_equal(sub_axis, [[3, 4, 5], [6, 7, 8]])
    assert len(sub_axis) == 1

    axis = Axis(np.arange(12).reshape((4, 3)), has_appendable_dim=True)
    sub_axis = axis[1:3]
    np.testing.assert_array_equal(sub_axis, [[3, 4, 5], [6, 7, 8]])
    assert len(sub_axis) == 2
