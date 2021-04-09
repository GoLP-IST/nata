# -*- coding: utf-8 -*-
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
