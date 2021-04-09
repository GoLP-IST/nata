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
