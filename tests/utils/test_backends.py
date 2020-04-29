# -*- coding: utf-8 -*-
import pytest

from nata.utils.backends import sort_particle_quantities


@pytest.mark.parametrize(
    "given, order, expected",
    [
        ("b,c,d,a", None, "a,b,c,d"),
        ("x1,x2,p1,p2", None, "p1,p2,x1,x2"),
        ("x2,p2,x1,p1", ["x", "p"], "x1,x2,p1,p2"),
        ("Ene,p1,q,x1,p2,x2,p3", ["x", "p"], "x1,x2,p1,p2,p3,Ene,q"),
    ],
)
def test_sort_particle_quantities(given, order, expected):
    given = given.split(",")
    expected = expected.split(",")
    actual = sort_particle_quantities(given, order=order)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])
