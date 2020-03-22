# -*- coding: utf-8 -*-
import attr
from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
from hypothesis.strategies import text

from nata.utils.attrs import attrib_equality
from nata.utils.attrs import have_attr


@given(
    x=one_of(floats(), integers(), text()),
    y=one_of(floats(), integers(), text()),
)
def test_attrib_equality_attrs_attributes_cmp_equality(x, y):
    for i, v1 in enumerate((x, y)):
        for j, v2 in enumerate((x, y)):
            if i == j:
                assume(v1 == v2)
            else:
                assume(v1 != v2)

    @attr.s
    class Foo:
        a = attr.ib()
        b = attr.ib(default=None)

    assert attrib_equality(Foo(x), Foo(x))
    assert not attrib_equality(Foo(x), Foo(y))

    assert attrib_equality(Foo(x), Foo(x), "a")
    assert not attrib_equality(Foo(x), Foo(y), "a")
    assert attrib_equality(Foo(x), Foo(y), "b")

    assert attrib_equality(Foo(x), Foo(x), "a, b")
    assert not attrib_equality(Foo(x), Foo(y), "a, b")

    assert attrib_equality(Foo(x), Foo(x), ("a", "b"))
    assert not attrib_equality(Foo(x), Foo(y), ("a", "b"))


@given(
    x=one_of(floats(), integers(), text()),
    y=one_of(floats(), integers(), text()),
    z=one_of(floats(), integers(), text()),
)
def test_attrib_equality_attrs_attributes_attrs_equality(x, y, z):
    for i, v1 in enumerate((x, y, z)):
        for j, v2 in enumerate((x, y, z)):
            if i == j:
                assume(v1 == v2)
            else:
                assume(v1 != v2)

    @attr.s
    class Foo:
        a = attr.ib()
        b = attr.ib(eq=False)

    assert attrib_equality(Foo(x, y), Foo(x, z))
    assert not attrib_equality(Foo(x, y), Foo(y, z))
    assert not attrib_equality(Foo(x, y), Foo(y, y))


@given(
    x=one_of(floats(), integers(), text()),
    y=one_of(floats(), integers(), text()),
)
def test_attrib_equality_attrs_test(x, y):
    for i, v1 in enumerate((x, y)):
        for j, v2 in enumerate((x, y)):
            if i == j:
                assume(v1 == v2)
            else:
                assume(v1 != v2)

    @attr.s
    class Bar:
        a = attr.ib(eq=True)

    @attr.s
    class Baz:
        a = attr.ib(eq=False)

    @attr.s
    class Foo:
        x = attr.ib()

    assert attrib_equality(Foo(Baz(x)), Foo(Baz(y)))
    assert not attrib_equality(Foo(Bar(x)), Foo(Bar(y)))


def test_have_attr_no_args():
    assert have_attr() is False


def test_have_attr_single_arg():
    @attr.s
    class Has:
        pass

    class Does_not_Have:
        pass

    assert have_attr(Has()) is True
    assert have_attr(Does_not_Have()) is False


def test_have_attr_multiple_args():
    @attr.s
    class Has:
        pass

    class Does_not_Have:
        pass

    assert have_attr(Has(), Has()) is True
    assert have_attr(Has(), Does_not_Have()) is False
    assert have_attr(Does_not_Have(), Does_not_Have()) is False
