# -*- coding: utf-8 -*-
from typing import Union

import dask.array as da
import numpy as np
import pytest
from numpy.lib import recfunctions

from nata.containers.core import HasNumpyInterface
from nata.containers.core import HasPluginSystem
from nata.containers.utils import register_plugin
from nata.containers.utils import to_dask
from nata.containers.utils import to_numpy
from nata.containers.utils import unstructured_to_structured


def test_unstructured_to_structured():
    arr = da.array([1, 2, 3])
    structured_dtype = np.dtype([("f1", int), ("f2", float), ("f3", np.bool8)])

    # expected is based on doing numpy 'unstructured_to_structured'
    expected = recfunctions.unstructured_to_structured(arr.compute(), structured_dtype)

    # do own 'unstructured_to_structured'
    converted_arr = unstructured_to_structured(arr, structured_dtype)

    assert isinstance(converted_arr, da.Array)
    np.testing.assert_array_equal(converted_arr, expected)


def test_to_dask():
    class ClassWithNumpyInterface(HasNumpyInterface):
        pass

    class ClassWithoutNumpyInterface:
        pass

    assert isinstance(to_dask(ClassWithNumpyInterface(da.arange(10))), da.Array)

    with pytest.raises(
        TypeError,
        match=r"requires object of type '.*HasNumpyInterface.*'",
    ):
        to_dask(ClassWithoutNumpyInterface())


def test_to_numpy():
    class ClassWithNumpyInterface(HasNumpyInterface):
        pass

    class ClassWithoutNumpyInterface:
        pass

    assert isinstance(to_numpy(ClassWithNumpyInterface(da.arange(10))), np.ndarray)

    with pytest.raises(
        TypeError,
        match=r"requires object of type '.*HasNumpyInterface.*'",
    ):
        to_numpy(ClassWithoutNumpyInterface())


def test_register_plugin_from_annotation():
    class Foo(HasPluginSystem):
        pass

    @register_plugin
    def plugin_function(obj: Foo):
        pass

    assert "plugin_function" in Foo.get_method_plugins()


def test_register_plugin_from_annotation_with_parenthesis():
    class Foo(HasPluginSystem):
        pass

    @register_plugin()
    def plugin_function(obj: Foo):
        pass

    assert "plugin_function" in Foo.get_method_plugins()


def test_register_plugin_from_arg():
    class Foo(HasPluginSystem):
        pass

    @register_plugin(Foo)
    def plugin_function(obj):
        pass

    assert "plugin_function" in Foo.get_method_plugins()


def test_register_plugin_register_multiple():
    class Foo(HasPluginSystem):
        pass

    class Bar(HasPluginSystem):
        pass

    @register_plugin([Foo, Bar])
    def plugin_function(obj):
        pass

    assert "plugin_function" in Foo.get_method_plugins()
    assert "plugin_function" in Bar.get_method_plugins()


def test_register_plugin_register_multiple_through_annotations():
    class Foo(HasPluginSystem):
        pass

    class Bar(HasPluginSystem):
        pass

    @register_plugin
    def plugin_function(obj: Union[Foo, Bar]):
        pass

    assert "plugin_function" in Foo.get_method_plugins()
    assert "plugin_function" in Bar.get_method_plugins()


def test_register_plugin_name_parameter():
    class Foo(HasPluginSystem):
        pass

    class Bar(HasPluginSystem):
        pass

    @register_plugin(name="a")
    def _(obj: Foo):
        pass

    @register_plugin(Foo, name="b")
    def _(obj):
        pass

    @register_plugin([Foo, Bar], name="c")
    def _(obj):
        pass

    assert "a" in Foo.get_method_plugins()
    assert "b" in Foo.get_method_plugins()
    assert "c" in Foo.get_method_plugins()
    assert "c" in Bar.get_method_plugins()


def test_register_plugin_plugin_type():
    class Foo(HasPluginSystem):
        pass

    class Bar(HasPluginSystem):
        pass

    @register_plugin(Foo, plugin_type="method")
    def a(obj):
        pass

    @register_plugin(Foo, plugin_type="property")
    def b(obj):
        pass

    @register_plugin([Foo, Bar], plugin_type="method")
    def c(obj):
        pass

    @register_plugin([Foo, Bar], plugin_type="property")
    def d(obj):
        pass

    assert "a" in Foo.get_method_plugins()

    assert "b" in Foo.get_property_plugins()

    assert "c" in Foo.get_method_plugins()
    assert "c" in Bar.get_method_plugins()

    assert "d" in Foo.get_property_plugins()
    assert "d" in Bar.get_property_plugins()
