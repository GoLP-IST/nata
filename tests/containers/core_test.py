# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Protocol
from typing import Tuple
from typing import runtime_checkable

import dask.array as da
import numpy as np
import pytest

from nata.containers.core import BackendType
from nata.containers.core import HasAnnotations
from nata.containers.core import HasBackends
from nata.containers.core import HasNumpyInterface
from nata.containers.core import HasPluginSystem


@pytest.fixture(name="ExtendedProtocol")
def _ExtendedProtocol() -> BackendType:
    @runtime_checkable
    class ExtendedProtocol(Protocol):
        name: str
        some_other_prop: Tuple[int, ...]

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            ...

        def some_other_method(self, foo: int) -> float:
            ...

    return ExtendedProtocol


@pytest.fixture(name="UseBackend")
def _ClassWithBackend(ExtendedProtocol: BackendType) -> HasBackends:
    class UseBackend(HasBackends, backend_protocol=ExtendedProtocol):
        pass

    return UseBackend


def test_custom_class_no_backends(UseBackend: HasBackends):
    """Ensures custom defined test class does not have registered backends."""
    assert len(UseBackend.get_backends()) == 0


def test_is_valid_backend_with_valid_protocol(
    UseBackend: HasBackends, ExtendedProtocol: BackendType
):
    """Make sure custom classes pass valid backend test"""

    class ClassSatisfiesProtocol:
        name = "name of class"
        some_other_prop = (1, 2, 3)

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            raise NotImplementedError("should never be reached")

        def some_other_method(self, foo: int) -> float:
            raise NotImplementedError("should never be reached")

    # requires init because
    # `TypeError: Protocols with non-method members don't support issubclass()`
    assert isinstance(ClassSatisfiesProtocol(), ExtendedProtocol)
    assert UseBackend.is_valid_backend(ClassSatisfiesProtocol) is True


def test_is_valid_backend_with_invalid_protocol(
    UseBackend: HasBackends, ExtendedProtocol: BackendType
):
    """Make sure custom classes pass valid backend test"""

    class ClassDoesNotSatisfyProtocol:
        name = "name of class"

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            raise NotImplementedError("should never be reached")

    # requires init because
    # `TypeError: Protocols with non-method members don't support issubclass()`
    assert not isinstance(ClassDoesNotSatisfyProtocol(), ExtendedProtocol)
    assert UseBackend.is_valid_backend(ClassDoesNotSatisfyProtocol) is False


def test_raise_when_no_protocol_specified():
    """Raise when no protocol is specified and backend is checked"""

    with pytest.raises(AttributeError, match=r"requires backend protocol"):

        class UseBackend(HasBackends):
            pass


def test_adding_removing_new_backend(UseBackend: HasBackends):
    """Check adding and removing of custom backends"""

    class NewBackend:
        # random asigned values for attrs
        name = "name of backend"
        some_other_prop = (10, 10)

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            raise NotImplementedError("should never be reached")

        def some_other_method(self, foo: int) -> float:
            raise NotImplementedError("should never be reached")

    assert len(UseBackend.get_backends()) == 0

    # add backend
    UseBackend.add_backend(NewBackend)
    assert UseBackend.get_backends()[NewBackend.name] is NewBackend

    # remove by value
    UseBackend.remove_backend(NewBackend)
    assert len(UseBackend.get_backends()) == 0

    # add backend
    UseBackend.add_backend(NewBackend)
    assert UseBackend.get_backends()[NewBackend.name] is NewBackend

    # remove by name
    UseBackend.remove_backend(NewBackend.name)
    assert len(UseBackend.get_backends()) == 0


def test_removing_raises(UseBackend: HasBackends):
    with pytest.raises(ValueError, match=r"backend '.*' not registered"):
        # remove by value
        UseBackend.remove_backend(object)

    with pytest.raises(ValueError, match=r"backend '.*' not registered"):
        # remove by name
        UseBackend.remove_backend("invalide backend name")


def test_adding_invalid_backend_raises(UseBackend: HasBackends):
    with pytest.raises(TypeError, match=r"invalid backend provided"):
        UseBackend.add_backend(object)


def test_get_valid_backend(UseBackend: HasBackends):
    """Check if correct backend is returned"""

    class DummyBackend1:
        # random asigned values for attrs
        name = "name of backend"
        some_other_prop = (10, 10)

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            return location == Path("/DummyBackend1")

        def some_other_method(self, foo: int) -> float:
            raise NotImplementedError("should never be reached")

    class DummyBackend2:
        # random asigned values for attrs
        name = "name of backend"
        some_other_prop = (10, 10)

        @staticmethod
        def is_valid_backend(location: Path) -> bool:
            return location == Path("/DummyBackend2")

        def some_other_method(self, foo: int) -> float:
            raise NotImplementedError("should never be reached")

    # adding backends
    UseBackend.add_backend(DummyBackend1)
    UseBackend.add_backend(DummyBackend2)

    assert UseBackend.get_valid_backend(Path("/DummyBackend1")) is DummyBackend1
    assert UseBackend.get_valid_backend(Path("/DummyBackend2")) is DummyBackend2
    assert UseBackend.get_valid_backend(Path("/does/not/have/valid/backend")) is None


def test_HasNumpyInterface_handled_array_ufunc():
    class ExtendedClass(HasNumpyInterface):
        pass

    some_ufunc = np.add

    def implementation_some_ufunc():
        pass

    assert len(ExtendedClass.get_handled_array_ufunc()) == 0
    ExtendedClass.add_handled_array_ufunc(some_ufunc, implementation_some_ufunc)
    assert some_ufunc in ExtendedClass.get_handled_array_ufunc()
    assert (
        ExtendedClass.get_handled_array_ufunc()[some_ufunc] is implementation_some_ufunc
    )
    ExtendedClass.remove_handled_array_ufunc(some_ufunc)
    assert len(ExtendedClass.get_handled_array_ufunc()) == 0


def test_HasNumpyInterface_raise_when_not_ufunc():
    class ExtendedClass(HasNumpyInterface):
        pass

    with pytest.raises(TypeError, match="function is not of type ufunc"):

        def not_ufunc():
            pass

        ExtendedClass.add_handled_array_ufunc(not_ufunc, lambda _: None)


def test_HasNumpyInterface_raise_when_remove_invalid_for_ufunc():
    class ExtendedClass(HasNumpyInterface):
        pass

    with pytest.raises(ValueError, match=r"ufunc '.*' is not registered"):
        ExtendedClass.remove_handled_array_ufunc(lambda _: None)


def test_HasNumpyInterface_handled_array_function():
    class ExtendedClass(HasNumpyInterface):
        pass

    some_array_function = np.fft.fft

    def implementation_some_function():
        pass

    assert len(ExtendedClass.get_handled_array_function()) == 0
    ExtendedClass.add_handled_array_function(
        some_array_function, implementation_some_function
    )
    assert some_array_function in ExtendedClass.get_handled_array_function()
    assert (
        ExtendedClass.get_handled_array_function()[some_array_function]
        is implementation_some_function
    )
    ExtendedClass.remove_handeld_array_function(some_array_function)
    assert len(ExtendedClass.get_handled_array_function()) == 0


def test_HasNumpyInterface_raise_when_remove_invalid_for_array_function():
    class ExtendedClass(HasNumpyInterface):
        pass

    with pytest.raises(ValueError, match=r"function '.*' is not registered"):
        ExtendedClass.remove_handeld_array_function(lambda _: None)


def test_HasNumpyInterface_implements_ufunc():
    class ExtendedClass(HasNumpyInterface):
        pass

    @ExtendedClass.implements(np.add)
    def custom_add() -> None:
        pass

    assert ExtendedClass.get_handled_array_ufunc()[np.add] is custom_add


def test_HasNumpyInterface_implements_array_func():
    class ExtendedClass(HasNumpyInterface):
        pass

    @ExtendedClass.implements(np.fft.fft)
    def custom_fft() -> None:
        pass

    assert ExtendedClass.get_handled_array_function()[np.fft.fft] is custom_fft


def test_HasNumpyInterface_to_dask():
    class ExtendedClass(HasNumpyInterface):
        pass

    obj = ExtendedClass(da.arange(10))
    assert isinstance(obj.to_dask(), da.Array)
    np.testing.assert_array_equal(obj.to_dask(), np.arange(10))


def test_HasNumpyInterface_to_numpy():
    class ExtendedClass(HasNumpyInterface):
        pass

    obj = ExtendedClass(da.arange(10))
    assert isinstance(obj.to_numpy(), np.ndarray)
    np.testing.assert_array_equal(obj.to_numpy(), np.arange(10))


def test_HasNumpyInterface_array_props():
    class ExtendedClass(HasNumpyInterface):
        pass

    obj = ExtendedClass(da.arange(10, dtype=int))
    assert obj.dtype == int
    assert obj.ndim == 1
    assert obj.shape == (10,)
    assert len(obj) == 10


@pytest.mark.skip
def test_HasNumpyInterface_array_ufunc_dispatch():
    # TODO: add tests
    pass


@pytest.mark.skip
def test_HasNumpyInterface_array_function_dispatch():
    # TODO: add tests
    pass


def test_HasPluginSystem_property_plugin():
    class ExtendedClass(HasPluginSystem):
        pass

    def plug_function(obj):
        return obj

    assert len(ExtendedClass.get_property_plugins()) == 0
    ExtendedClass.add_property_plugin("my_plugin", plug_function)

    obj = ExtendedClass()
    assert obj.my_plugin is obj
    assert ExtendedClass.get_property_plugins()["my_plugin"] is plug_function
    ExtendedClass.remove_property_plugin("my_plugin")
    assert len(ExtendedClass.get_property_plugins()) == 0


def test_HasPluginSystem_property_pluging_raise_invalid_plugin_name():
    class ExtendedClass(HasPluginSystem):
        pass

    with pytest.raises(TypeError, match="'plugin_name' has to be a 'str'"):
        ExtendedClass.add_property_plugin(1, lambda _: None)

    with pytest.raises(
        ValueError, match="'not identifier' has to be a valid identifier"
    ):
        ExtendedClass.add_property_plugin("not identifier", lambda _: None)


def test_HasPluginSystem_property_pluging_raise_not_registered():
    class ExtendedClass(HasPluginSystem):
        pass

    with pytest.raises(ValueError, match="plugin 'not_registered' is not registered"):
        ExtendedClass.remove_property_plugin("not_registered")


def test_HasPluginSystem_method_plugin():
    class ExtendedClass(HasPluginSystem):
        pass

    def plug_function(obj):
        return obj

    assert len(ExtendedClass.get_method_plugins()) == 0
    ExtendedClass.add_method_plugin("my_plugin", plug_function)

    obj = ExtendedClass()
    assert obj.my_plugin() is obj
    assert ExtendedClass.get_method_plugins()["my_plugin"] is plug_function
    ExtendedClass.remove_method_plugin("my_plugin")
    assert len(ExtendedClass.get_method_plugins()) == 0


def test_HasPluginSystem_method_pluging_raise_invalid_plugin_name():
    class ExtendedClass(HasPluginSystem):
        pass

    with pytest.raises(TypeError, match="'plugin_name' has to be a 'str'"):
        ExtendedClass.add_method_plugin(1, lambda _: None)

    with pytest.raises(
        ValueError, match="'not identifier' has to be a valid identifier"
    ):
        ExtendedClass.add_method_plugin("not identifier", lambda _: None)


def test_HasPluginSystem_method_pluging_raise_not_registered():
    class ExtendedClass(HasPluginSystem):
        pass

    with pytest.raises(ValueError, match="plugin 'not_registered' is not registered"):
        ExtendedClass.remove_method_plugin("not_registered")


def test_HasAnnotations():
    class ExtendedClass(HasAnnotations):
        def __init__(self) -> None:
            self._name = "some_name"
            self._label = "some label"
            self._unit = "some unit"

    obj = ExtendedClass()

    assert obj.name == "some_name"
    assert obj.label == "some label"
    assert obj.unit == "some unit"

    obj.name = "some_new_name"
    assert obj.name == "some_new_name"

    obj.label = "some new label"
    assert obj.label == "some new label"

    obj.unit = "some new unit"
    assert obj.unit == "some new unit"


def test_HasAnnotations_raise_invalid_name():
    class ExtendedClass(HasAnnotations):
        pass

    obj = ExtendedClass()

    with pytest.raises(ValueError, match="'name' has to be an identifier"):
        obj.name = "not an identifier"
