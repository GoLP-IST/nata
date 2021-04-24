# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Protocol
from typing import Tuple
from typing import runtime_checkable

import pytest

from nata.containers.core import BackendType
from nata.containers.core import HasBackends


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
