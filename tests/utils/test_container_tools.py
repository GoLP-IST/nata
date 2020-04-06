# -*- coding: utf-8 -*-
import pytest

from nata.types import BackendType
from nata.types import DatasetType
from nata.utils.container_tools import register_backend


def test_register_backend_check_registration():
    class SimpleDataset:
        _backends = set()
        _allowed_backend_type = None

        @classmethod
        def add_backend(cls, backend: BackendType) -> None:
            cls._backends.add(backend)

        @classmethod
        def is_valid_backend(cls, backend: BackendType) -> bool:
            return True

    @register_backend(SimpleDataset)
    class DummyBackend:
        pass

    # ensures a SimpleDataset being a valid DatasetType
    assert isinstance(SimpleDataset, DatasetType)

    assert DummyBackend in SimpleDataset._backends


def test_register_backend_raise_invalid_container():
    class InvalidDataset:
        pass

    assert isinstance(InvalidDataset, DatasetType) is False

    with pytest.raises(TypeError, match="Requires container of type"):

        @register_backend(InvalidDataset)
        class DummyBackend:
            pass


def test_register_backend_raise_invalid_backend_for_container():
    class SimpleDataset:
        _backends = set()
        _allowed_backend_type = None

        @classmethod
        def add_backend(cls, backend: BackendType) -> None:
            cls._backends.add(backend)

        @classmethod
        def is_valid_backend(cls, backend: BackendType) -> bool:
            return False

    # ensures a SimpleDataset being a valid DatasetType
    assert isinstance(SimpleDataset, DatasetType)

    with pytest.raises(TypeError, match="Passed invalid backend for"):

        @register_backend(SimpleDataset)
        class DummyBackend:
            pass
