# -*- coding: utf-8 -*-
from typing import Dict

import pytest

from nata.types import BackendType
from nata.types import DatasetType
from nata.utils.container_tools import register_backend


def test_register_backend_check_registration():
    class SimpleDataset:
        _backends = set()

        @classmethod
        def add_backend(cls, backend: BackendType) -> None:
            cls._backends.add(backend)

        @classmethod
        def is_valid_backend(cls, backend: BackendType) -> bool:
            return True

        @classmethod
        def remove_backend(cls, backend: BackendType) -> None:
            ...

        @classmethod
        def get_backends(cls) -> Dict[str, BackendType]:
            ...

        def append(self, other: "SimpleDataset") -> None:
            ...

        def equivalent(self, other: "SimpleDataset") -> bool:
            ...

    # ensures a SimpleDataset being a valid DatasetType
    assert isinstance(SimpleDataset, DatasetType) is True

    # performs registration
    @register_backend(SimpleDataset)
    class DummyBackend:
        pass

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

        @classmethod
        def add_backend(cls, backend: BackendType) -> None:
            cls._backends.add(backend)

        @classmethod
        def is_valid_backend(cls, backend: BackendType) -> bool:
            return False

        @classmethod
        def remove_backend(cls, backend: BackendType) -> None:
            ...

        @classmethod
        def get_backends(cls) -> Dict[str, BackendType]:
            ...

        def append(self, other: "SimpleDataset") -> None:
            ...

        def equivalent(self, other: "SimpleDataset") -> bool:
            ...

    # ensures a SimpleDataset being a valid DatasetType
    assert isinstance(SimpleDataset, DatasetType)

    with pytest.warns(UserWarning, match="Passed invalid backend for"):

        @register_backend(SimpleDataset)
        class DummyBackend:
            pass
