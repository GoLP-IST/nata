# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set

import numpy as np
import pytest

from nata.types import BackendType
from nata.types import DatasetType
from nata.types import GridBackendType
from nata.types import HasArrayInterface
from nata.types import ParticleBackendType
from nata.types import is_basic_indexing


@pytest.mark.parametrize(
    "key, true_or_false",
    [
        # basic slicing
        (1, True),
        (np.s_[:], True),
        (np.s_[0, 1], True),
        (np.s_[0, :], True),
        (np.s_[:, :], True),
        (np.s_[:, :, :, :], True),
        (np.s_[...], True),
        (np.s_[np.newaxis], True),
        # non-basic slicing
        (1.0, False),
        ("a", False),
        (np.s_["s", :], False),
        (np.s_["s", 0], False),
    ],
    ids=[
        "int",
        "slice",
        "int, int",
        "int, slice",
        "slice, slice",
        "slice, slice, slice, slice",
        "Ellipsis",
        "np.newaxis",
        "float",
        "string",
        "string, slice",
        "string, int",
    ],
)
def test_is_basic_indexing(key, true_or_false):
    assert is_basic_indexing(key) is true_or_false


@pytest.fixture(name="InvalidBackend")
def _InvalidBackend():
    class InvalidBackend:
        pass

    return InvalidBackend


@pytest.fixture(name="ValidBackend")
def _ValidBackend():
    class ValidBackend:
        name = ""
        location = Path(".")

        @staticmethod
        def is_valid_backend(path: Path) -> bool:
            raise NotImplementedError("Should never be called")

        def get_data(self, indexing=None) -> np.ndarray:
            raise NotImplementedError("Should never be called")

    return ValidBackend


def test_BackendType_runtime_check_class(InvalidBackend, ValidBackend):
    assert isinstance(InvalidBackend, BackendType) is False
    assert isinstance(ValidBackend, BackendType) is True


def test_BackendType_runtime_check_instance(InvalidBackend, ValidBackend):
    assert isinstance(InvalidBackend(), BackendType) is False
    assert isinstance(ValidBackend(), BackendType) is True


@pytest.fixture(name="InvalidGridBackend")
def _InvalidGridBackend():
    class InvalidGridBackend:
        pass

    return InvalidGridBackend


@pytest.fixture(name="ValidGridBackend")
def _ValidGridBackend():
    class ValidGridBackend:
        name = ""
        location = Path(".")
        dataset_name = ""
        dataset_label = ""
        dataset_unit = ""

        axes_names = [""]
        axes_labels = [""]
        axes_units = [""]
        axes_min = np.array([-1.0])
        axes_max = np.array([1.0])

        iteration = 1
        time_step = 1.0
        time_unit = ""

        shape = (10,)
        dtype = np.dtype(float)
        ndim = 1

        @staticmethod
        def is_valid_backend(path: Path) -> bool:
            raise NotImplementedError("Should never be called")

        def get_data(self, indexing=None) -> np.ndarray:
            raise NotImplementedError("Should never be called")

    return ValidGridBackend


def test_GridBackendType_runtime_check_class(
    InvalidGridBackend, ValidGridBackend
):
    assert isinstance(InvalidGridBackend, GridBackendType) is False
    assert isinstance(ValidGridBackend, GridBackendType) is True


def test_GridBackendType_runtime_check_instance(
    InvalidGridBackend, ValidGridBackend
):
    assert isinstance(InvalidGridBackend(), GridBackendType) is False
    assert isinstance(ValidGridBackend(), GridBackendType) is True


@pytest.fixture(name="InvalidParticleBackend")
def _InvalidParticleBackend():
    class InvalidParticleBackend:
        pass

    return InvalidParticleBackend


@pytest.fixture(name="ValidParticleBackend")
def _ValidParticleBackend():
    class ValidParticleBackend:
        name = ""
        location = Path(".")

        dataset_name = ""
        num_particles = 1

        quantity_names = ["q1"]
        quantity_labels = ["q_1"]
        quantity_units = [""]

        iteration = 1
        time_step = 1.0
        time_unit = ""

        dtype = np.dtype({"names": ["q1"], "formats": [int]})

        @staticmethod
        def is_valid_backend(path: Path) -> bool:
            raise NotImplementedError("Should never be called")

        def get_data(self, indexing=None) -> np.ndarray:
            raise NotImplementedError("Should never be called")

    return ValidParticleBackend


def test_ParticleBackendType_runtime_check_class(
    InvalidParticleBackend, ValidParticleBackend
):
    assert isinstance(InvalidParticleBackend, ParticleBackendType) is False
    assert isinstance(ValidParticleBackend, ParticleBackendType) is True


def test_ParticleBackendType_runtime_check_instance(
    InvalidParticleBackend, ValidParticleBackend
):
    assert isinstance(InvalidParticleBackend(), ParticleBackendType) is False
    assert isinstance(ValidParticleBackend(), ParticleBackendType) is True


@pytest.fixture(name="InvalidDataset")
def _InvalidDatasetType():
    class InvalidDataset:
        pass

    return InvalidDataset


@pytest.fixture(name="ValidDataset")
def _ValidDatasetType():
    class ValidDatasetType:
        _backends: Set[BackendType] = set()

        @classmethod
        def add_backend(cls, backend: BackendType) -> None:
            raise NotImplementedError("Should never be called")

        @classmethod
        def remove_backend(cls, backend: BackendType) -> None:
            raise NotImplementedError("Should never be called")

        @classmethod
        def get_backends(cls) -> Dict[str, BackendType]:
            raise NotImplementedError("Should never be called")

        @classmethod
        def is_valid_backend(cls, backend: BackendType) -> bool:
            raise NotImplementedError("Should never be called")

        def append(self, other: Any) -> None:
            raise NotImplementedError("Should never be called")

        def equivalent(self, other: Any) -> bool:
            raise NotImplementedError("Should never be called")

        @property
        def backend(self) -> Optional[str]:
            return None

    return ValidDatasetType


def test_DatasetType_check_add_backend(InvalidDataset, ValidDataset):
    assert isinstance(InvalidDataset, DatasetType) is False
    assert isinstance(ValidDataset, DatasetType) is True


@pytest.fixture(name="InvalidHasArrayInterface")
def _InvalidHasArrayInterface():
    class InvalidHasArrayInterface:
        pass

    return InvalidHasArrayInterface


def test_HasArrayInterface_runtime_check_class(InvalidHasArrayInterface):
    assert isinstance(np.ndarray, HasArrayInterface) is True
    assert isinstance(InvalidHasArrayInterface, HasArrayInterface) is False
