# -*- coding: utf-8 -*-
"""Nata types

This file contain all the different types available with nata. It is meant to
be used for typechecking and type annotation. It supports type checking at
runtime.
"""
from pathlib import Path
from typing import Collection
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

# "Protocol" and "runtime_checkable" are builtin for 3.8+
# otherwise use "typing_extension" package
try:
    from typing import Protocol
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import Protocol  # noqa: F401
    from typing_extensions import runtime_checkable  # noqa: F401


_BasicIndexing = Union[int, slice, Tuple[slice, int]]
_FieldIndex = Union[
    List[str], str,
]


@runtime_checkable
class BackendType(Protocol):
    name: str
    location: Optional[Path]

    @staticmethod
    def is_valid_backend(path: Union[Path, str]) -> bool:
        ...

    def get_data(self, indexing=Optional[_BasicIndexing]) -> np.ndarray:
        ...


@runtime_checkable
class GridBackendType(BackendType, Protocol):
    dataset_name: str
    dataset_label: str
    dataset_unit: str

    axes_names: Collection[str]
    axes_labels: Collection[str]
    axes_units: Collection[str]
    axes_min: np.ndarray
    axes_max: np.ndarray

    iteration: int
    time_step: float
    time_unit: str

    shape: Collection[int]
    dtype: np.dtype
    ndim: int


@runtime_checkable
class ParticleBackendType(BackendType, Protocol):
    dataset_name: str
    num_particles: int

    quantity_names: Collection[str]
    quantity_labels: Collection[str]
    quantity_units: Collection[str]

    iteration: int
    time_step: float
    time_unit: str

    dtype: np.dtype

    # Particle backend has a custom get_data method for accessing fields
    def get_data(
        self, indexing=Optional[_BasicIndexing], fields=Optional[_FieldIndex]
    ) -> np.ndarray:
        ...
