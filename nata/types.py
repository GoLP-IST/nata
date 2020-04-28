# -*- coding: utf-8 -*-
"""Nata types

This file contain all the different types available with nata. It is meant to
be used for typechecking and type annotation. It supports type checking at
runtime.
"""
import sys
from pathlib import Path
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

# "Protocol" and "runtime_checkable" are builtin for 3.8+
# otherwise use "typing_extension" package
if sys.version_info >= (3, 8):
    from typing import Protocol
    from typing import runtime_checkable
    from typing import TypedDict
else:
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable
    from typing_extensions import TypedDict


def is_basic_indexing(key: Any):
    indexing = np.index_exp[key]
    passes = []
    for ind in indexing:
        if isinstance(ind, (int, slice)):
            passes.append(True)
        elif ind is Ellipsis:
            passes.append(True)
        elif ind is np.newaxis:
            passes.append(True)
        else:
            passes.append(False)

    if all(passes):
        return True
    return False


@runtime_checkable
class BackendType(Protocol):
    name: str
    location: Optional[Union[str, Path]]

    def __init__(self, location: Optional[Union[str, Path]] = None) -> None:
        ...

    @staticmethod
    def is_valid_backend(path: Union[Path, str]) -> bool:
        ...


@runtime_checkable
class GridBackendType(BackendType, Protocol):
    dataset_name: str
    dataset_label: str
    dataset_unit: str

    axes_names: Sequence[str]
    axes_labels: Sequence[str]
    axes_units: Sequence[str]
    axes_min: np.ndarray
    axes_max: np.ndarray

    iteration: int
    time_step: float
    time_unit: str

    shape: Tuple[int, ...]
    dtype: np.dtype
    ndim: int


@runtime_checkable
class GridDataReader(GridBackendType, Protocol):
    def get_data(
        self, indexing=Optional[Union[int, slice, Tuple[slice, int]]]
    ) -> np.ndarray:
        ...


@runtime_checkable
class ParticleBackendType(BackendType, Protocol):
    dataset_name: str
    num_particles: int

    quantity_names: Sequence[str]
    quantity_labels: Sequence[str]
    quantity_units: Sequence[str]

    iteration: int
    time_step: float
    time_unit: str

    dtype: np.dtype


@runtime_checkable
class ParticleDataReader(ParticleBackendType, Protocol):
    def get_data(
        self,
        indexing=Optional[Union[int, slice, Tuple[slice, int]]],
        fields=Optional[Union[str, Sequence[str]]],
    ) -> np.ndarray:
        ...


@runtime_checkable
class DatasetType(Protocol):
    _backends: AbstractSet[BackendType]

    @classmethod
    def add_backend(cls, backend: BackendType) -> None:
        ...

    @classmethod
    def remove_backend(cls, backend: BackendType) -> None:
        ...

    @classmethod
    def is_valid_backend(cls, backend: BackendType) -> bool:
        ...

    @classmethod
    def get_backends(cls) -> Dict[str, BackendType]:
        ...

    def append(self, other: "DatasetType") -> None:
        ...

    def equivalent(self, other: "DatasetType") -> bool:
        ...


@runtime_checkable
class HasArrayInterface(Protocol):
    data: np.ndarray

    dtype: np.dtype
    shape: Tuple[int]
    ndim: int

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        ...


@runtime_checkable
class AxisType(HasArrayInterface, Protocol):
    name: str
    label: str
    unit: str
    axis_dim: int

    def append(self, other: "AxisType") -> None:
        ...

    def equivalent(self, other: "AxisType") -> bool:
        ...


@runtime_checkable
class GridAxisType(AxisType, Protocol):
    axis_type: str


@runtime_checkable
class QuantityType(HasArrayInterface, Protocol):
    name: str
    label: str
    unit: str

    def append(self, other: "QuantityType") -> None:
        ...

    def equivalent(self, other: "QuantityType") -> bool:
        ...


class GridDatasetAxes(TypedDict):
    iteration: Optional[AxisType]
    time: Optional[AxisType]
    grid_axes: Sequence[AxisType]


@runtime_checkable
class GridDatasetType(HasArrayInterface, DatasetType, Protocol):
    name: str
    label: str
    unit: str

    axes: GridDatasetAxes
    grid_shape: Tuple[int]


class ParticleDatasetAxes(TypedDict):
    iteration: Optional[AxisType]
    time: Optional[AxisType]


@runtime_checkable
class ParticleDatasetType(DatasetType, Protocol):
    name: str

    quantities: Mapping[str, QuantityType]
    axes: ParticleDatasetAxes


# Scalars and numbers
Number = Union[float, int]

# Arrays
ArrayLike = Union[np.ndarray, Sequence[Number]]
