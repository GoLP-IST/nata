# -*- coding: utf-8 -*-
"""Nata types

This file contain all the different types available with nata. It is meant to
be used for typechecking and type annotation. In addition, type checking at
runtime is supported.
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


#: Scalars and numbers
Number = Union[float, int]

#: Type which can be supplied to `numpy.array` and the resulting output is an
#: array
ArrayLike = Union[np.ndarray, Sequence[Number]]

#: Type for basic indexing
BasicIndex = Union[int, slice]

#: Type for basic indexing
BasicIndexing = Union[BasicIndex, Tuple[BasicIndex, ...]]

#: Type for file location
FileLocation = Union[Path, str]


@runtime_checkable
class BackendType(Protocol):
    """General type for retreiving data.

    The :class:`.BackendType` characterizes a general behavior of a backend.
    It is a Protocol which characterizes available attributes and their
    corresponding type annotation.
    """

    #: Name of the backend. The name can be chosen individually and is used
    #: for providing users with information about the underlaying data
    #: storage. It should follow the convention
    #: ``"CODENAME_VERSION_DATATYPE_STORAGE"``, e.g.
    #: ``"osiris_4.4.4_grid_hdf5"``
    name: str

    #: Location of the data. This attribute can be used inside a backend to
    #: point to data, either to open a file or to retrieve it.
    location: Optional[Union[str, Path]]

    @staticmethod
    def is_valid_backend(location: Union[Path, str]) -> bool:
        """Determine if a backend is a valid backend.

        Parameters
        ----------
        location : :obj:`str` or :obj:`pathlib.Path`
            Checks if a location can be passed to backend to initiate it.

        Returns
        -------
        out : :obj:`bool`
            Returns ``True`` if ``location`` is valid input parameter for
            instantiation and ``False`` otherwise.
        """
        ...


@runtime_checkable
class GridBackendType(BackendType, Protocol):
    """Backend representing a grid.

    `GridBackendType` is a protocol with the purpose of characterizing
    attributes being available for object to be recognized as a
    `GridBackendType`. Reading data is not part of this protocol but is
    characterized by `GridDataReader` which extends this protocol.
    """

    #: Name of the dataset. It has to be identifiable, e.g.
    #: ``"some_dataset_name"``.
    dataset_name: str

    #: Descriptive label of the dataset. Can be an arbitrary string, e.g.
    #: ``"some long label with space"``.
    dataset_label: str

    #: Unit of the correspinding grid. Can be an string including some latex
    #: symbols, e.g. ``"m_e c \\omega_p e^{-1}"``.
    dataset_unit: str

    #: A sequence of strings for each grid axis. Each string of the sequence
    #: has to be identifiable, e.g. ``["axis0", "axis1", "axis2"]``.
    axes_names: Sequence[str]
    #: A sequence of strings for each grid axis. Each string of the sequence
    #: is a descriptive label for each axis, e.g.
    #: ``["some axis 0", "some axis 1", "some axis 2"]``.
    axes_labels: Sequence[str]
    #: A sequence of strings for each grid axis. Each string of the sequence
    #: is the unit for each axis including some latex symbols, e.g.
    #: ``["c / \\omega_p", "mm", "\\omega_p^{-1}"]``.
    axes_units: Sequence[str]
    #: An array representing the lower limits of each grid axis.
    axes_min: np.ndarray
    #: An array representing the upper limits of each grid axis.
    axes_max: np.ndarray

    #: Associated iteration step of the underlaying data.
    iteration: int
    #: ::Associated time step of the underlaying data in code units.
    time_step: float
    #: Unit for time. Can be an arbitrary string, e.g. ``"1 / \\omega_p"``.
    time_unit: str

    #: Tuple of grid array dimensions. Corresponds to
    #: `numpy.ndarray.shape`.
    shape: Tuple[int, ...]
    #: Data type object of the grid array. Corresponds to `numpy.dtype`.
    dtype: np.dtype
    #: Dimensionality of the grid. Corresponds to `numpy.ndarray.ndim`
    ndim: int


@runtime_checkable
class GridDataReader(GridBackendType, Protocol):
    """Extended backend which handles grid data reading"""

    def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
        """Routine for reading underlaying grid data.

        Parameters
        ----------
        indexing : `int`, `slice`, \
                   :class:`typing.Tuple[Union[slice, int], ...]]`, optional
            Optional indexing for reading a section of the grid. Any `basic
            slicing and indexing  <https://numpy.org/doc/stable/reference/\
            arrays.indexing.html#basic-slicing-and-indexing>`_ can be passed
            here.

        Returns
        -------
        out : :class:`numpy.ndarray`
            Data array of the underlaying grid.
        """
        ...


@runtime_checkable
class ParticleBackendType(BackendType, Protocol):
    """Backend representing a particles.

    `ParticleBackendType` is a protocol with the purpose of characterizing
    attributes being available for object to be recognized as a
    `ParticleBackendType`. Reading data is not part of this protocol but is
    characterized by `ParticleDataReader` which extends this protocol.
    """

    #: Name of the dataset. It has to be identifiable, e.g.
    #: ``"some_dataset_name"``.
    dataset_name: str
    #: Number of particles for a given backend.
    num_particles: int

    #: A sequence of strings for each quantity stored in a backend. Each
    #: string of the sequence has to be identifiable, e.g. ``["quant0",
    #: "quant1", "quant2"]``.
    quantity_names: Sequence[str]
    #: A sequence of strings for each quantity stored in a backend. Each
    #: string of the sequence is a descriptive label for each quantity, e.g.
    #: ``["some quantity 0", "some quantity 1", "some quantity 2"]``.
    quantity_labels: Sequence[str]
    #: A sequence of strings for the unit of each quantity. Each string of
    #: the sequence is the unit for each quantity including some latex symbols,
    #: e.g. ``["m_e", "c / \\omega_p"]``.
    quantity_units: Sequence[str]

    #: Associated iteration step of the underlaying data.
    iteration: int
    #: Associated time step of the underlaying data in code units.
    time_step: float
    #: Unit for time. Can be an arbitrary string, e.g. ``"1 / \\omega_p"``.
    time_unit: str

    #: Structured type of the underlaying particle backend. Field names
    #: correspond to quantity names, and the type corresponds to the array type.
    dtype: np.dtype


@runtime_checkable
class ParticleDataReader(ParticleBackendType, Protocol):
    """Extended backend which handles particle data reading"""

    def get_data(
        self,
        indexing: Optional[BasicIndex] = None,
        fields: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        """Routine for reading underlaying grid data.

        Parameters
        ----------
        indexing : `int`, `slice`, \
                   :class:`typing.Tuple[Union[slice, int], ...]]`, optional
            Optional indexing for reading a section of the grid. Any `basic
            slicing and indexing  <https://numpy.org/doc/stable/reference/\
            arrays.indexing.html#basic-slicing-and-indexing>`_ can be passed
            here.
        fields: `str`, :class:`typing.Sequence[str]`, optional
            Optional field or Sequence of fields to read out. Each field
            correspond to a quantity stored in the backend.

        Returns
        -------
        out : :class:`numpy.ndarray`
            Data array of the underlaying particle array.
        """
        ...


@runtime_checkable
class DatasetType(Protocol):
    """Base protocol for datasets.

    Each dataset has one private attribute which stores the information about
    available backends and the possibility of storing/removing backends to
    the backend store.
    """

    #: Storage of available backends for a dataset.
    _backends: AbstractSet[BackendType]

    #: Name of associated backend.
    backend: Optional[str]

    @classmethod
    def add_backend(cls, backend: BackendType) -> None:
        """Attach a new backend to backend store.

        Parameters
        ----------
        backend : `BackendType`
            Backend which will be stored in `_backends`.
        """
        ...

    @classmethod
    def remove_backend(cls, backend: BackendType) -> None:
        """Remove an attached backend from backend store.

        Parameters
        ----------
        backend : `BackendType`
            Backend which will be removed from `._backends`.
        """
        ...

    @classmethod
    def is_valid_backend(cls, backend: BackendType) -> bool:
        """Checks if a backend is a valid backend for a dataset.

        Parameters
        ----------
        backend : `BackendType`
            Backend which will be checked if it is a valid backend for a
            dataset.

        Returns
        -------
        out : `bool`
            `True` if a backend is a valid backend for a Dataset and `False`
            otherwise.
        """
        ...

    @classmethod
    def get_backends(cls) -> Dict[str, BackendType]:
        """Obtain information over stored backends.

        Returns a `dict` with information of the stored backends inside a
        dataset. The keys are of type `str` and are the names of the
        backends. The values of the dictionary are the backends
        `BackendType`.
        """
        ...

    def append(self, other: "DatasetType") -> None:
        """Appends another dataset.

        Parameters
        ----------
        other : `DatasetType`
            The other dataset which will be appended.
        """
        ...

    def equivalent(self, other: "DatasetType") -> bool:
        """Checks for equivalence of two datasets.

        Parameters
        ----------
        other : `DatasetType`
            The other dataset which will be checked.

        Returns
        -------
        out : `bool`
            Returns `True` if an instance of a datasets is equivalent with
            another datasets, `False` otherwise.
        """
        ...


@runtime_checkable
class HasArrayInterface(Protocol):
    """Base protocol for an object to be 'characterized' as an array."""

    #: Represents the stored data for an object. It is of type `numpy.ndarray`.
    data: np.ndarray

    #: Data type object of the stored data. It is of type `numpy.dtype`.
    dtype: np.dtype
    #: Shape of the underlaying stored data. Similar to `numpy.ndarray.shape`
    shape: Tuple[int, ...]
    #: Dimensionality of the underlaying stored data. Similar to
    #: `numpy.ndarray.ndim`
    ndim: int

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Array interface of an object.

        Method which will be called when `numpy.array` or similar function
        called with the object provided as an input.
        """
        ...


@runtime_checkable
class AxisType(Protocol):
    """Base protocol for an axis."""

    #: Name of the axis. It has to be identifiable, e.g.
    #: ``"some_axis_name"``.
    name: str
    #: Descriptive label of the axis. Can be an arbitrary string, e.g.
    #: ``"some long label for an axis"``.
    label: str
    #: Unit of the correspinding axis. Can be an string including some latex
    #: symbols, e.g. ``"m_e c \\omega_p e^{-1}"``.
    unit: str
    #: Dimensionality of an axis.
    axis_dim: int

    def append(self, other: "AxisType") -> None:
        """Appends another axis.

        Parameters
        ----------
        other : `AxisType`
            The other axis which will be appended.
        """
        ...

    def equivalent(self, other: "AxisType") -> bool:
        """Checks for equivalence of two axes.

        Parameters
        ----------
        other : `AxisType`
            The other axis which will be checked.

        Returns
        -------
        out : `bool`
            Returns `True` if an instance of a  is equivalent
            with another particle quantity, `False` otherwise.
        """
        ...


@runtime_checkable
class GridAxisType(AxisType, Protocol):
    #: Axis type of a grid axis.
    axis_type: str


@runtime_checkable
class QuantityType(Protocol):
    """Base protocol for particle quantities."""

    #: Name of the quantity. It has to be identifiable, e.g.
    #: ``"some_quantity_name"``.
    name: str
    #: Descriptive label of the particle quantity. Can be an arbitrary string,
    #: e.g. ``"some long label for a particle quantity"``.
    label: str
    #: Unit of the correspinding particle quantity. Can be an string including
    #: some latex symbols, e.g. ``"m_e c \\omega_p e^{-1}"``.
    unit: str

    def append(self, other: "QuantityType") -> None:
        """Appends another a particle quantity.

        Parameters
        ----------
        other : `QuantityType`
            The other quantity which will be appended.
        """
        ...

    def equivalent(self, other: "QuantityType") -> bool:
        """Checks for equivalence of two particle quantaties.

        Parameters
        ----------
        other : `QuantityType`
            The other particle quantity which will be checked.

        Returns
        -------
        out : `bool`
            Returns `True` if an instance of a particle quantity is equivalent
            with another particle quantity, `False` otherwise.
        """
        ...


class GridDatasetAxes(TypedDict):
    """Typed dictionary containing axes for grid datasets.

    Typed dictionary `typing.TypedDict` which correspond to a `dict` at
    runtime are merely there for type checking. The attributes correspond to
    required keys inside a dictionary.
    """

    #: Axis to store iteration information.
    iteration: Optional[AxisType]
    #: Axis to store time information.
    time: Optional[AxisType]
    #: Sequence of `GridAxisType` representing an axis for each grid dimension.
    grid_axes: Sequence[GridAxisType]


@runtime_checkable
class GridDatasetType(DatasetType, Protocol):
    """Base protocol for GridDatasets.

    Extends `DatasetType` to include additional information for grids.
    """

    #: Name of the grid dataset. It has to be identifiable, e.g.
    #: ``"some_grid_name"``.
    name: str
    #: Descriptive label of the grid. Can be an arbitrary string, e.g.
    #: ``"some long label for a grid"``.
    label: str
    #: Unit of the correspinding grid. Can be an string including some latex
    #: symbols, e.g. ``"m_e c \\omega_p e^{-1}"``.
    unit: str

    #: Axes for `GridDatasetType`. It is a dictionary of type `GridDatasetAxes`.
    axes: GridDatasetAxes
    #: Shape of the grid. The grid shape corresponds to the underlaying grid
    #: and does not include temporal infortions.
    grid_shape: Tuple[int, ...]


class ParticleDatasetAxes(TypedDict):
    """Typed dictionary containing axes for particle datasets.

    Typed dictionary `typing.TypedDict` which correspond to a `dict` at
    runtime are merely there for type checking. The attributes correspond to
    required keys inside a dictionary.
    """

    #: Axis to store iteration information.
    iteration: Optional[AxisType]
    #: Axis to store time information.
    time: Optional[AxisType]


@runtime_checkable
class ParticleDatasetType(DatasetType, Protocol):
    """Base protocol for ParticleDatasets.

    Extends `DatasetType` to include additional information for particles.
    """

    #: Name of the particle dataset. It has to be identifiable, e.g.
    #: ``"some_particle_species_name"``.
    name: str

    #: Mapping storing information about stored quantaties. Keys represent are
    #: names for quantities and values are Store
    quantities: Mapping[str, QuantityType]
    #: Axes for `ParticleDatasetType`. It is a dictionary of type
    #: `ParticleDatasetAxes`.
    axes: ParticleDatasetAxes
