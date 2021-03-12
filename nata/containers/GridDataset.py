# -*- coding: utf-8 -*-
from copy import copy
from pathlib import Path
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import runtime_checkable
from warnings import warn

import dask.array as da
import ndindex as ndx
import numpy as np

from nata.axes import Axis
from nata.containers.formatting import Table
from nata.utils.formatting import array_format
from nata.utils.io import FileList
from nata.utils.types import BasicIndexing
from nata.utils.types import FileLocation


@runtime_checkable
class GridBackendType(Protocol):
    name: str
    location: Path

    def __init__(self, location: FileLocation) -> None:
        ...

    @staticmethod
    def is_valid_backend(location: FileLocation) -> bool:
        ...

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
    def get_data(self, indexing: Optional[BasicIndexing] = None) -> np.ndarray:
        ...


class GridDatasetAxes:
    def __init__(
        self,
        axes: Sequence[Axis],
        *,
        time: Optional[Axis] = None,
        iteration: Optional[Axis] = None,
    ) -> None:
        if iteration is not None and not isinstance(iteration, Axis):
            raise TypeError("Argument 'iteration' axes has to by of type `Axis`")

        if time is not None and not isinstance(time, Axis):
            raise TypeError("Argument 'time' axes has to by of type `Axis`")

        if not all(isinstance(ax, Axis) for ax in axes):
            raise TypeError("Argument 'axes' has to by of type `List[Axis]`")

        if any(ax.ndim == 0 for ax in axes):
            raise ValueError("0-dimensional axis not allowed!")

        # check for mismatch in axis (temporal information -> stored in 1st index)
        if axes and len(set(len(ax) for ax in axes)) != 1:
            raise ValueError("Length of individual axis mismatch!")

        if any(ax.axis_dim not in (0, 1) for ax in axes):
            raise NotImplementedError("Only 0d and 1d axis are supported!")

        self._axes = axes if isinstance(axes, list) else list(axes)
        self._time = self._find_axis_in_axes("time", axes) if time is None else time
        self._iteration = (
            self._find_axis_in_axes("iteration", axes)
            if iteration is None
            else iteration
        )

        self._axes_map = {}
        for ax in self._axes:
            if ax.name in ("time", "iteration"):
                continue
            elif ax.name not in self._axes_map:
                self._axes_map[ax.name] = ax
            else:
                warn(f"Inconsistency in axes object. Axis '{ax.name}' present!")

    def __eq__(self, other: Union[Any, "GridDatasetAxes"]) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.time is None and other.time is not None:
            return False

        if self.time is not None and not self.time.is_equiv_to(other.time):
            return False

        if self.iteration is None and other.iteration is not None:
            return False

        if self.iteration is not None and not self.iteration.is_equiv_to(
            other.iteration
        ):
            return False

        if len(self.grid_axes) != len(other.grid_axes):
            return False

        if not all(s.is_equiv_to(o) for s, o in zip(self.grid_axes, other.grid_axes)):
            return False

        return True

    def __len__(self) -> int:
        return len(self._axes)

    def __contains__(self, item: str) -> bool:
        if item in ("time", "iteration"):
            return True
        else:
            return hasattr(self, item) and isinstance(getattr(self, item), Axis)

    def __iter__(self) -> Axis:
        for ax in self._axes:
            yield ax

    def __getattr__(self, name: str) -> Axis:
        if name in self._axes_map:
            return self._axes_map[name]
        else:
            raise AttributeError(f"'{self}' object has no attribute '{name}'")

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            return self._axes[key]
        elif isinstance(key, str):
            for ax in self:
                if key == ax.name:
                    return ax
            raise KeyError(f"Couldn't find matching `{key}`")
        else:
            raise KeyError("Supplied an invalid key. Only `str` and `int` supporeted!")

    def __repr__(self) -> str:
        all_axes = ", ".join(ax.name for ax in self._axes)
        return f"{self.__class__.__name__}[{all_axes}]"

    def copy(self) -> "GridDatasetAxes":
        return self.__class__(
            copy(self._axes), iteration=copy(self._iteration), time=copy(self._time)
        )

    @staticmethod
    def _find_axis_in_axes(axis_name: str, axes: List[Axis]) -> Optional[Axis]:
        for ax in axes:
            if axis_name == ax.name:
                return ax

        return None

    @property
    def time(self) -> Optional[Axis]:
        return self._time

    @property
    def iteration(self) -> Optional[Axis]:
        return self._iteration

    @property
    def grid_axes(self) -> List[Axis]:
        return self._axes

    @property
    def mapping(self) -> Dict[int, Axis]:
        return {i: a for i, a in enumerate(self)}

    @property
    def span(self) -> Tuple[int, ...]:
        return tuple(ax.shape[-1] for ax in self._axes)

    @property
    def has_temporal_axes(self) -> bool:
        return any((ax is self.time) or (ax is self.iteration) for ax in self.grid_axes)

    def index(self, axis_name: str) -> Optional[int]:
        for i, ax in enumerate(self):
            if ax.name == axis_name:
                return i

        return None

    def insert(self, i: int, new_axis: Axis) -> None:
        if not isinstance(new_axis, Axis):
            raise TypeError("Only axis are supported to be inserted!")
        self._axes.insert(i, new_axis)

    @staticmethod
    def _default_time_axis(data: Union[Axis, Sequence]) -> Axis:
        return Axis(data, axis_dim=0, name="time", label="time")

    @staticmethod
    def _default_iteration_axis(data: Union[Axis, Sequence]) -> Axis:
        return Axis(data, axis_dim=0, name="iteration", label="iteration")

    @staticmethod
    def _default_grid_axis(data: Union[Axis, Sequence], dim: int) -> Axis:
        if isinstance(data, Axis):
            return data
        else:
            return Axis(data, axis_dim=1, name=f"axis{dim}", label=f"axis {dim}")

    @classmethod
    def default_axes_from_shape(
        cls,
        shape: Tuple[int, ...],
        time: Optional[Union[Axis, Sequence]],
        iteration: Optional[Union[Axis, Sequence]],
        grid_axes: Optional[Sequence[Union[Axis, Sequence]]],
    ):
        axes = []

        time = None if time is None else cls._default_time_axis(time)
        iteration = (
            None if iteration is None else cls._default_iteration_axis(iteration)
        )

        # if multiple time steps -> 1st axis is time
        if time and len(time) > 1:
            if not shape or len(time) != shape[0]:
                raise ValueError("Mismatch between shape and time axis!")
            axes.append(time)
            _, *shape = shape

        # if multiple iteration steps and time axis is None -> 1st axis is iteration
        if iteration and (len(iteration) > 1 and time is None):
            if not shape or len(iteration) != shape[0]:
                raise ValueError("Mismatch between shape and iteration axis!")
            axes.append(iteration)
            _, *shape = shape

        # make sure number of grid axes matches with number of dimension of shape
        if grid_axes is not None:
            if len(grid_axes) != len(shape):
                raise ValueError("Number of grid axes mismatches provided shape!")

            for i, axis in enumerate(grid_axes):
                grid_axis = cls._default_grid_axis(axis, i)
                axes.append(grid_axis)

        # default behavior
        if time is None and iteration is None and grid_axes is None:
            for i, s in enumerate(shape):
                axes.append(Axis(np.arange(s), name=f"axis{i}", label=f"axis{i}"))

        return cls(axes, time=time, iteration=iteration)


class GridDataset(np.lib.mixins.NDArrayOperatorsMixin):

    _backends: AbstractSet[GridBackendType] = set()
    _handled_ufuncs = {}
    _handled_array_function = {}

    def __init__(
        self,
        data: da.core.Array,
        axes: GridDatasetAxes,
        *,
        backend: Optional[str] = None,
        locations: Optional[List[Path]] = None,
        name: str = "unnamed",
        label: str = "unnamed",
        unit: str = "",
    ):
        data = data if isinstance(data, da.core.Array) else da.asanyarray(data)

        if not isinstance(axes, GridDatasetAxes):
            raise TypeError("GridDatasetAxes object is required for 'axes'")

        if data.ndim != len(axes):
            raise ValueError("Mistmatched dimensionality between `data` and `axes`!")

        if data.shape != axes.span:
            raise ValueError("Shape and span of axes mismatch!")

        self._data = data
        self._axes = axes

        self._backend = backend
        self._locations = (
            locations
            if locations is None or isinstance(locations, list)
            else list(locations)
        )

        self._name = name
        self._label = label
        self._unit = unit

    @classmethod
    def from_array(
        cls,
        array: Union[np.ndarray, da.core.Array, Sequence],
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        dataset_axes: Optional[GridDatasetAxes] = None,
        time: Optional[Union[Axis, Sequence]] = None,
        iteration: Optional[Union[Axis, Sequence]] = None,
        grid_axes: Optional[Sequence[Union[Axis, Sequence]]] = None,
    ) -> "GridDataset":
        data = da.asanyarray(array)

        axes = (
            dataset_axes
            if dataset_axes
            else GridDatasetAxes.default_axes_from_shape(
                data.shape,
                time=time,
                iteration=iteration,
                grid_axes=grid_axes,
            )
        )

        return cls(
            data,
            axes,
            name=name,
            label=label,
            unit=unit,
            backend=None,
            locations=None,
        )

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "GridDataset":
        file_list = FileList(path, recursive=False)
        dataset = None

        for p in file_list.paths:
            backend = cls.get_valid_backend(p)
            grid, name, label, unit, axes = cls._unpack_backend(backend, p)

            tmp = cls(
                grid,
                axes,
                name=name,
                label=label,
                unit=unit,
                locations=[backend.location],
                backend=backend.name,
            )

            if not dataset:
                dataset = tmp
            else:
                dataset.append(tmp)

        return dataset

    @staticmethod
    def _unpack_backend(backend: GridBackendType, path: Path):
        grid = backend(path)
        grid_arr = da.from_array(backend(path))

        name = grid.dataset_name
        label = grid.dataset_label
        unit = grid.dataset_unit

        time = Axis(grid.time_step, name="time", label="time", unit=grid.time_unit)
        iteration = Axis(grid.iteration, name="iteration", label="iteration")

        grid_axes = []
        for min_, max_, ax_pts, ax_name, ax_label, ax_unit in zip(
            grid.axes_min,
            grid.axes_max,
            grid.shape,
            grid.axes_names,
            grid.axes_labels,
            grid.axes_units,
        ):
            ax = Axis.from_limits(
                min_, max_, ax_pts, name=ax_name, label=ax_label, unit=ax_unit
            )
            grid_axes.append(ax)

        return (
            grid_arr,
            name,
            label,
            unit,
            GridDatasetAxes(grid_axes, time=time, iteration=iteration),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.name}]"

    def _repr_html_(self) -> str:
        general_props = {
            "name": self.name,
            "label": self.label,
            "unit": self.unit or "None",
            "backend": self.backend or "None",
        }
        array_props = {
            "ndim": self.ndim,
            "shape": self.shape,
            "dtype": self.dtype,
        }
        grid_props = {
            "grid_ndim": self.grid_ndim,
            "grid_shape": self.grid_shape,
            "axes": ", ".join(axis.name for axis in self.axes),
        }

        html = Table(
            f"{type(self).__name__}:",
            general_props,
            foldable=False,
        ).render_as_html()
        html += Table("Grid Properties", grid_props).render_as_html()
        html += Table("Array Properties", array_props).render_as_html()

        return html

    def __len__(self) -> int:
        if self._axes.has_temporal_axes:
            if self._axes.time:
                return len(self._axes.time)
            else:
                return len(self._axes.iteration)

        return 1

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if dtype:
            return self.as_numpy().astype(dtype)
        else:
            return self.as_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # > Special case
        if ufunc in self._handled_ufuncs:
            return self._handled_ufuncs[ufunc](method, inputs, kwargs)

        # > General case
        # repack inputs to mimic passing as dask array
        inputs = tuple(self._data if in_ is self else in_ for in_ in inputs)

        # required additional repacking if in-place
        if "out" in kwargs:
            kwargs["out"] = tuple(
                self._data if in_ is self else in_ for in_ in kwargs["out"]
            )

        data = self._data.__array_ufunc__(ufunc, method, *inputs, **kwargs)

        # __array_ufunc__ return not implemented by dask
        if data is NotImplemented:
            raise NotImplementedError(
                f"ufunc '{ufunc}' "
                + f"for {method=}, "
                + f"{inputs=}, "
                + f"and {kwargs=} not implemented!"
            )
        # in-place scenario
        elif data is None:
            self._data = kwargs["out"][0]
            return self
        else:
            return self.__class__(
                data,
                self._axes.copy(),
                backend=copy(self.backend),
                locations=copy(self.locations),
                name=self.name,
                label=self.label,
                unit=self.unit,
            )

    def __array_function__(self, func, types, args, kwargs):
        # > Special case
        if func in self._handled_array_function:
            return self._handled_array_function[func](types, args, kwargs)

        # > General case
        # repack arguments
        types = tuple(type(self._data) if t is type(self) else t for t in types)
        args = tuple(self._data if arg is self else arg for arg in args)
        data = self._data.__array_function__(func, types, args, kwargs)

        return self.__class__(
            data,
            self._axes.copy(),
            backend=copy(self.backend),
            locations=copy(self.locations),
            name=self.name,
            label=self.label,
            unit=self.unit,
        )

    def __getitem__(self, key: Any) -> "GridDataset":
        # reconstruct key to map indexing to a shape
        index = ndx.ndindex(key).expand(self.shape).raw
        without_newaxis = [ind for ind in index if ind is not np.newaxis]
        index_of_newaxis = [i for i, ind in enumerate(index) if ind is np.newaxis]

        # index data -> no special treatment required
        data = self._data[index]

        # index each axis -> skip if index is `int` as dimension will reduce
        grid_axes = [
            ax[ind]
            for (ax, ind) in zip(self._axes, without_newaxis)
            if not isinstance(ind, int)
        ]

        # deduce time axis
        time = None

        if self._axes.time:
            t_ind = self._axes.index("time")

            if t_ind is not None and isinstance(index[t_ind], int):
                time = self._axes.time[index[t_ind]]
            else:
                time = self._axes.time

        # deduce iteration axis
        iteration = None

        if self._axes.iteration:
            iter_ind = self._axes.index("iteration")

            if iter_ind is not None and isinstance(index[iter_ind], int):
                iteration = self._axes.iteration[index[iter_ind]]
            else:
                iteration = self._axes.iteration

        # insert new axis for each np.newaxis
        if index_of_newaxis:
            if time:
                num_fills = len(time)
            elif iteration:
                num_fills = len(iteration)
            else:
                num_fills = 1

            fill_values = np.zeros((1, 1)).repeat(num_fills, axis=0)

            for i, ind in enumerate(index_of_newaxis):
                grid_axes.insert(ind, Axis(fill_values, axis_dim=1, name=f"newaxis{i}"))

        axes = GridDatasetAxes(grid_axes, time=time, iteration=iteration)

        return self.__class__(
            data,
            axes,
            backend=copy(self.backend),
            locations=copy(self.locations),
            name=self.name,
            label=self.label,
            unit=self.unit,
        )

    def __setitem__(self, key: Any, value: Any):
        raise NotImplementedError(
            "Setting individual subitems is not supported!"
            + "Please use the `.data` property!"
        )

    @property
    def backend(self) -> Optional[str]:
        """Backend associated with instance."""
        return self._backend

    @property
    def locations(self) -> Optional[List[Path]]:
        """Location associated with the source."""
        return self._locations

    @property
    def data(self) -> np.ndarray:
        """Return data for the grid as a numpy array."""
        return self.as_numpy()

    @data.setter
    def data(self, value: Union[np.ndarray, da.core.Array, Sequence]) -> None:
        value = da.asanyarray(value)
        if value.shape != self.shape:
            raise ValueError(f"Shapes inconsistent {self.shape} -> {value.shape}")
        self._data = value if len(self) != 1 else value[np.newaxis]
        self._dtype = value.dtype

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new: str) -> None:
        self._name = new

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new: str) -> None:
        self._label = new

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new: str) -> None:
        self._unit = new

    @property
    def axes(self) -> GridDatasetAxes:
        return self._axes

    @property
    def temporal_steps(self) -> int:
        if self._axes.time:
            temporal_steps = len(self._axes.time)
        elif self._axes.iteration:
            temporal_steps = len(self._axes.iteration)
        else:
            temporal_steps = 1

        return temporal_steps

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return tuple(
            s
            for s, ax in zip(self._data.shape, self._axes)
            if ax.name not in ("time", "iteration")
        )

    @property
    def grid_ndim(self) -> int:
        return len(self.grid_shape)

    def is_equiv_to(self, other: Union[Any, "GridDataset"]) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attr in ["name", "label", "unit"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if (self.grid_ndim != other.grid_ndim) or (self.grid_shape != other.grid_shape):
            return False

        return True

    def append(self, other: Union[Any, "GridDataset"]) -> None:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Can not append '{type(other).__name__}' to `GridDataset`!"
            )

        if not self.is_equiv_to(other):
            raise ValueError("GridDatasets are not equivalent")

        # update data
        if len(self.axes.time) == 1:
            self._data = self._data[np.newaxis]

        if len(other.axes.time) == 1:
            other_data = other.as_dask(squeeze=False)[np.newaxis]
        else:
            other_data = other.as_dask(squeeze=False)

        self._data = da.concatenate((self._data, other_data))

        # update axes
        for ax in self.axes:
            if ax.name in ("time", "iteration"):
                continue
            ax.append(other.axes[ax.name])

        if len(self.axes.time) == 1:
            self._axes.insert(0, self.axes.time)

        self.axes.time.append(other.axes.time)
        self.axes.iteration.append(other.axes.iteration)

    @classmethod
    def add_backend(cls, backend: GridBackendType) -> None:
        """Classmethod to add Backend to GridDatasets"""
        if cls.is_valid_backend(backend):
            cls._backends.add(backend)
        else:
            raise ValueError("Invalid backend provided")

    @classmethod
    def remove_backend(cls, backend: GridBackendType) -> None:
        """Remove a backend which is stored in ``GridDatasets``."""
        cls._backends.remove(backend)

    @classmethod
    def is_valid_backend(cls, backend: GridBackendType) -> bool:
        """Check if a backend is a valid backend for ``GridDatasets``."""
        return isinstance(backend, GridBackendType)

    @classmethod
    def get_backends(cls) -> Dict[str, GridBackendType]:
        """Dictionary of registered backends for :class:`.GridDataset`"""
        backends_dict = {}
        for backend in cls._backends:
            backends_dict[backend.name] = backend
        return backends_dict

    @classmethod
    def get_valid_backend(cls, path: Path) -> Optional[GridBackendType]:
        for backend in cls._backends:
            if backend.is_valid_backend(path):
                return backend

        return None

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)

    def as_numpy(self, squeeze: bool = False) -> np.ndarray:
        arr = self.as_dask(squeeze).compute()
        if isinstance(arr, np.ndarray):
            return arr
        else:
            return np.asanyarray(arr)

    def as_dask(self, squeeze: bool = False) -> da.core.Array:
        if squeeze:
            return da.squeeze(self._data)
        else:
            return self._data

    def info(self) -> str:
        str_ = f"{self.__class__.__name__}("
        str_ += f"name='{self.name}', "
        str_ += f"label='{self.label}', "
        str_ += f"unit='{self.unit}', "
        str_ += f"ndim={self.ndim}, "
        str_ += f"shape={self.shape}, "
        str_ += f"dtype={self.dtype}, "

        iteration_axis = self.axes.iteration
        str_ += "iteration="
        str_ += array_format(iteration_axis.data) if iteration_axis else "None"
        str_ += ", "

        time_axis = self.axes.time
        str_ += "time="
        str_ += array_format(time_axis.data) if time_axis else "None"
        str_ += ", "

        if self.axes.grid_axes:
            axes_formmating = []
            for axis in self.axes.grid_axes:
                axes_formmating.append(
                    f"Axis('{axis.name}', "
                    + f"len={len(axis)}, "
                    + f"shape={axis.shape})"
                )
            str_ += f"grid_axes=[{', '.join(axes_formmating)}]"
        else:
            str_ += f"grid_axes={self.axes.grid_axes}"

        str_ += ")"

        return str_
