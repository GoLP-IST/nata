# -*- coding: utf-8 -*-
from copy import copy
from functools import partial
from pathlib import Path
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from typing import runtime_checkable

import dask.array as da
import ndindex as ndx
import numpy as np
from numpy.typing import ArrayLike

from nata.containers.formatting import Table
from nata.utils.io import FileList
from nata.utils.types import BasicIndexing
from nata.utils.types import FileLocation

from .axis import Axis
from .utils import get_doc_heading

__all__ = ["GridAxes", "GridDataset"]


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


class GridAxes:
    def __init__(
        self,
        indexable: Sequence[Axis],
        hidden: Sequence[Axis] = (),
    ) -> None:
        self._indexable = tuple(indexable)
        self._hidden = tuple(hidden)

        self._all_axes = {ax.name: ax for ax in (self._indexable + self._hidden)}

    @property
    def indexable(self) -> Tuple[Axis]:
        return self._indexable

    @property
    def hidden(self) -> Tuple[Axis]:
        return self._hidden

    def __repr__(self) -> str:
        return (
            "GridAxes<"
            f"indexable={tuple(ax.name for ax in self.indexable)}, "
            f"hidden={tuple(ax.name for ax in self.hidden)}"
            ">"
        )

    def _repr_html_(self) -> str:
        content = {
            "indexable": ", ".join(ax.name for ax in self.indexable),
            "hidden": ", ".join(ax.name for ax in self.hidden) or "-",
        }
        return Table(f"{type(self).__name__}", content, foldable=False).render_as_html()

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            return self._indexable[key]
        elif isinstance(key, str):
            return self._all_axes[key]
        else:
            raise KeyError("Supplied an invalid key. Only `str` and `int` supporeted!")


class GridDataset(np.lib.mixins.NDArrayOperatorsMixin):

    _backends: AbstractSet[GridBackendType] = set()
    _handled_ufuncs = {}
    _handled_array_function = {}

    def __init__(
        self,
        data: da.core.Array,
        axes: GridAxes,
        *,
        backend: Optional[str] = None,
        locations: Optional[List[Path]] = None,
        name: str = "unnamed",
        label: str = "unnamed",
        unit: str = "",
        has_appendable_dim: bool = False,
    ):
        data = data if isinstance(data, da.core.Array) else da.asanyarray(data)

        self._data = data
        self._axes = axes
        self._has_appendable_dim = has_appendable_dim

        self._backend = backend
        self._locations = locations

        if not name.isidentifier():
            raise ValueError("Argument 'name' has to be an identifier")

        self._name = name
        self._label = label
        self._unit = unit

    @classmethod
    def from_array(
        cls,
        array: Union[ArrayLike, da.core.Array],
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        indexable_axes: Optional[Sequence[Union[Axis, ArrayLike]]] = None,
        hidden_axes: Optional[Sequence[Union[Axis, ArrayLike]]] = None,
    ) -> "GridDataset":
        data = da.asanyarray(array)

        data_indexable_axes = []
        if indexable_axes:
            for i, axis in enumerate(indexable_axes):
                if isinstance(axis, Axis):
                    data_indexable_axes.append(axis)
                else:
                    data_indexable_axes.append(Axis(axis, name=f"axis{i}"))
        else:
            for i, entries in enumerate(data.shape):
                data_indexable_axes.append(Axis(np.arange(entries), name=f"axis{i}"))

        data_hidden_axes = []
        if hidden_axes:
            for i, axis in enumerate(hidden_axes, start=len(data_indexable_axes)):
                if isinstance(axis, Axis):
                    data_hidden_axes.append(axis)
                else:
                    data_hidden_axes.append(Axis(axis, name=f"axis{i}"))

        grid_axes = GridAxes(data_indexable_axes, data_hidden_axes)

        return cls(data, grid_axes, name=name, label=label, unit=unit)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "GridDataset":
        file_list = FileList(path, recursive=False)
        dataset = None

        for p in file_list.paths:
            backend = cls.get_valid_backend(p)
            if not backend:
                continue
            backend, grid, name, label, unit, axes = cls._unpack_backend(backend, p)

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

        if not dataset:
            raise ValueError("provided path has no valid files to init GridDataset")

        return dataset

    @staticmethod
    def _unpack_backend(
        backend: GridBackendType, path: Path
    ) -> Tuple[da.Array, str, str, str, GridAxes]:
        grid = backend(path)
        grid_arr = da.from_array(grid)

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
                min_,
                max_,
                ax_pts,
                name=ax_name,
                label=ax_label,
                unit=ax_unit,
            )
            grid_axes.append(ax)

        return (
            grid,
            grid_arr,
            name,
            label,
            unit,
            GridAxes(indexable=grid_axes, hidden=(time, iteration)),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.name}]"

    def _repr_html_(self) -> str:
        general_props = {
            "name": self.name,
            "label": self.label,
            "unit": self.unit or "''",
            "backend": self.backend or "None",
        }
        array_props = {
            "ndim": self.ndim,
            "shape": self.shape,
            "dtype": self.dtype,
        }

        html = (
            Table(
                f"{type(self).__name__}",
                general_props,
                foldable=False,
            ).render_as_html()
            + Table(
                "Array Properties",
                array_props,
            ).render_as_html()
        )

        return html

    def __len__(self) -> int:
        if self._has_appendable_dim:
            return len(self._data)
        else:
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

        axes = GridAxes(grid_axes, time=time, iteration=iteration)

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
        raise NotImplementedError("Setting individual subitems is not supported")

    @property
    def backend(self) -> Optional[str]:
        """Backend associated with instance."""
        return self._backend

    @property
    def locations(self) -> Optional[List[Path]]:
        """Location associated with the source."""
        return self._locations

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
        new = new if isinstance(new, str) else str(new, encoding="utf-8")
        if not new.isidentifier():
            raise ValueError("New name has to be an identifier")

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
    def axes(self) -> GridAxes:
        return self._axes

    def _make_appendable(self) -> "GridDataset":
        if self._has_appendable_dim:
            return self
        else:
            if self._axes.hidden:
                new_indexable = (self._axes.hidden[0],) + self._axes.indexable
                new_hidden = self._axes.hidden[1:]
            else:
                new_indexable = (Axis(0, name="new"),) + self._axes.indexable
                new_hidden = ()

            new_data = self._data[np.newaxis]
            axes = GridAxes(new_indexable, new_hidden)

            return GridDataset(new_data, axes, has_appendable_dim=True)

    def append(self, other: "GridDataset") -> None:
        if not isinstance(other, type(self)):
            raise NotImplementedError(
                f"Appending of '{type(other)}' is not yet supported"
            )

        other = other._make_appendable()
        other_data = other.as_dask()

        if not self._has_appendable_dim:
            self._data = self._data[np.newaxis]
            self._has_appendable_dim = True

            if self._axes.hidden:
                new_indexable = (self._axes.hidden[0],) + self._axes.indexable
                new_hidden = self._axes.hidden[1:]
            else:
                new_indexable = (Axis(0, name="new"),) + self._axes.indexable
                new_hidden = ()

            self._axes = GridAxes(new_indexable, new_hidden)

        self._data = da.concatenate((self._data, other_data), axis=0)

        for self_ax, other_ax in zip(self.axes.indexable, other.axes.indexable):
            self_ax.append(other_ax)

        for self_ax, other_ax in zip(self.axes.hidden, other.axes.hidden):
            self_ax.append(other_ax)

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

    @staticmethod
    def is_valid_backend(backend: GridBackendType) -> bool:
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

    def as_numpy(self) -> np.ndarray:
        return self._data.compute()

    def as_dask(self) -> da.core.Array:
        return self._data


class GridArray(np.lib.mixins.NDArrayOperatorsMixin):
    _backends: Set[GridBackendType] = set()
    _plugin_as_property: Dict[str, Callable] = {}
    _plugin_as_method: Dict[str, Callable] = {}

    def __init__(
        self,
        data: da.Array,
        axes: Tuple[Axis, ...],
        time: Axis,
        name: str,
        label: str,
        unit: str,
    ):
        # name property has to be a valid identifier
        if not name.isidentifier():
            raise ValueError("'name' has to be a valid identifier")

        # time axis has to be unsized
        if time.ndim != 0:
            raise ValueError("time axis has to be 0 dimensional")

        # ensure that every dimension has a corresponding axis
        if len(axes) != data.ndim:
            raise ValueError("number of axes mismatches with dimensionality of data")

        # ensure consistency between .data and .axes
        for ax, s in zip(axes, data.shape):
            if ax.ndim != 1:
                raise ValueError("only 1D axis for GridArray are supported")

            if ax.shape[0] != s:
                raise ValueError("inconsistency between data and axis shape")

        self._data = data
        self._axes = axes
        self._time = time
        self._name = name
        self._label = label
        self._unit = unit

    def __repr__(self) -> str:
        repr_ = (
            "GridArray<"
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"time={self.time.as_numpy()}, "
            f"axes=({', '.join(f'Axis({ax.name})' for ax in self.axes)})"
            ">"
        )
        return repr_

    def __getattribute__(self, name: str) -> Any:
        if name == "_plugin_as_property":
            return super().__getattribute__(name)

        if name == "_plugin_as_method":
            return super().__getattribute__(name)

        if name in self._plugin_as_property:
            return self._plugin_as_property[name](self)

        if name in self._plugin_as_method:
            func = partial(self._plugin_as_method[name], self)
            func.__doc__ = self._plugin_as_method[name].__doc__
            return func

        return super().__getattribute__(name)

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        axes: Optional[Sequence[ArrayLike]] = None,
        time: Optional[Union[Axis, int, float]] = None,
    ) -> "GridArray":
        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if axes is None:
            axes = ()
            for i, l in enumerate(data.shape):
                axes += (Axis(da.arange(l), name=f"axis{i}"),)
        else:
            # ensure that every element in axes is an axis
            if any(not isinstance(ax, Axis) for ax in axes):
                tmp = []
                for i, ax in enumerate(axes):
                    if not isinstance(ax, Axis):
                        ax = Axis(da.asanyarray(ax), name=f"axis{i}")
                    tmp.append(ax)

                axes = tuple(tmp)

        if time is None:
            time = Axis(0.0, name="time", label="time")
        else:
            if not isinstance(time, Axis):
                time = Axis(time, name="time", label="time")

        return cls(data, axes, time, name, label, unit)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def axes(self) -> Tuple[Axis, ...]:
        return self._axes

    @property
    def time(self) -> Axis:
        return self._time

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new: str) -> None:
        new = new if isinstance(new, str) else str(new, encoding="utf-8")
        if not new.isidentifier():
            raise ValueError("name has to be an identifier")
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

    @staticmethod
    def is_valid_backend(backend: GridBackendType) -> bool:
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
        else:
            return None

    @classmethod
    def register_plugin(
        cls,
        plugin_name: str,
        plugin: Callable,
        plugin_type: str = "property",
    ) -> None:
        if (
            plugin_name in cls._plugin_as_property
            or plugin_name in cls._plugin_as_method
        ):
            raise ValueError(f"plugin '{plugin_name}' already registerd")

        if plugin_type not in ("property", "method"):
            raise ValueError("'plugin_type' can only be 'property' or 'method'")

        if not plugin_name.isidentifier():
            raise ValueError("plugin name has to be an identifier")

        if plugin_type == "property":
            cls._plugin_as_property[plugin_name] = plugin
        else:
            cls._plugin_as_method[plugin_name] = plugin

    @classmethod
    def remove_plugin(cls, plugin_name: str) -> None:
        if (
            plugin_name not in cls._plugin_as_property
            and plugin_name not in cls._plugin_as_method
        ):
            raise ValueError(f"plugin '{plugin_name}' not registerd")

        if plugin_name in cls._plugin_as_property:
            del cls._plugin_as_property[plugin_name]
        else:
            del cls._plugin_as_method[plugin_name]

    @classmethod
    def get_plugins(cls) -> Dict[str, str]:
        plugins = {**cls._plugin_as_method, **cls._plugin_as_property}
        return {name: get_doc_heading(plugin) for name, plugin in plugins.items()}
