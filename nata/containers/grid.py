# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import partial
from pathlib import Path
from textwrap import dedent
from types import FunctionType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union
from typing import runtime_checkable
from warnings import warn

import dask.array as da
import ndindex as ndx
import numpy as np
from numpy.typing import ArrayLike

from nata.utils.io import FileList
from nata.utils.types import BasicIndexing
from nata.utils.types import FileLocation

from .axis import Axis
from .utils import get_doc_heading

__all__ = ["GridArray", "GridDataset", "stack"]


def is_unique(iterable: Iterable) -> bool:
    return len(set(iterable)) == 1


def stack(grid_arrs: Sequence["GridArray"]) -> "GridDataset":
    if not len(grid_arrs):
        raise ValueError("can not iterate over 0-length sequence of GridArrays")

    if not is_unique(hash(grid) for grid in grid_arrs):
        raise ValueError("provided GridArrays are not equivalent to each other")

    base = grid_arrs[0]

    name = base.name
    label = base.label
    unit = base.unit

    data = da.stack([grid.to_dask() for grid in grid_arrs])
    time = Axis(
        da.stack([grid.time.to_dask() for grid in grid_arrs]),
        name=base.time.name,
        label=base.time.label,
        unit=base.time.unit,
    )

    axes = []
    for i, ax in enumerate(base.axes):
        axes_data = da.stack([grid.axes[i].to_dask() for grid in grid_arrs])
        axes.append(Axis(axes_data, name=ax.name, label=ax.label, unit=ax.unit))

    return GridDataset(data, (time,) + tuple(axes), name=name, label=label, unit=unit)


class NoValidBackend(Exception):
    pass


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


class _HasGridBackend:
    _backends: Set[GridBackendType] = set()

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

    @staticmethod
    def _unpack_backend(
        backend: GridBackendType, path: Path, time_axis: str
    ) -> Tuple[da.Array, Tuple[Axis, ...], Axis, str, str, str]:
        grid = backend(path)
        data = da.from_array(grid)

        name = grid.dataset_name
        label = grid.dataset_label
        unit = grid.dataset_unit

        if time_axis == "time":
            time = Axis(grid.time_step, name="time", label="time", unit=grid.time_unit)
        else:
            time = Axis(grid.iteration, name="iteration", label="iteration")

        axes = ()
        for min_, max_, ax_pts, ax_name, ax_label, ax_unit in zip(
            grid.axes_min,
            grid.axes_max,
            grid.shape,
            grid.axes_names,
            grid.axes_labels,
            grid.axes_units,
        ):
            axes += (
                Axis.from_limits(
                    min_, max_, ax_pts, name=ax_name, label=ax_label, unit=ax_unit
                ),
            )

        return data, axes, time, name, label, unit


class _HasNumpyInterface(np.lib.mixins.NDArrayOperatorsMixin):
    _handled_array_ufunc: Dict[np.ufunc, Callable]
    _handled_array_function: Dict[FunctionType, Callable]

    _data: da.Array

    @classmethod
    def implements(cls, numpy_function: Union[np.ufunc, FunctionType]):
        def decorator(func):
            if isinstance(numpy_function, np.ufunc):
                cls._handled_array_ufunc[numpy_function] = func
            else:
                cls._handled_array_function[numpy_function] = func

            return func

        return decorator

    @classmethod
    def get_handled_ufuncs(cls) -> Set[np.ufunc]:
        return set(func for func in cls._handled_array_ufunc)

    @classmethod
    def remove_handled_ufuncs(cls, function: np.ufunc) -> None:
        del cls._handled_array_ufunc[function]

    @classmethod
    def get_handled_array_function(cls) -> Set[np.ufunc]:
        return set(func for func in cls._handled_array_function)

    @classmethod
    def remove_handled_array_function(cls, function: np.ufunc) -> None:
        del cls._handled_array_function[function]

    @classmethod
    def from_array(cls, data, *args, **kwargs) -> "_HasNumpyInterface":
        raise NotImplementedError

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        # ensures it is a ndarray -> tries to avoid copying, hence `np.asanyarray`
        arr = self.to_numpy()
        if not isinstance(arr, np.ndarray):
            arr = np.asanyarray(self._data.compute())

        if dtype:
            return arr.astype(dtype)
        else:
            return arr

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Tuple[Any, ...],
        **kwargs: Dict[str, Any],
    ) -> "_HasNumpyInterface":
        if ufunc in self._handled_array_ufunc:
            return self._handled_array_ufunc[ufunc](method, inputs, kwargs)

        # repack inputs to mimic passing as dask array
        inputs = tuple(self._data if input_ is self else input_ for input_ in inputs)

        # required additional repacking if in-place
        if "out" in kwargs:
            output = tuple(self._data if arg is self else arg for arg in kwargs["out"])
            kwargs["out"] = output

        data = self._data.__array_ufunc__(ufunc, method, *inputs, **kwargs)

        # __array_ufunc__ return not implemented by dask
        if data is NotImplemented:
            raise NotImplementedError(f"ufunc '{ufunc}' not implemented!")

        # in-place scenario
        elif data is None:
            self._data = kwargs["out"][0]
            return self
        else:
            return self.from_array(data)

    def __array_function__(
        self,
        func: FunctionType,
        types: Tuple[Type],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> "_HasNumpyInterface":
        if func in self._handled_array_function:
            return self._handled_array_function[func](types, args, kwargs)

        # repack arguments
        types = tuple(type(self._data) if t is type(self) else t for t in types)
        args = tuple(self._data if arg is self else arg for arg in args)
        data = self._data.__array_function__(func, types, args, kwargs)

        return self.from_array(data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def to_dask(self) -> da.Array:
        return self._data

    def to_numpy(self) -> np.ndarray:
        return self._data.compute()


class _HasPluginSystem:
    _plugin_as_property: Dict[str, Callable]
    _plugin_as_method: Dict[str, Callable]

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


class _HasAnnotations:
    _name: str
    _label: str
    _unit: str

    _axes: Tuple[Axis, ...]
    _time: Axis

    def __repr__(self) -> str:
        repr_ = (
            f"{type(self).__name__}<"
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"time={self.time.to_numpy()}, "
            f"axes=({', '.join(f'Axis({ax.name})' for ax in self.axes)})"
            ">"
        )
        return repr_

    def _repr_markdown_(self) -> str:
        md = f"""
        | **{type(self).__name__}** | |
        | ---: | :--- |
        | **name**  | {self.name} |
        | **label** | {self.label} |
        | **unit**  | {self.unit or "''"} |
        | **shape** | {self.shape} |
        | **dtype** | {self.dtype} |
        | **time**  | {self.time.to_numpy()} |
        | **axes**  | {', '.join(f"Axis({ax.name})" for ax in self.axes)} |

        """
        return dedent(md)

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

    @property
    def axes(self) -> Tuple[Axis, ...]:
        return self._axes

    @property
    def time(self) -> Axis:
        return self._time

    def __hash__(self) -> int:
        # general naming
        key = (self.name, self.label, self.unit, self.shape)

        # time specific
        key += (self.time.name, self.time.label, self.time.unit)

        # axes specific
        for ax in self.axes:
            key += (ax.name, ax.label, ax.unit)

        return hash(key)


class GridArray(_HasAnnotations, _HasGridBackend, _HasNumpyInterface, _HasPluginSystem):
    _plugin_as_property: Dict[str, Callable] = {}
    _plugin_as_method: Dict[str, Callable] = {}

    _handled_array_ufunc: Dict[np.ufunc, Callable] = {}
    _handled_array_function: Dict[FunctionType, Callable] = {}

    def __init__(
        self,
        data: da.Array,
        axes: Tuple[Axis, ...],
        time: Axis,
        name: str,
        label: str,
        unit: str,
    ) -> None:
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

    def __getitem__(self, key: Any) -> "GridArray":
        index = ndx.ndindex(key).expand(self.shape).raw

        new_axis_excluded = tuple(ind for ind in index if ind is not None)
        indices_of_new_axis = tuple(i for i, ind in enumerate(index) if ind is None)
        reductions = tuple(isinstance(idx, int) for idx in new_axis_excluded)

        axes = []
        for ax, ind, red in zip(self.axes, new_axis_excluded, reductions):
            if not red:
                axes.append(ax[ind])

        for pos in indices_of_new_axis:
            axes.insert(pos, Axis([0]))

        data = self._data[index]
        axes = tuple(axes)
        time = self.time
        name = self.name
        label = self.label
        unit = self.unit

        return GridArray(data, axes, time, name, label, unit)

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

    @classmethod
    def from_path(cls, path: Union[str, Path], time_axis: str = "time") -> "GridArray":
        if not isinstance(path, Path):
            path = Path(path)

        # check validity of passed arguments
        if not path.is_file():
            raise ValueError("only a single file is supported")

        if time_axis not in ("time", "iteration"):
            raise ValueError("only 'time' and 'iteration' are supported for time axis")

        # get valid backend
        backend = cls.get_valid_backend(path)
        if not backend:
            raise NoValidBackend(f"no valid backend for '{path}' found")

        return cls(*cls._unpack_backend(backend, path, time_axis))


class GridDataset(
    _HasAnnotations, _HasGridBackend, _HasNumpyInterface, _HasPluginSystem
):

    _plugin_as_property: Dict[str, Callable] = {}
    _plugin_as_method: Dict[str, Callable] = {}

    _handled_array_ufunc: Dict[np.ufunc, Callable] = {}
    _handled_array_function: Dict[FunctionType, Callable] = {}

    def __init__(
        self,
        data: da.Array,
        axes: Tuple[Axis, ...],
        name: str,
        label: str,
        unit: str,
    ) -> None:
        # name property has to be a valid identifier
        if not name.isidentifier():
            raise ValueError("'name' has to be a valid identifier")

        # ensure that every dimension has a corresponding axis
        if len(axes) != data.ndim:
            raise ValueError("number of axes mismatches with dimensionality of data")

        # ensure consistency between .data and .axes, e.g.
        # data.shape = (5, 6, 7)
        #    > axes[0].shape = (5,)        =|> corresponds to 'time' axis
        #    > axes[1].shape = (5, 6)      =|> corresponds to 'axis0' axis
        #    > axes[2].shape = (5, 7)      =|> corresponds to 'axis1' axis
        for i, (ax, s) in enumerate(zip(axes, data.shape)):
            if i == 0:
                if ax.ndim != 1:
                    raise ValueError("time axis has to be 1D")
                if ax.shape[0] != s:
                    raise ValueError("inconsistency between data and axis shape")
            else:
                if ax.ndim != 2:
                    raise ValueError("only 2D axis for GridDataset are supported")
                if ax.shape[1] != s:
                    raise ValueError("inconsistency between data and axis shape")

        self._data = data
        self._axes = axes
        self._time = axes[0]
        self._name = name
        self._label = label
        self._unit = unit

    def __getitem__(self, key: Any) -> Union["GridDataset", GridArray]:
        # unpack axes -> remember the first axis is responsible for time slicing
        time_axis, *rest_axes = self.axes

        # unpack indexing
        index = ndx.ndindex(key).expand(self.shape).raw
        time_slicing, *axes_slicing = index

        # nata does not support adding a new axis
        if time_slicing is np.newaxis:
            msg = (
                "creating a new axis as time axis is not supported\n"
                "use `.to_dask` and `.to_numpy` and convert to a GridDataset afterwards"
            )
            raise IndexError(msg)

        new_axis_excluded = tuple(ind for ind in axes_slicing if ind is not None)
        indices_of_new_axis = tuple(
            i for i, ind in enumerate(axes_slicing) if ind is None
        )
        reductions = tuple(isinstance(idx, int) for idx in new_axis_excluded)

        axes = [time_axis[time_slicing]]
        for ax, ind, red in zip(rest_axes, new_axis_excluded, reductions):
            if not red:
                axes.append(ax[time_slicing, ind])

        for pos in indices_of_new_axis:
            axes.insert(pos + 1, Axis(da.zeros((len(time_axis), 1))))

        data = self._data[index]
        name = self.name
        label = self.label
        unit = self.unit

        if axes[0].shape:
            return GridDataset(data, tuple(axes), name, label, unit)
        else:
            time, *axes = axes
            return GridArray(data, tuple(axes), time, name, label, unit)

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        name: str = "unnamed",
        label: str = "unlabeled",
        unit: str = "",
        axes: Optional[Sequence[ArrayLike]] = None,
    ) -> "GridDataset":
        if not isinstance(data, da.Array):
            data = da.asanyarray(data)

        if axes is None:
            axes = ()
            time_steps = None
            for i, l in enumerate(data.shape):
                if i == 0:
                    time_steps = l
                    time = Axis(da.arange(time_steps), name="time", label="time")
                    axes += (time,)
                else:
                    axis_shape = (time_steps, 1)
                    axis = Axis(da.tile(da.arange(l), axis_shape), name=f"axis{i-1}")
                    axes += (axis,)

        else:
            # ensure that every element in axes is an axis
            if any(not isinstance(ax, Axis) for ax in axes):
                tmp = []

                for i, ax in enumerate(axes):
                    name = "time" if i == 0 else f"axis{i-1}"
                    label = "time" if i == 0 else "unlabeled"

                    if not isinstance(ax, Axis):
                        ax = Axis(da.asanyarray(ax), name=name, label=label)

                    tmp.append(ax)

                axes = tuple(tmp)

        return cls(data, axes, name, label, unit)

    @classmethod
    def from_path(
        cls, path: Union[str, Path], time_axis: str = "time"
    ) -> "GridDataset":
        files = FileList(path, recursive=False)

        # go over all the files and
        grid_arrs: Dict[int, List[GridArray]] = defaultdict(list)
        for f in files.paths:
            try:
                grid_arr = GridArray.from_path(f, time_axis=time_axis)
            except NoValidBackend:
                continue

            grid_arrs[hash(grid_arr)].append(grid_arr)

        # convert dict of
        grids = list(grid_arrs.values())

        if not len(grids):
            raise ValueError(f"no valid grid found for '{path}'")

        # warn if multiple grids were found
        if len(grids) > 1:
            warn(f"found multiple grids and picking grid '{grids[0].name}'")

        return stack(grids[0])
