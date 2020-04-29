# -*- coding: utf-8 -*-
from pathlib import Path
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from numpy.lib import recfunctions as rfn

from .axes import Axis
from .axes import GridAxis
from .types import ArrayLike
from .types import AxisType
from .types import DatasetType
from .types import GridAxisType
from .types import GridBackendType
from .types import GridDatasetAxes
from .types import GridDatasetType
from .types import ParticleBackendType
from .types import ParticleDatasetAxes
from .types import QuantityType
from .types import is_basic_indexing
from .utils.array import expand_ellipsis
from .utils.exceptions import NataInvalidContainer
from .utils.formatting import array_format
from .utils.formatting import make_identifiable

_extract_from_backend = object()
_extract_from_data = object()


def _separation_newaxis(key, two_types=True):
    if two_types:
        # find last occupance of newaxis
        for i, k in enumerate(key):
            if k is np.newaxis:
                continue
            else:
                break

        type_1 = key[:i]
        key = key[i:]
    else:
        type_1 = ()

    # determine positions of new axis for second type
    type_2 = tuple([k for k, v in enumerate(key) if v is np.newaxis])
    key = tuple(filter(lambda v: v is not np.newaxis, key))

    return key, type_1, type_2


def _transform_particle_data_array(data: np.ndarray):
    """Transform a array into a required particle data array.

    Data is assumed to fulfill the condition `data.ndim =< 3` for
    unstructured and `data.ndim =< 2` for structured array.

    Array specification:
        * axis == 0: time/iteration
        * axis == 1: particle index
        * dtype: [('q0', 'type'), ('q1', 'type')]
            -> 'q0/q1' represent quantity names
            -> 'type' represents the type of dtype (e.g. int, float, ...)
    """
    if data.ndim == 2 and data.dtype.fields:
        return data

    # data has fields
    if data.dtype.fields:
        if data.ndim == 0:
            data = data[np.newaxis, np.newaxis]
        else:  # data.ndim == 1
            data = data[np.newaxis]

    # data has not fields -> is associated with quantity index
    else:
        if data.ndim == 0:
            data = data[(np.newaxis,) * 3]
        elif data.ndim == 1:
            data = data[np.newaxis, ..., np.newaxis]
        elif data.ndim == 2:
            data = data[..., np.newaxis]

        field_names = [f"quant{i}" for i in range(data.shape[-1])]
        data = rfn.unstructured_to_structured(data, names=field_names)

    return data


def _convert_to_backend(dataset: DatasetType, data: Union[str, Path]):
    if isinstance(data, str):
        data = Path(data)

    if isinstance(data, Path):
        for backend in dataset.get_backends().values():
            if backend.is_valid_backend(data):
                data = backend(data)
                break
        else:
            raise NataInvalidContainer(f"No valid backend found for '{data}'")

    return data


def _axis_type_reduction(value):
    if value is not None and not isinstance(value, (AxisType, np.ndarray)):
        return np.asanyarray(value)
    else:
        return value


def _value_to_axis_transformation(value, name_and_label, unit, none_value):
    requires_new_axis = False

    if value is None:
        requires_new_axis = True
        value = Axis(
            none_value, name=name_and_label, label=name_and_label, unit=unit
        )
    elif isinstance(value, np.ndarray):
        value = Axis(
            value, name=name_and_label, label=name_and_label, unit=unit
        )

    return value, requires_new_axis


def _grid_axis_transformation(axes, temporal_steps, axes_cells):
    if axes is None:
        axes = []
        for i, cells in enumerate(axes_cells):
            data = [np.arange(cells)] * temporal_steps
            axes.append(
                GridAxis(data, name=f"axis{i}", label=f"axis{i}", unit="")
            )
    else:
        for i, axis in enumerate(axes):
            if isinstance(axis, GridAxisType):
                continue
            else:
                axis = np.asanyarray(axis)
                if temporal_steps == 1:
                    axis = axis[np.newaxis]

                if (temporal_steps, axes_cells[i]) != axis.shape:
                    raise ValueError("Dimension for axis do not match!")

                axes[i] = GridAxis(
                    axis,
                    name=f"axis{i}",
                    label=f"axis{i}",
                    unit="",
                    axis_type="custom",
                )

    return axes


class GridDataset(np.lib.mixins.NDArrayOperatorsMixin):
    """Container holding data associated with a grid."""

    _backends: AbstractSet[GridBackendType] = set()
    # TODO: special treatment of ufuncs and array_function is not yet supported
    _handled_ufuncs = {}
    _handled_array_function = {}

    def __init__(
        self,
        data: Union[np.ndarray, GridBackendType, str, Path],
        *,
        iteration: Optional[AxisType] = _extract_from_backend,
        time: Optional[AxisType] = _extract_from_backend,
        grid_axes: Sequence[Optional[GridAxisType]] = _extract_from_backend,
        name: str = _extract_from_backend,
        label: str = _extract_from_backend,
        unit: str = _extract_from_backend,
    ):
        """For initialization of a :class:`.GridDataset`:

        Parameters
        ----------
        data: :class:`numpy.ndarray`, :class:`nata.type.GridBackendType`, \
              :class:`str`, :class:`pathlib.Path`
            Input data which is used to initialize the grid container:

            * If it is a numpy array or array-like, the first dimension is
              consumed and is considered to represent time/iteration.
            * If it is a backend, then it must follow the
              :class:`nata.types.GridBackendType` Protocol to be recognized
              as such.
            * If it is a string, then it will be converted to a path and is
              equivalent as passing the string as a path.
            * If it is a path, then it will be used to initialize a backend
              based on registered backends for :class:`.GridDatasets`. The
              first matching backend is used and if no valid backend was
              found, an exception is raised.

        iteration: ``None`` or :class:`nata.types.AxisType`, \
                   extract from ``data`` if not provided
            Keyword-only argument which represents the iteration axis for the
            container. If not provided, it is extracted from ``data``. If
            ``data`` is a valid backend, an iteration axis is created and
            otherwise the iteration axis is ``None``.

        time: ``None`` or :class:`nata.types.AxisType`, extract from ``data`` \
              if not provided
            Keyword-only argument which represents the time axis for the
            container. If not provided, it is extracted from ``data``. If
            ``data`` is a valid backend, an time axis is created and
            otherwise the time axis is ``None``.

        grid_axes: ``None`` or Sequence of :class:`nata.types.AxisType`,  \
                   extract from ``data`` if not provided
            Keyword-only argument which represents a sequence of grid axes
            for the container. If not provided, it is extracted from
            ``data``. If ``data`` is a valid backend, a sequence og grid axes
            is created and otherwise the sequence contains ``None``.

        name: ``str``, extract from ``data`` if not provided
            Keyword-only argument which represents the name of the grid
            container. If not possible to extract a valid name from ``data``,
            a default string ``'unnamed'`` is used.

        label: ``str``, extract from ``data`` if not provided
            Keyword-only argument which represents the label of the grid
            container. If not possible to extract a valid label from ``data``,
            a default string ``'unnamed'`` is used.

        unit: ``str``, extract from ``data`` if not provided
            Keyword-only argument which represents the unit of the grid
            container. If not possible to extract a valid unit from ``data``,
            an empty string is used.

        Raises
        ------
        :class:`nata.utils.exceptions.NataInvalidContainer``:
            If ``data`` is a string or a path and no valid backend is found.
        """
        if isinstance(data, str):
            data = Path(data)

        if isinstance(data, Path):
            for backend in self._backends:
                if backend.is_valid_backend(data):
                    data = backend(data)
                    break
            else:
                raise NataInvalidContainer(
                    f"No valid backend found for '{data}'"
                )

        if isinstance(data, GridBackendType):
            self._backend = data.name
        else:
            self._backend = None

        if iteration is _extract_from_backend:
            if isinstance(data, GridBackendType):
                iteration = Axis(
                    data.iteration, name="iteration", label="iteration", unit=""
                )
            else:
                iteration = None

        if time is _extract_from_backend:
            if isinstance(data, GridBackendType):
                time = Axis(
                    data.time_step,
                    name="time",
                    label="time",
                    unit=data.time_unit,
                )
            else:
                time = None

        if grid_axes is _extract_from_backend:
            if isinstance(data, GridBackendType):
                grid_axes = []
                for (
                    axis_name,
                    axis_label,
                    axis_unit,
                    min_,
                    max_,
                    grid_points,
                ) in zip(
                    data.axes_names,
                    data.axes_labels,
                    data.axes_units,
                    data.axes_min,
                    data.axes_max,
                    data.shape,
                ):
                    grid_axes.append(
                        GridAxis.from_limits(
                            min_,
                            max_,
                            grid_points,
                            name=axis_name,
                            label=axis_label,
                            unit=axis_unit,
                        )
                    )
            else:
                grid_axes = [None] * (np.ndim(data) - 1)

        # TODO: make it an identifier
        if name is _extract_from_backend:
            if isinstance(data, GridBackendType):
                name = data.dataset_name
            else:
                name = "unnamed"

        if label is _extract_from_backend:
            if isinstance(data, GridBackendType):
                label = data.dataset_label
            else:
                label = "unnamed"

        if unit is _extract_from_backend:
            if isinstance(data, GridBackendType):
                unit = data.dataset_unit
            else:
                unit = ""

        self._name = name
        self._label = label
        self._unit = unit
        self._axes = {
            "iteration": iteration,
            "time": time,
            "grid_axes": grid_axes,
        }

        if isinstance(data, GridBackendType):
            self._dtype = data.dtype
            self._grid_shape = data.shape
            self._data = np.asanyarray(data, dtype=object)[np.newaxis]
        else:
            data = np.asanyarray(data)
            self._dtype = data.dtype
            self._grid_shape = data.shape[1:] if data.ndim > 0 else ()
            self._data = data if data.ndim > 0 else data[np.newaxis]

    def __repr__(self):
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"shape={self.shape}, "

        iteration_axis = self.axes["iteration"]
        if isinstance(iteration_axis, AxisType):
            repr_ += f"iteration={array_format(iteration_axis.data)}, "
        else:
            repr_ += f"iteration={iteration_axis}, "

        time_axis = self.axes["time"]
        if isinstance(time_axis, AxisType):
            repr_ += f"time={array_format(time_axis.data)}, "
        else:
            repr_ += f"time={time_axis}, "

        grid_axes = self.axes["grid_axes"]
        if grid_axes:
            axes_formmating = []
            for axis in grid_axes:
                if isinstance(axis, AxisType):
                    axes_formmating.append(
                        f"Axis('{axis.name}', "
                        + f"len={len(axis)}, "
                        + f"shape={axis.shape})"
                    )
                else:
                    axes_formmating.append(f"{axis}")
            repr_ += f"grid_axes=[{', '.join(axes_formmating)}]"
        else:
            repr_ += f"grid_axes={self.axes['grid_axes']}"

        repr_ += ")"

        return repr_

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> "GridDataset":
        if len(self) == 1:
            yield self
        else:
            for d, it, t in zip(
                self._data, self.axes["iteration"], self.axes["time"]
            ):
                yield self.__class__(
                    d,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                    iteration=it,
                    time=t,
                    grid_axes=self.axes["grid_axes"],
                )

    def __getitem__(
        self, key: Union[int, slice, Tuple[int, slice]]
    ) -> "GridDataset":
        if not is_basic_indexing(key):
            raise IndexError("Only basic indexing is supported!")

        # expand ellipsis to all dimensions -> newaxis are added
        key = expand_ellipsis(key, self.ndim)

        # >>>> data
        # > determine if new axis extension is required
        new_axis_for_data = False
        if len(self) != 1:
            # revert dimensionality reduction
            if isinstance(key[0], int):
                new_axis_for_data = True
        else:
            new_axis_for_data = True

        data = self.data[key]

        if new_axis_for_data:
            data = data[np.newaxis]

        # >>>> separate new axis from key
        key, temporal_new_axis, grid_new_axis = _separation_newaxis(
            key, two_types=len(self) != 1
        )

        # >>>> iteration/time axis
        # string is the best way to insure not overlapping with key
        # -> 'np.newaxis is None == True' therefore we use a string
        temporal_indexing = "None"
        if len(self) != 1:
            temporal_indexing = key[0]

        time = (
            self.axes["time"][temporal_indexing]
            if temporal_indexing != "None"
            else self.axes["time"]
        )
        iteration = (
            self.axes["iteration"][temporal_indexing]
            if temporal_indexing != "None"
            else self.axes["iteration"]
        )

        if temporal_new_axis:
            time = time[temporal_new_axis]
            iteration = iteration[temporal_new_axis]

        # >>>> grid_axes
        index_for_grid_axes = [slice(None) for _ in self.axes["grid_axes"]]
        if len(self) == 1:
            for i, k in enumerate(key):
                index_for_grid_axes[i] = k
        else:
            for i, k in enumerate(key[1:]):
                index_for_grid_axes[i] = k

        grid_axes = []
        for index, grid_axis in zip(
            index_for_grid_axes, self.axes["grid_axes"]
        ):
            # dimension will be reduced
            if isinstance(index, int):
                continue

            if temporal_indexing == "None":
                grid_axes.append(grid_axis[index])
            else:
                grid_axes.append(grid_axis[temporal_indexing, index])

        for i in grid_new_axis:
            if len(self) == 1:
                # shift not required as no time index is included
                grid_axes.insert(i, None)
            else:
                grid_axes.insert(i - 1, None)

        # finally return the reduced data entries
        return self.__class__(
            data,
            iteration=iteration,
            time=time,
            grid_axes=grid_axes,
            name=self.name,
            label=self.label,
            unit=self.unit,
        )

    def __setitem__(
        self,
        key: Union[int, slice, Tuple[int, slice]],
        value: Union[np.ndarray, float, int],
    ) -> None:
        self.data[key] = value

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if self._data.dtype == object:
            data = np.empty((len(self),) + self._grid_shape, dtype=self.dtype)
            for i, d in enumerate(self._data):
                if isinstance(d, np.ndarray):
                    data[i] = d
                else:
                    data[i] = d.get_data(indexing=None)

            self._data = data

        # check for length is required as np.squeeze raises ValueError
        if len(self) == 1:
            return np.squeeze(self._data, axis=0)
        else:
            return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in self._handled_ufuncs:
            return self._handled_ufuncs[ufunc](method, *inputs, **kwargs)
        elif method == "__call__":
            out = kwargs.pop("out", ())

            # repack all inputs and use `.data` if its of type GridDataset
            new_inputs = []
            for input_ in inputs:
                if isinstance(input_, self.__class__):
                    new_inputs.append(input_.data)
                else:
                    new_inputs.append(input_)

            new_data = ufunc(*new_inputs, **kwargs)
            new_data = new_data[np.newaxis] if len(self) == 1 else new_data

            # should only occur if in-place operation are occuring
            if out and isinstance(out[0], self.__class__):
                self.data = new_data
                return self
            else:
                return self.__class__(
                    new_data,
                    iteration=self.axes["iteration"],
                    time=self.axes["time"],
                    grid_axes=self.axes["grid_axes"],
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                )
        else:
            raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        if func in self._handled_array_function:
            return self._handled_array_function[func](*args, **kwargs)
        else:
            new_args = []
            for arg in args:
                if isinstance(arg, self.__class__):
                    new_args.append(arg.data)
                else:
                    new_args.append(arg)

            new_data = func(*new_args, **kwargs)
            # func can return something which is not an array
            if not isinstance(new_data, np.ndarray):
                return new_data

            new_data = new_data[np.newaxis] if len(self) == 1 else new_data

            return self.__class__(
                new_data,
                iteration=self.axes["iteration"],
                time=self.axes["time"],
                grid_axes=self.axes["grid_axes"],
                name=self.name,
                label=self.label,
                unit=self.unit,
            )

    # TODO: allow for dictionaries for time, iteration, and grid_axes, e.g.:
    #       time={
    #           "value": [0, 1, 2],
    #           "name":  "some_time",
    #           "label": "some_time_label",
    #           "unit":  "some_time_unit"
    #       }
    @classmethod
    def from_array(
        cls,
        array: ArrayLike,
        name: str = "unnamed",
        label: str = "unnamed",
        unit: str = "",
        time: Optional[Union[AxisType, ArrayLike]] = None,
        iteration: Optional[Union[AxisType, ArrayLike]] = None,
        grid_axes: Optional[Sequence[Union[GridAxisType, ArrayLike]]] = None,
    ):
        """Initialize GridDataset from an array.

        As in general, :class:`.GridDataset` container provide a rich-API,
        :meth:`.GridDataset.from_array` allows to create naively a object
        with the source data coming from a numpy array and pre-defined objects
        for the axes.

        Parameters
        ----------
        array : array-like object
            Input data, in any form that can be converted to an numpy array
            or a numpy array itself. The array dimension correspond the grid
            dimension if no temporal information is provided. Otherwise, the
            first dimension is consumed.

        name : :obj:`str`, default value: ``"unnamed"``
            Name of the grid container and expected to be identifiable.

        label : :obj:`str`, default value: ``"unnamed"``
            Label of the grid container with a descriptive meaning. It is not
            expected to be identifiable.

        unit : :obj:`str`, default value: ``""``
            Unit of the grid container.

        time : array-like, axis object, optional
            Time axis of the grid container. If an array-like object is
            provided, an axis object is created underneath. In addition, an
            axis object can be provided which has to fulfill the
            :py:class:`nata.types.AxisType` protocol. If nothing is provided
            (default option), a time axis with the single value ``0.0`` is
            created.

        iteration : array-like, axis object, optional
            Iteration axis of the grid container. If an array-like object is
            provided, an axis object is created underneath. In addition, an
            axis object can be provided which has to fulfill the
            :py:class:`nata.types.AxisType` protocol. If nothing is provided
            (default option), a iteration axis with a single value ``0`` is
            created.

        grid_axes : sequence of array like objects and/or grid axis objects, \
                    optional
            Sequence characterizing each grid axis. The length of the
            sequence has to correspond to the dimension of the array. In the
            absence of temporal axes, the length corresponds to the array
            dimension otherwise the first axes of the array is consumed.
        """
        data = np.asanyarray(array)

        # XOR of time and iteration
        # -> use others values
        if (time is None and iteration is not None) or (
            time is not None and iteration is None
        ):
            if time is None:
                time = iteration
            else:
                iteration = time

        time = _axis_type_reduction(time)
        time, add_temporal_axis = _value_to_axis_transformation(
            time, "time", "", 0.0
        )

        iteration = _axis_type_reduction(iteration)
        iteration, add_temporal_axis = _value_to_axis_transformation(
            iteration, "iteration", "", 0
        )

        if add_temporal_axis:
            data = data[np.newaxis]

        # now we can check that
        if not (len(data) == len(time) == len(iteration)):
            raise ValueError(
                f"Creating an '{cls.__name__}' from array failed! "
                + "Temporal dimension mismatch for data, time, and iteration"
            )

        temporal_steps, *shape_grid_axes = data.shape

        grid_axes = _grid_axis_transformation(
            grid_axes, temporal_steps, shape_grid_axes
        )

        if any(len(data) != len(axis) for axis in grid_axes):
            raise ValueError(
                f"Creating an '{cls.__name__}' from array failed! "
                + "Temporal dimension mismatch for data and grid_axes"
            )

        return cls(
            data,
            iteration=iteration,
            time=time,
            grid_axes=grid_axes,
            name=name,
            label=label,
            unit=unit,
        )

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

    @property
    def backend(self) -> Optional[str]:
        """Backend associated with instance."""
        return self._backend

    @property
    def data(self) -> np.ndarray:
        """Underlaying numpy array which stores the data for the grid."""
        return self.__array__()

    @data.setter
    def data(self, value: Union[np.ndarray, Any]) -> None:
        value = np.asanyarray(value)
        if value.shape != self.shape:
            raise ValueError(
                f"Shapes inconsistent {self.shape} -> {value.shape}"
            )
        self._data = value
        self._dtype = value.dtype

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        if len(self) == 1:
            shape = self.grid_shape
        else:
            shape = (len(self),) + self.grid_shape

        return shape

    @property
    def ndim(self) -> int:
        if len(self) == 1:
            return len(self.grid_shape)
        else:
            return len(self.grid_shape) + 1

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new) -> None:
        self._name = str(new)

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, new) -> None:
        self._label = str(new)

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, new) -> None:
        self._unit = str(new)

    @property
    def axes(self) -> GridDatasetAxes:
        return self._axes

    @property
    def grid_shape(self) -> Tuple[int]:
        return self._grid_shape

    def equivalent(self, other: Union[Any, GridDatasetType]) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attr in ["name", "label", "unit"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if (self.axes["iteration"] is not None) and (
            not self.axes["iteration"].equivalent(other.axes["iteration"])
        ):
            return False

        if (self.axes["time"] is not None) and (
            not self.axes["time"].equivalent(other.axes["time"])
        ):
            return False

        for grid_axis, other_grid_axis in zip(
            self.axes["grid_axes"], other.axes["grid_axes"]
        ):
            if grid_axis is None:
                if other_grid_axis is None:
                    continue
                else:
                    return False
            else:
                if not grid_axis.equivalent(other_grid_axis):
                    return False

        return True

    def append(self, other: Union[Any, GridDatasetType]) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Can not append "
                + f"'{type(other).__name__}' to '{type(self).__name__}'"
            )

        if not self.equivalent(other):
            raise ValueError("GridDatasets are not equivalent")

        if self.axes["iteration"]:
            self.axes["iteration"].append(other.axes["iteration"])
        if self.axes["time"]:
            self.axes["time"].append(other.axes["time"])

        if self.axes["grid_axes"]:
            for self_grid_axis, other_grid_axis in zip(
                self.axes["grid_axes"], other.axes["grid_axes"]
            ):
                self_grid_axis.append(other_grid_axis)

        self._data = np.append(self._data, other._data, axis=0)

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)


class ParticleQuantity(np.lib.mixins.NDArrayOperatorsMixin):
    # TODO: special treatment of ufuncs and array_function is not yet supported
    _handled_ufuncs = {}
    _handled_array_function = {}

    def __init__(
        self,
        data: Union[np.ndarray, ParticleBackendType],
        *,
        name: str = "",
        label: str = "",
        unit: str = "",
        particles: Union[int, np.ndarray] = _extract_from_data,
        dtype: np.dtype = _extract_from_data,
    ) -> None:

        if not isinstance(data, np.ndarray):
            data = np.asanyarray(data)

        # data.shape is 2d -> (#iteration, particle_id) unless delayed reading
        if data.ndim == 0 and data.dtype == object:
            data = data[np.newaxis]
        elif data.ndim == 0 and data.dtype != object:
            data = data[np.newaxis, np.newaxis]
        elif data.ndim == 1 and data.dtype != object:
            data = data[np.newaxis]

        self._data = data

        # particle number required if delayed reading
        if data.dtype == object and particles is _extract_from_data:
            raise ValueError("Number of particles required delayed reading!")
        elif data.dtype == object:
            self._num_prt = np.array([particles])
        else:
            self._num_prt = np.array([data.shape[1]])

        # dtype is required for delayed reading
        if data.dtype == object and dtype is _extract_from_data:
            raise ValueError("Number of particles required delayed reading!")
        elif data.dtype != object:
            dtype = data.dtype

        self._dtype = dtype

        name = make_identifiable(name)
        self._name = name if name else "unnamed"
        self._label = label
        self._unit = unit

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"label='{self.label}', "
        repr_ += f"unit='{self.unit}', "
        repr_ += f"len={len(self)}"
        repr_ += ")"

        return repr_

    def __iter__(self) -> Iterable["ParticleQuantity"]:
        for d, num in zip(self._data, self._num_prt):
            yield self.__class__(
                d,
                name=self.name,
                label=self.label,
                unit=self.unit,
                particles=num,
                dtype=self.dtype,
            )

    def __getitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]]
    ) -> "ParticleQuantity":
        if not is_basic_indexing(key):
            raise IndexError("Only basic indexing is supported!")

        index = np.index_exp[key]
        requires_new_axis = False

        # > determine if axis extension is required
        # 1st index (temporal slicing) not hidden if ndim == axis_dim + 1
        if len(self) != 1:
            # revert dimensionality reduction
            if isinstance(index[0], int):
                requires_new_axis = True
        else:
            requires_new_axis = True

        data = self.data[index]

        if requires_new_axis:
            data = data[np.newaxis]

        return self.__class__(
            data,
            name=self.name,
            label=self.label,
            unit=self.unit,
            particles=self._num_prt,
        )

    def __setitem__(
        self, key: Union[int, slice, Tuple[Union[int, slice]]], value: Any
    ) -> None:
        self.data[key] = value

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        if self._data.dtype == object:
            max_ = np.max(self._num_prt)
            data = np.ma.empty((len(self._data), max_), dtype=self._dtype)
            data.mask = np.ones((len(self._data), max_), dtype=np.bool)

            for i, (d, entries) in enumerate(zip(self._data, self._num_prt)):
                if isinstance(d, np.ndarray):
                    data[i, :entries] = d[:entries]
                else:
                    data[i, :entries] = d.get_data(fields=self.name)

                data.mask[i, entries:] = np.zeros(max_ - entries, dtype=np.bool)

            self._data = data

        return np.squeeze(self._data, axis=0) if len(self) == 1 else self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in self._handled_ufuncs:
            return self._handled_ufuncs[ufunc](method, *inputs, **kwargs)
        elif method == "__call__":
            out = kwargs.pop("out", ())

            # repack all inputs and use `.data` if its of type GridDataset
            new_inputs = []
            for input_ in inputs:
                if isinstance(input_, self.__class__):
                    new_inputs.append(input_.data)
                else:
                    new_inputs.append(input_)

            new_data = ufunc(*new_inputs, **kwargs)
            new_data = new_data[np.newaxis] if len(self) == 1 else new_data

            # should only occur if in-place operation are occuring
            if out and isinstance(out[0], self.__class__):
                self.data = new_data
                return self
            else:
                return self.__class__(
                    new_data,
                    name=self.name,
                    label=self.label,
                    unit=self.unit,
                    particles=self._num_prt,
                )
        else:
            raise NotImplementedError

    def __array_function__(self, func, types, args, kwargs):
        if func in self._handled_array_function:
            return self._handled_array_function[func](*args, **kwargs)
        else:
            new_args = []
            for arg in args:
                if isinstance(arg, self.__class__):
                    new_args.append(arg.data)
                else:
                    new_args.append(arg)

            new_data = func(*new_args, **kwargs)
            new_data = new_data[np.newaxis] if len(self) == 1 else new_data

            return self.__class__(
                new_data,
                name=self.name,
                label=self.label,
                unit=self.unit,
                particles=self._num_prt,
            )

    @property
    def ndim(self) -> int:
        if len(self) == 1:
            return 1
        else:
            return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        if len(self) == 1:
            return (np.max(self._num_prt),)
        else:
            return (len(self), np.max(self._num_prt))

    @property
    def num_particles(self) -> np.ndarray:
        return np.squeeze(self._num_prt)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def data(self) -> np.ndarray:
        return np.asanyarray(self)

    @data.setter
    def data(self, value: Union[np.ndarray, Any]) -> None:
        new = np.broadcast_to(value, self.shape, subok=True)
        if len(self) == 1:
            self._data = np.array(new, subok=True)[np.newaxis]
        else:
            self._data = np.array(new, subok=True)

        self._dtype = self._data.dtype

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value):
        parsed_value = make_identifiable(str(value))
        if not parsed_value:
            raise ValueError(
                "Invalid name provided! Has to be able to be valid code"
            )
        self._name = parsed_value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        value = str(value)
        self._label = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        value = str(value)
        self._unit = value

    def equivalent(self, other: "ParticleQuantity") -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if self.label != other.label:
            return False

        if self.unit != other.unit:
            return False

        return True

    def append(self, other: "ParticleQuantity") -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can not append '{other}' to '{self}'")

        if not self.equivalent(other):
            raise ValueError(
                f"Mismatch in attributes between '{self}' and '{other}'"
            )

        self._data = np.append(self._data, other._data, axis=0)
        self._num_prt = np.append(self._num_prt, other._num_prt)


class ParticleDataset:
    _backends: AbstractSet[ParticleBackendType] = set()

    def __init__(
        self,
        data: Optional[
            Union[ParticleBackendType, np.ndarray, str, Path]
        ] = None,
        *,
        name: str = _extract_from_backend,
        iteration: Optional[AxisType] = _extract_from_backend,
        time: Optional[AxisType] = _extract_from_backend,
        # TODO: remove quantities -> use it with data together
        quantities: Mapping[str, QuantityType] = _extract_from_backend,
    ):
        if data is None and quantities is _extract_from_backend:
            raise ValueError("Requires either data or quantities being set!")

        # conversion to a valid backend
        if isinstance(data, (str, Path)):
            data = _convert_to_backend(self, data)

        # ensure data is only a valid backend for particles or numpy array
        # TODO: assumed that dtype object corresponds to particle dataset
        data = np.asanyarray(data)

        if data.ndim > 3 or (data.dtype.fields and data.ndim > 2):
            raise ValueError(
                "Unsupported data dimensionality! "
                + "Dimensionality of the data has to be <= 3 "
                + "for an unstructured array "
                + "or <= 2 for a structured array!"
            )

        if data.dtype != object:
            data = _transform_particle_data_array(data)

        if name is _extract_from_backend:
            if data.dtype == object:
                name = data.item().dataset_name
            else:
                name = "unnamed"

        self._name = name

        if iteration is _extract_from_backend:
            if data.dtype == object:
                iteration = Axis(
                    data.item().iteration,
                    name="iteration",
                    label="iteration",
                    unit="",
                )
            else:
                iteration = None

        if time is _extract_from_backend:
            if data.dtype == object:
                time = Axis(
                    data.item().time_step,
                    name="time",
                    label="time",
                    unit=data.item().time_unit,
                )
            else:
                iteration = None
                time = None

        self._axes = {"time": time, "iteration": iteration}

        if quantities is _extract_from_backend:
            quantities = {}

            if data.dtype == object:
                for name, label, unit in zip(
                    data.item().quantity_names,
                    data.item().quantity_labels,
                    data.item().quantity_units,
                ):
                    quantities[name] = ParticleQuantity(
                        data.item(),
                        name=name,
                        label=label,
                        unit=unit,
                        particles=data.item().num_particles,
                        dtype=data.item().dtype[name],
                    )

                num_particles = data.item().num_particles

            else:
                quantity_names = [f for f in data.dtype.fields.keys()]
                quantity_labels = quantity_names
                quantity_units = [""] * len(quantity_names)

                for name, label, unit in zip(
                    quantity_names, quantity_labels, quantity_units
                ):
                    quantities[name] = ParticleQuantity(
                        data[name], name=name, label=label, unit=unit
                    )

                num_particles = data.shape[1]

        else:
            q = next(iter(quantities.values()))
            num_particles = np.full(
                shape=len(q), fill_value=q.num_particles, dtype=int
            )

        self._quantities = quantities
        self._num_particles = Axis(
            num_particles,
            name="num_particles",
            label="num. of particles",
            unit="",
        )

    def __repr__(self) -> str:
        repr_ = f"{self.__class__.__name__}("
        repr_ += f"name='{self.name}', "
        repr_ += f"len={len(self)}, "
        repr_ += f"quantities={[q for q in self.quantities]}"
        repr_ += ")"

        return repr_

    def __len__(self) -> int:
        return len(self.num_particles)

    def __getitem__(
        self, key: Union[int, slice, Tuple[int, slice]] = None
    ) -> "ParticleDataset":
        # >>>> iteration/time axis
        time = self.axes["time"][key] if key is not None else self.axes["time"]
        iteration = (
            self.axes["iteration"][key]
            if key is not None
            else self.axes["iteration"]
        )
        quantities = {
            quant.name: quant[key] for quant in self.quantities.values()
        }

        # finally return the reduced data entries
        return self.__class__(
            iteration=iteration,
            time=time,
            name=self.name,
            quantities=quantities,
        )

    @property
    def axes(self) -> ParticleDatasetAxes:
        return self._axes

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantities(self) -> Dict[str, QuantityType]:
        return self._quantities

    @property
    def num_particles(self) -> AxisType:
        return self._num_particles

    @classmethod
    def add_backend(cls, backend: ParticleBackendType) -> None:
        if cls.is_valid_backend(backend):
            cls._backends.add(backend)
        else:
            raise ValueError("Invalid backend provided")

    @classmethod
    def remove_backend(cls, backend: ParticleBackendType) -> None:
        cls._backends.remove(backend)

    @classmethod
    def is_valid_backend(cls, backend: ParticleBackendType) -> bool:
        return isinstance(backend, ParticleBackendType)

    @classmethod
    def get_backends(cls) -> Dict[str, ParticleBackendType]:
        backends_dict = {}
        for backend in cls._backends:
            backends_dict[backend.name] = backend
        return backends_dict

    def append(self, other: "ParticleDataset") -> None:
        if not self.equivalent(other):
            raise ValueError(
                f"Can not append '{other}' particle datasets are unequal!"
            )

        if self.axes["iteration"]:
            self.axes["iteration"].append(other.axes["iteration"])
        if self.axes["time"]:
            self.axes["time"].append(other.axes["time"])

        for quant_name in self.quantities:
            self.quantities[quant_name].append(other.quantities[quant_name])
        self.num_particles.append(other.num_particles)

    def equivalent(self, other: "ParticleDataset") -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if set(self.quantities.keys()) ^ set(other.quantities.keys()):
            return False

        for quant_name in self.quantities:
            if not self.quantities[quant_name].equivalent(
                other.quantities[quant_name]
            ):
                return False

        if not self.num_particles.equivalent(other.num_particles):
            return False

        return True

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)


class DatasetCollection:
    _container_set = set([GridDataset, ParticleDataset])

    def __init__(self, root_path: Union[str, Path]) -> None:
        self.root_path = (
            root_path if isinstance(root_path, Path) else Path(root_path)
        )
        self.store = dict()

    def __repr__(self) -> str:
        try:
            path = self.root_path.relative_to(Path().absolute())
        except ValueError:
            path = self.root_path

        repr_ = f"{self.__class__.__name__}("
        repr_ += f"root_path='{path}', "
        repr_ += f"stored={[k for k in self.store]}"
        repr_ += ")"

        return repr_

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    @property
    def datasets(self):
        return np.array([k for k in self.store.keys()], dtype=str)

    def _append_datasetcollection(self, obj):
        self.store.update(obj.store)

    def _append_file(self, obj):
        for container in self._container_set:
            try:
                dataset = container(obj)
                break
            except NataInvalidContainer:
                continue
        else:
            # not possible to append the file -> not a valid container found
            return

        if dataset.name in self.store:
            existing_ds = self.store[dataset.name]
            existing_ds.append(dataset)
        else:
            self.store[dataset.name] = dataset

    def append(self, obj: Union[str, Path, "DatasetCollection"]) -> None:
        """Takes a path to a diagnostic and appends it to the collection."""
        if isinstance(obj, DatasetCollection):
            self._append_datasetcollection(obj)

        elif isinstance(obj, (str, Path)):
            self._append_file(obj)

        else:
            raise ValueError(
                f"Can not append object of type '{type(obj)}' to collection"
            )

    def __getitem__(self, key):
        return self.store[key]

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)
