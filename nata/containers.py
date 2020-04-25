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
        else:
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


class GridDataset:
    _backends: AbstractSet[GridBackendType] = set()

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

    @classmethod
    def add_backend(cls, backend: GridBackendType) -> None:
        if cls.is_valid_backend(backend):
            cls._backends.add(backend)
        else:
            raise ValueError("Invalid backend provided")

    @classmethod
    def remove_backend(cls, backend: GridBackendType) -> None:
        cls._backends.remove(backend)

    @classmethod
    def is_valid_backend(cls, backend: GridBackendType) -> bool:
        return isinstance(backend, GridBackendType)

    @classmethod
    def get_backends(cls) -> Dict[str, GridBackendType]:
        backends_dict = {}
        for backend in cls._backends:
            backends_dict[backend.name] = backend
        return backends_dict

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def data(self) -> np.ndarray:
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


class ParticleQuantity:
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
                d[np.newaxis],
                name=self.name,
                label=self.label,
                unit=self.unit,
                particles=num,
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
        axes: ParticleDatasetAxes = _extract_from_backend,
        # TODO: remove quantaties -> use it with data together
        quantities: Mapping[str, QuantityType] = _extract_from_backend,
    ):
        if data is None and quantities is _extract_from_backend:
            raise ValueError("Requires either data or quantaties being set!")

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

        if axes is _extract_from_backend:
            if data.dtype == object:
                iteration = Axis(
                    data.item().iteration,
                    name="iteration",
                    label="iteration",
                    unit="",
                )
                time = Axis(
                    data.item().time_step,
                    name="time",
                    label="time",
                    unit=data.item().time_unit,
                )
            else:
                iteration = None
                time = None

            axes = {"time": time, "iteration": iteration}

        self._axes = axes

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
                        dtype=data.item().dtype,
                    )

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

        self._quantaties = quantities

        if data.dtype == object:
            self._num_particles = Axis(
                data.item().num_particles,
                name="num_particles",
                label="num. of particles",
                unit="",
            )
        else:
            self._num_particles = Axis(
                data.shape[1],
                name="num_particles",
                label="num. of particles",
                unit="",
            )

    def __getitem__(self, key: str) -> ParticleQuantity:
        return self.quantities[key]

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantities(self) -> Dict[str, QuantityType]:
        return self._quantaties

    @property
    def axes(self) -> ParticleDatasetAxes:
        return self._axes

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


class DatasetCollection:
    pass


# > PREVIOUS VERSION OF CONTAINERS
# from pathlib import Path
# from typing import Any
# from typing import Dict
# from typing import List
# from typing import Optional
# from typing import Set
# from typing import Tuple
# from typing import TypeVar
# from typing import Union

# import attr
# import numpy as np
# from attr import converters
# from attr.validators import deep_iterable
# from attr.validators import instance_of
# from attr.validators import optional

# from nata.backends.grid import GridArray
# from nata.backends.grid import GridBackend
# from nata.backends.particles import ParticleArray
# from nata.backends.particles import ParticleBackend
# from nata.utils.attrs import attrib_equality
# from nata.utils.attrs import location_exists
# from nata.utils.attrs import subdtype_of

# from .axes import GridAxis
# from .axes import IterationAxis
# from .axes import ParticleQuantity
# from .axes import TimeAxis
# from .axes import UnnamedAxis
# from .utils.exceptions import NataInvalidContainer
# from .utils.formatting import make_as_identifier


# class BaseDataset:
#     _backends: Set[Any] = set()
#     appendable = False

#     @classmethod
#     def register_plugin(cls, plugin_name, plugin):
#         setattr(cls, plugin_name, plugin)

#     @classmethod
#     def add_backend(cls, backend):
#         cls._backends.add(backend)

#     def _convert_to_backend(self, obj):
#         for backend in self._backends:
#             if backend.is_valid_backend(obj):
#                 return backend(obj)

#         raise NataInvalidContainer(
#             f"Unable to find proper backend for {type(obj)}"
#         )

#     def _check_dataset_equality(self, other):
#         raise NotImplementedError

#     def _check_appendability(self, other: "BaseDataset"):
#         if not self.appendable:
#             raise TypeError(f"'{self.__class__}' is not appendable")

#         if not isinstance(other, self.__class__):
#             raise TypeError(
#                 f"Can not append '{type(other)}' to '{self.__class__}'"
#             )

#         if not self._check_dataset_equality(other):
#             raise ValueError(f"{other} can not be appended")


# GridBackendBased = TypeVar("GridBackendBased", str, Path, GridBackend)


# def _asanyarray_converter(data):
#     data = np.asanyarray(data)

#     if not data.ndim:
#         data = data.reshape((1,))

#     return data


# def _deduct_prop_from_data(data, deduced_prop, default=None):
#     prop_store = set()
#     err_msg = f"Mixed data provided for property '{deduced_prop}'"

#     for d in data:
#         prop_store |= set((getattr(d, deduced_prop, None),))

#     if len(prop_store) not in (1, 2):
#         raise ValueError(err_msg)

#     if len(prop_store) == 2:
#         try:
#             prop_store.remove(None)
#         except KeyError:
#             raise ValueError(err_msg)

#     prop = prop_store.pop()
#     return prop if prop is not None else default


# def _generate_axes_list(data):
#     axes_names = tuple()
#     axes_labels = tuple()
#     axes_units = tuple()
#     axes_lengths = tuple()

#     axes_min = []
#     axes_max = []

#     for d in data:
#         if not hasattr(d, "axes_names"):
#             continue

#         names = tuple(d.axes_names)
#         labels = tuple(d.axes_labels)
#         units = tuple(d.axes_units)
#         lengths = d.shape

#         axes_min.append(getattr(d, "axes_min", None))
#         axes_max.append(getattr(d, "axes_max", None))

#         if any(
#            tuple() == v
#            for v in (axes_names, axes_labels, units, lengths)
#         ):
#             axes_names = names
#             axes_labels = labels
#             axes_units = units
#             axes_lengths = lengths
#         else:
#             if (names, labels, units, lengths) != (
#                 axes_names,
#                 axes_labels,
#                 axes_units,
#                 axes_lengths,
#             ):
#                 raise ValueError("Mismatch between axes props in data")

#     axes = []
#     axes_min = np.transpose(axes_min)
#     axes_max = np.transpose(axes_max)

#     for name, label, unit, min_, max_, l in zip(
#         axes_names, axes_labels, axes_units, axes_min, axes_max, axes_lengths
#     ):
#         axes.append(
#             GridAxis(
#                 min=min_,
#                 max=max_,
#                 axis_length=l,
#                 name=name,
#                 label=label,
#                 unit=unit,
#             )
#         )

#     return axes


# @attr.s(eq=False)
# class GridDataset(BaseDataset):
#     """Container class storing grid datasets"""

#     _backends: Set[GridBackend] = set((GridArray,))
#     appendable = True

#     _data: np.ndarray = attr.ib(converter=_asanyarray_converter, repr=False)
#     _shape: Tuple[int] = attr.ib(
#         default=None,
#         validator=optional(
#             deep_iterable(
#                 member_validator=instance_of(int),
#                 iterable_validator=instance_of(tuple),
#             )
#         ),
#     )
#     dtype: np.dtype = attr.ib(
#         default=None, validator=optional(instance_of(np.dtype))
#     )
#     backend: Optional[str] = attr.ib(
#         default=None, validator=optional(subdtype_of(np.str_))
#     )

#     name: str = attr.ib(
#         default=None,
#         converter=converters.optional(make_as_identifier),
#         validator=optional(subdtype_of(np.str_)),
#     )
#     label: str =
#           attr.ib(default=None, validator=optional(subdtype_of(np.str_)))
#     unit: str =
#           attr.ib(default=None, validator=optional(subdtype_of(np.str_)))
#     iteration: IterationAxis = attr.ib(
#         default=None, validator=optional(instance_of(IterationAxis))
#     )
#     time: TimeAxis = attr.ib(
#         default=None, validator=optional(instance_of(TimeAxis))
#     )

#     axes: List[Optional[GridAxis]] = attr.ib(
#         default=None,
#         validator=optional(
#             deep_iterable(
#                 member_validator=optional(instance_of(GridAxis)),
#                 iterable_validator=instance_of(list),
#             )
#         ),
#     )

#     # TODO: add dimensionality of the dataset

#     def __attrs_post_init__(self):
#         for i, d in enumerate(self._data):
#             if not isinstance(d, (np.ndarray, GridBackend)):
#                 self._data[i] = self._convert_to_backend(d)

#         self.dtype = (
#             self.dtype
#             if self.dtype is not None
#             else _deduct_prop_from_data(self._data, "dtype")
#         )
#         self._shape = (
#             self._shape
#             if self._shape is not None
#             else _deduct_prop_from_data(self._data, "shape")
#         )

#         self.name = (
#             self.name
#             if self.name is not None
#             else _deduct_prop_from_data(self._data, "dataset_name")
#         )
#         self.name = make_as_identifier(self.name)

#         self.label = (
#             self.label
#             if self.label is not None
#             else _deduct_prop_from_data(self._data, "dataset_label")
#         )
#         self.unit = (
#             self.unit
#             if self.unit is not None
#             else _deduct_prop_from_data(self._data, "dataset_unit")
#         )

#         self.iteration = (
#             self.iteration
#             if self.iteration is not None
#             else IterationAxis(
#                 data=_deduct_prop_from_data(self._data, "iteration")
#             )
#         )
#         self.time = (
#             self.time
#             if self.time is not None
#             else TimeAxis(
#                 data=_deduct_prop_from_data(self._data, "time_step"),
#                 unit=_deduct_prop_from_data(self._data, "time_unit", ""),
#             )
#         )

#         self.axes = (
#             self.axes
#             if self.axes is not None
#             else _generate_axes_list(self._data)
#         )

#         for axis in self.axes:
#             if axis.name == "time":
#                 raise NotImplementedError(
#                     "Grid axes named `time` are not yet supported"
#                 )
#             setattr(self, axis.name, axis)

#         # cross validate
#           - just for the ake for safety
#           - we can remove later on attr.validate(self)

#     def __iter__(self):
#         if len(self) == 1:
#             yield self
#         else:
#             for d, it, t in zip(self._data, self.iteration, self.time):
#                 yield self.__class__(
#                     data=[d],
#                     dtype=self.dtype,
#                     backend=self.backend,
#                     name=self.name,
#                     label=self.label,
#                     unit=self.unit,
#                     iteration=it,
#                     time=t,
#                     axes=self.axes,
#                 )

#     def __array__(self, dtype=None):
#         if self._data.dtype == object:
#             data = np.empty((len(self),) + self._shape, dtype=self.dtype)
#             for i, d in enumerate(self._data):
#                 if isinstance(d, np.ndarray):
#                     data[i] = d
#                 else:
#                     data[i] = d.get_data(indexing=None)

#             self._data = data

#         return np.squeeze(self._data)

#     def __len__(self):
#         return len(self._data)

#     def __getitem__(self, key):
#         if isinstance(key, int):
#             # new data
#             new_data = self._data[key]
#             # new iteration
#             new_iteration = IterationAxis(
#                 data=self.iteration.data[key],
#                 name=self.iteration.name,
#                 label=self.iteration.label,
#                 unit=self.iteration.unit,
#             )
#             # new time
#             new_time = TimeAxis(
#                 data=self.time.data[key],
#                 name=self.time.name,
#                 label=self.time.label,
#                 unit=self.time.unit,
#             )
#             # new axes
#             new_grid_axes = []

#             for axis in self.axes:
#                 new_grid_axes.append(
#                     GridAxis(
#                         min=axis.data[key, 0],
#                         max=axis.data[key, 1],
#                         axis_length=axis.axis_length,
#                         axis_type=axis.axis_type,
#                         name=axis.name,
#                         label=axis.label,
#                         unit=axis.unit,
#                     )
#                 )

#             return self.__class__(
#                 data=new_data,
#                 dtype=self.dtype,
#                 backend=self.backend,
#                 name=self.name,
#                 label=self.label,
#                 unit=self.unit,
#                 iteration=new_iteration,
#                 time=new_time,
#                 axes=new_grid_axes,
#             )

#         else:
#             return self.__array__()[key]

#     def info(self, full: bool = False):  # pragma: no cover
#         return self.__repr__()

#     @property
#     def shape(self):
#         if self._data.dtype == object:
#             if len(self) == 1:
#                 return self._shape
#             else:
#                 return (len(self),) + self._shape
#         else:
#             if len(self) == 1:
#                 return self._data.shape[1:]
#             else:
#                 return self._data.shape

#     @property
#     def ndim(self):
#         return len(self.shape)

#     @property
#     def grid_dim(self):
#         return len(self.axes)

#     @classmethod
#     def from_array(
#         cls,
#         array,
#         name=None,
#         label=None,
#         unit=None,
#         iteration=None,
#         time=None,
#         time_unit=None,
#         axes_names=None,
#         axes_min=None,
#         axes_max=None,
#         axes_labels=None,
#         axes_units=None,
#         slice_array=True,
#     ):
#         try:
#             time_iter = iter(time)
#         except TypeError:
#             time_iter = None

#         try:
#             iteration_iter = iter(iteration)
#         except TypeError:
#             iteration_iter = None

#         if (time_iter is None) and (iteration_iter is None):
#             if iteration is None:
#                 iteration = 0
#             if time is None:
#                 time = 0.0

#             arr_backend = GridArray(
#                 array=array,
#                 dataset_name=name,
#                 dataset_label=label,
#                 dataset_unit=unit,
#                 iteration=iteration,
#                 time_step=time,
#                 time_unit=time_unit,
#                 axes_names=axes_names,
#                 axes_min=axes_min,
#                 axes_max=axes_max,
#                 axes_labels=axes_labels,
#                 axes_units=axes_units,
#             )

#             return cls(arr_backend)

#         if len(time) != len(iteration):
#             raise ValueError(
#                 "mismatched in length between parameters time and iteration"
#             )

#         if slice_array:
#             if array.shape[0] != len(time):
#                 raise ValueError("mismatch in iteration and array dimension")
#             arr_index = 0
#         else:
#             arr_index = slice(None)

#         t = next(time_iter)
#         it = next(iteration_iter)

#         ds = cls(
#             GridArray(
#                 array=array[arr_index],
#                 dataset_name=name,
#                 dataset_label=label,
#                 dataset_unit=unit,
#                 iteration=it,
#                 time_step=t,
#                 time_unit=time_unit,
#                 axes_names=axes_names,
#                 axes_min=axes_min,
#                 axes_max=axes_max,
#                 axes_labels=axes_labels,
#                 axes_units=axes_units,
#             )
#         )
#         for i, (t, it) in enumerate(zip(time_iter, iteration_iter), start=1):
#             if slice_array:
#                 arr_index = i
#             else:
#                 arr_index = slice(None)

#             ds.append(
#                 cls(
#                     GridArray(
#                         array=array[arr_index],
#                         dataset_name=name,
#                         dataset_label=label,
#                         dataset_unit=unit,
#                         iteration=it,
#                         time_step=t,
#                         time_unit=time_unit,
#                         axes_names=axes_names,
#                         axes_min=axes_min,
#                         axes_max=axes_max,
#                         axes_labels=axes_labels,
#                         axes_units=axes_units,
#                         keep_creation_count=True,
#                     )
#                 )
#             )

#         return ds

#     def _check_dataset_equality(self, other: Union["GridDataset", Any]):
#         if not isinstance(other, self.__class__):
#             return False

#         if not attrib_equality(self, other, "name, label, unit"):
#             return False

#         if self.iteration != other.iteration and self.time != other.time:
#             return False

#         for axis in self.axes:
#             if not hasattr(other, axis.name):
#                 return False

#             if axis != getattr(other, axis.name):
#                 return False

#         return True

#     def append(self, other: "GridDataset"):
#         self._check_appendability(other)

#         self._data = np.array(
#             [d for d in self._data] + [d for d in other._data]
#         )

#         self.iteration.append(other.iteration)
#         self.time.append(other.time)
#         for axis, other_axis in zip(self.axes, other.axes):
#             axis.append(other_axis)


# ParticleBackendBased = TypeVar(
#     "ParticleBackendBased", str, Path, ParticleBackend
# )


# def _generate_quantities_list(data: List[Union[ParticleBackend, np.ndarray]]):
#     quantity_names = tuple()
#     quantity_labels = tuple()
#     quantity_units = tuple()
#     quantity_dtype = None

#     number_particles = []

#     for d in data:
#         if not hasattr(d, "quantities"):
#             continue

#         names = tuple(d.quantities)
#         labels = tuple(d.quantity_labels)
#         units = tuple(d.quantity_units)
#         dtype = d.dtype

#         number_particles.append(d.num_particles)

#         if (
#             any(
#                 tuple() == v
#                 for v in (quantity_names, quantity_labels, quantity_units,)
#             )
#             and quantity_dtype is None
#         ):
#             quantity_names = names
#             quantity_labels = labels
#             quantity_units = units
#             quantity_dtype = dtype
#         else:
#             if (names, labels, units, dtype) != (
#                 quantity_names,
#                 quantity_labels,
#                 quantity_units,
#                 quantity_dtype,
#             ):
#                 raise ValueError("Mismatch between quantity props in data")

#     quantities = []

#     for name, label, unit in zip(
#         quantity_names, quantity_labels, quantity_units
#     ):
#         quantities.append(
#             ParticleQuantity(
#                 name=name,
#                 label=label,
#                 unit=unit,
#                 data=data,
#                 prt_num=number_particles,
#                 dtype=quantity_dtype[name],
#             )
#         )

#     return quantities


# @attr.s(eq=False)
# class ParticleDataset(BaseDataset):
#     """Container class storing particle datasets"""

#     _backends: Set[ParticleBackend] = set((ParticleArray,))
#     appendable = True

#     _data: np.ndarray = attr.ib(
#         converter=_asanyarray_converter, repr=False, eq=False
#     )
#     backend: Optional[str] = attr.ib(
#         default=None, validator=optional(subdtype_of(np.str_))
#     )

#     name: str = attr.ib(
#         default=None,
#         converter=converters.optional(make_as_identifier),
#         validator=optional(subdtype_of(np.str_)),
#     )

#     iteration: IterationAxis = attr.ib(
#         default=None, validator=optional(instance_of(IterationAxis))
#     )
#     time: TimeAxis = attr.ib(
#         default=None, validator=optional(instance_of(TimeAxis))
#     )

#     num_particles: UnnamedAxis = attr.ib(
#         default=None, validator=optional(instance_of(UnnamedAxis))
#     )
#     quantities: List[Optional[ParticleQuantity]] = attr.ib(
#         default=None,
#         validator=optional(
#             deep_iterable(
#                 member_validator=optional(instance_of(ParticleQuantity)),
#                 iterable_validator=instance_of(list),
#             )
#         ),
#     )

#     def info(self, full: bool = False):  # pragma: no cover
#         return self.__repr__()

#     def __attrs_post_init__(self):
#         for i, d in enumerate(self._data):
#             if not isinstance(d, (np.ndarray, ParticleBackend)):
#                 self._data[i] = self._convert_to_backend(d)

#         self.backend = (
#             self.backend
#             if self.backend is not None
#             else _deduct_prop_from_data(self._data, "name")
#         )

#         self.name = (
#             self.name
#             if self.name is not None
#             else _deduct_prop_from_data(self._data, "dataset_name")
#         )
#         self.name = make_as_identifier(self.name)

#         self.iteration = (
#             self.iteration
#             if self.iteration is not None
#             else IterationAxis(
#                 data=_deduct_prop_from_data(self._data, "iteration")
#             )
#         )
#         self.time = (
#             self.time
#             if self.time is not None
#             else TimeAxis(
#                 data=_deduct_prop_from_data(self._data, "time_step"),
#                 unit=_deduct_prop_from_data(self._data, "time_unit", ""),
#             )
#         )

#         self.num_particles = (
#             self.num_particles
#             if self.num_particles is not None
#             else UnnamedAxis(
#                 data=_deduct_prop_from_data(self._data, "num_particles")
#             )
#         )

#         self.quantities = (
#             self.quantities
#             if self.quantities is not None
#             else _generate_quantities_list(self._data)
#         )

#         for quant in self.quantities:
#             setattr(self, quant.name, quant)

#         attr.validate(self)

#     def __len__(self):
#         return len(self.iteration)

#     def __getitem__(self, key):
#         if isinstance(key, int):
#             new_data = self._data[key]

#             new_iteration = IterationAxis(
#                 data=self.iteration.data[key],
#                 name=self.iteration.name,
#                 label=self.iteration.label,
#                 unit=self.iteration.unit,
#             )

#             new_time = TimeAxis(
#                 data=self.time.data[key],
#                 name=self.time.name,
#                 label=self.time.label,
#                 unit=self.time.unit,
#             )

#             new_num_particles = UnnamedAxis(data=self.num_particles.data[key])

#             new_quants = []

#             for quant in self.quantities:
#                 new_quants.append(
#                     ParticleQuantity(
#                         data=[quant.data[key]],
#                         dtype=quant.dtype,
#                         prt_num=[quant.prt_num[key]],
#                         name=quant.name,
#                         label=quant.label,
#                         unit=quant.unit,
#                     )
#                 )

#             return self.__class__(
#                 data=[new_data],
#                 backend=self.backend,
#                 name=self.name,
#                 iteration=new_iteration,
#                 time=new_time,
#                 num_particles=new_num_particles,
#                 quantities=new_quants,
#             )
#         else:
#             raise NotImplementedError("not yet fully implemented")

#     def _check_dataset_equality(self, other: Union["ParticleDataset", Any]):
#         if not isinstance(other, self.__class__):
#             return False

#         if not attrib_equality(self, other):
#             return False

#         return True

#     def append(self, other: "ParticleDataset"):
#         self._check_appendability(other)

#         self._data = np.array(
#             [d for d in self._data] + [d for d in other._data]
#         )

#         self.num_particles.append(other.num_particles)
#         self.iteration.append(other.iteration)
#         self.time.append(other.time)
#         for self_quant, other_quant in zip(self.quantities, other.quantities):
#             self_quant.append(other_quant)


# DatasetTypes = TypeVar("DatasetTypes", GridDataset, ParticleDataset)


# @attr.s
# class DatasetCollection:
#     root_path: Path = attr.ib(converter=Path, validator=location_exists)
#     _container_set: Set[DatasetTypes] = set([GridDataset, ParticleDataset])
#     store: Dict[str, DatasetTypes] = attr.ib(factory=dict)

#     def info(self, full: bool = False):  # pragma: no cover
#         return self.__repr__()

#     @property
#     def datasets(self):
#         return np.array([k for k in self.store.keys()], dtype=str)

#     def _append_datasetcollection(self, obj):
#         self.store.update(obj.store)

#     def _append_file(self, obj):
#         for container in self._container_set:
#             try:
#                 dataset = container(obj)
#                 break
#             except NataInvalidContainer:
#                 continue
#         else:
#             # not possible to append the file -> not a valid container found
#             return

#         if dataset.name in self.store:
#             existing_ds = self.store[dataset.name]

#             if existing_ds.appendable:
#                 existing_ds.append(dataset)
#             else:
#                 raise ValueError(
#                     f"Dataset '{existing_ds.name}' is not appendable!"
#                 )
#         else:
#             self.store[dataset.name] = dataset

#     def append(self, obj: Union[str, Path, "DatasetCollection"]) -> None:
#         """Takes a path to a diagnostic and appends it to the collection."""
#         if isinstance(obj, DatasetCollection):
#             self._append_datasetcollection(obj)

#         elif isinstance(obj, (str, Path)):
#             self._append_file(obj)

#         else:
#             raise ValueError(
#                 f"Can not append object of type '{type(obj)}' to collection"
#             )

#     def __getitem__(self, key):
#         return self.store[key]

#     @classmethod
#     def register_plugin(cls, plugin_name, plugin):
#         setattr(cls, plugin_name, plugin)
