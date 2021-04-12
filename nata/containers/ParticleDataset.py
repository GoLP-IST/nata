# -*- coding: utf-8 -*-
from pathlib import Path
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.lib import recfunctions as rfn

from nata.types import AxisType
from nata.types import DatasetType
from nata.types import ParticleBackendType
from nata.types import ParticleDatasetAxes
from nata.types import QuantityType
from nata.types import is_basic_indexing
from nata.utils.exceptions import NataInvalidContainer
from nata.utils.formatting import make_identifiable

from .axis import Axis

_extract_from_backend = object()
_extract_from_data = object()


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
            raise ValueError("Invalid name provided! Has to be able to be valid code")
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
            raise ValueError(f"Mismatch in attributes between '{self}' and '{other}'")

        self._data = np.append(self._data, other._data, axis=0)
        self._num_prt = np.append(self._num_prt, other._num_prt)


class ParticleDataset:
    _backends: AbstractSet[ParticleBackendType] = set()

    def __init__(
        self,
        data: Optional[Union[ParticleBackendType, np.ndarray, str, Path]] = None,
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

        if data.dtype == object:
            self._backend = data.item().name
        else:
            self._backend = None

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
            num_particles = np.full(shape=len(q), fill_value=q.num_particles, dtype=int)

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
            self.axes["iteration"][key] if key is not None else self.axes["iteration"]
        )
        quantities = {quant.name: quant[key] for quant in self.quantities.values()}

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

    @property
    def backend(self) -> Optional[str]:
        """Backend associated with instance."""
        return self._backend

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
            raise ValueError(f"Can not append '{other}' particle datasets are unequal!")

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
            if not self.quantities[quant_name].equivalent(other.quantities[quant_name]):
                return False

        if not self.num_particles.equivalent(other.num_particles):
            return False

        return True

    @classmethod
    def register_plugin(cls, plugin_name, plugin):
        setattr(cls, plugin_name, plugin)
